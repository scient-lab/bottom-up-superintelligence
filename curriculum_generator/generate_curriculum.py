'''
Copyright (c) 2024 The Trustees of Princeton University
Authors: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha

Licensed for academic and research use only.
See LICENSE file for full terms.
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curriculum_generator.generate_questions import QAGenerator
import json
from typing import List
import os
import time
from tqdm import tqdm
from datetime import datetime
from collections import deque
import argparse
import random
    

def generate_curriculum(
    max_k_hops: int,
    num_questions: int,
    output_dir: str,
    api_key: str,
    # kg-pipeline fork-patch: error_threshold default bumped 10 → 50 because
    # specialty-density KGs (e.g. topic-curated diabetes 300) trigger the
    # quality-filter "Options are near duplicates" rejection at ~35% rate —
    # distractors from a narrow KG vocabulary look similar to the answer.
    # Also see semantics change below: error_window now resets on success
    # so it's "consecutive errors" not "total errors over the run."
    # Honor BUS_ERROR_THRESHOLD env var for further tuning.
    error_threshold: int = int(os.environ.get("BUS_ERROR_THRESHOLD", "50")),
    # kg-pipeline fork-patch: the original default of 60 was set for free-tier
    # Gemini (10 RPM). On paid tier (1000+ RPM) this is ~120x overkill and
    # makes a 100-question run take ~2 hours instead of ~15 minutes. Default
    # to 5; honor BUS_PER_Q_SLEEP env var for further tuning.
    sleep_threshold: int = int(os.environ.get("BUS_PER_Q_SLEEP", "5")),
    ) -> None:
    """
    Generate a dataset of medical questions, explanations, and answers from the UMLS KG and save them locally.
    
    Args:
        max_k_hops: Maximum hop length for path generation
        num_questions: Number of questions to generate
        api_key: API key for the LLM
        output_dir: Directory to save the generated questions
        error_threshold: Number of errors to tolerate before restarting
        sleep_threshold: Number of seconds to sleep before restarting to prevent being rate limitied by API
    """
    
    output_file = os.path.join(output_dir, f"curriculum_dataset_hop_{max_k_hops}.json")
    dataset = []
    question_idx = 0
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            dataset = json.load(f)
            if dataset:
                question_idx = max(entry['id'] for entry in dataset) + 1
                print(f"Resuming from question {question_idx}+1")
    
    qa_gym = QAGenerator(api_key=api_key)
    error_window = 0

    while question_idx < num_questions:
        
        if error_window > error_threshold:
            raise Exception("Too many errors. Stopping script.")
        
        try:
            k_hops = random.randint(1, max_k_hops)
            # Generate question
            question_data = qa_gym.generate_questions( 
                k_hops=k_hops
            )
            
            quality = qa_gym.quality_filtering(question_data['question'])
            if not quality:
                raise Exception("Quality filtering failed. Skipping question.")
            
            explanation = qa_gym.llm.generate_thinking_trace(
                question=question_data['question'],
                paths=question_data['paths']
            )
            
            combined = qa_gym.llm.combine_question_and_thinking_trace_with_answer(
                question=question_data['question'],
                explanation=explanation,
                answer=question_data['answer']
            )
            correctness = qa_gym.correctness_filtering(combined, question_data['paths'])
            if not correctness:
                raise Exception("Correctness filtering failed. Skipping question.")
            
            question_entry = {
                "id": question_idx,
                "k_hops": k_hops,
                "source_concept": question_data['source_concept'],
                "target_concept": question_data['target_concept'],
                "paths": question_data['paths'],
                "question_and_explanation": combined
            } 
            
            dataset.append(question_entry)
            
            if (question_idx + 1) % 5 == 0:
                print(f"Checkpoint: Saved {question_idx+1} questions to {output_file}")
                output_file = os.path.join(output_dir, f"curriculum_dataset_hop_{max_k_hops}.json")
                with open(output_file, 'w') as f:
                    json.dump(dataset, f, indent=2)
                with open(os.path.join(output_dir, f"nodes_freq.json"), 'w') as f:
                    json.dump(qa_gym.generator.vocab_freq, f, indent=2)
                    
            question_idx += 1
            # kg-pipeline fork-patch: reset on success so error_window measures
            # *consecutive* errors, not total over the run. Without this, a
            # specialty-domain run accumulates ~35% rejects over many successes
            # and aborts even when it's making steady progress.
            error_window = 0
            for path in question_data['paths']:
                qa_gym.generator.vocab_freq[path['start']] += 1
                
            time.sleep(sleep_threshold)
            
        except Exception as e:
            error_window += 1
            print(f"Error generating question {e}")
            continue
    
    # Final save
    output_file = os.path.join(output_dir, f"curriculum_dataset_hop_{max_k_hops}.json")
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} questions. Dataset saved to {output_file}")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_k_hops", type=int, default=3)
    parser.add_argument("--num_questions", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="/curriculum_training_data/")
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()
    
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    
    generate_curriculum(args.max_k_hops, args.num_questions, args.output_dir, args.api_key)

    

if __name__ == "__main__":
    main() 