import argparse
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

def process_chunk_with_index(args):
    """
    Worker function to process a chunk of SMILES given by (index, smiles_chunk).
    Returns (index, data).
    """
    idx, smiles_chunk = args
    data = []
    for line in smiles_chunk:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        smiles, zincid = parts[0], parts[1]
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue
        descriptors = {
            'ID': zincid,
            'MolecularWeight': Descriptors.MolWt(mol),
            'RotatableBonds': Descriptors.NumRotatableBonds(mol),
            'TPSA': rdMolDescriptors.CalcTPSA(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'QED': QED.qed(mol)
        }
        data.append(descriptors)
    return idx, data

def compute_descriptors_parallel(smiles_file, num_processes, chunk_size):
    with open(smiles_file, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    tasks_count = num_processes * 4
    lines_per_task = math.ceil(total_lines / tasks_count)
    chunks = [lines[i:i + lines_per_task] for i in range(0, total_lines, lines_per_task)]
    chunk_sizes = [len(c) for c in chunks]

    # Create indexed chunks
    indexed_chunks = list(enumerate(chunks))

    pbar = tqdm(total=total_lines, desc="Processing molecules", unit="mol")

    all_data = [None] * len(chunks)
    with Pool(num_processes) as pool:
        # Process with imap_unordered
        for idx_data in pool.imap_unordered(process_chunk_with_index, indexed_chunks, chunksize=chunk_size):
            idx, data = idx_data
            all_data[idx] = data
            # Update progress by the size of the processed chunk
            pbar.update(chunk_sizes[idx])

    pbar.close()

    # Flatten all_data
    merged_data = [item for sublist in all_data for item in sublist]
    return pd.DataFrame(merged_data)

def plot_histograms(df, output_dir):
    """
    Generate histograms for chemical descriptors.
    """
    os.makedirs(output_dir, exist_ok=True)
   
    descriptors = ['MolecularWeight', 'RotatableBonds', 'TPSA', 'HBA', 'HBD', 'QED']
    
    for desc in descriptors:
        plt.figure(figsize=(8, 6))
        plt.hist(df[desc], bins=50, alpha=0.75)
        plt.title(f"{desc} Distribution")
        plt.xlabel(desc)
        plt.ylabel("Count")
        plt.savefig(f"{output_dir}/{desc}_histogram.png")
        plt.close()
    print("Histograms saved to:", output_dir)

def main():
    parser = argparse.ArgumentParser(description="Assess chemical diversity of a library with improved parallelization and progress tracking.")
    parser.add_argument("--input", required=True, help="Path to the SMILES text file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save histograms.")
    parser.add_argument("--num_processes", type=int, default=cpu_count(),
                        help="Number of processes to use (default: all available cores).")
    parser.add_argument("--chunk_size", type=int, default=1,
                        help="Number of tasks to assign to each process at a time. Smaller = more updates, larger = less overhead. (default: 1)")
    args = parser.parse_args()

    print("Computing descriptors in parallel...")
    df = compute_descriptors_parallel(args.input, args.num_processes, args.chunk_size)
    
    print("\nDescriptor Summary:")
    print(df.describe())
    summary_path = os.path.join(args.output_dir, "descriptor_summary.csv")
    df.describe().to_csv(summary_path)
    print(f"Summary statistics saved to: {summary_path}")

    print("\nGenerating histograms...")
    plot_histograms(df, args.output_dir)

if __name__ == "__main__":
    main()
