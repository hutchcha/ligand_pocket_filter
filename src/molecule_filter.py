import argparse
import logging
from typing import Tuple, List, Dict

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem import QED

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MoleculeFilter:
    def __init__(self,
                 mw_range: Tuple[float, float] = (150, 500),
                 hba_range: Tuple[int, int] = (0, 10),
                 hbd_range: Tuple[int, int] = (0, 5),
                 tpsa_range: Tuple[float, float] = (0, 200),
                 logp_range: Tuple[float, float] = (-5, 10),
                 clogS_range: Tuple[float, float] = (-5, 10),  # Placeholder
                 rotb_range: Tuple[int, int] = (0, 15),
                 drug_score_range: Tuple[float, float] = (0, 1),
                 complexity_range: Tuple[int, int] = (10, 100)
                ):
        """
        Initialize the MoleculeFilter with specified thresholds.
        """
        self.mw_min, self.mw_max = mw_range
        self.hba_min, self.hba_max = hba_range
        self.hbd_min, self.hbd_max = hbd_range
        self.tpsa_min, self.tpsa_max = tpsa_range
        self.logp_min, self.logp_max = logp_range
        self.clogS_min, self.clogS_max = clogS_range
        self.rotb_min, self.rotb_max = rotb_range
        self.drug_score_min, self.drug_score_max = drug_score_range
        self.complexity_min, self.complexity_max = complexity_range
        
        # Predefined lists of polar and nonpolar residues
        self.POLAR_RESIDUES = {
            'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
            'HIS', 'LYS', 'SER', 'THR', 'TYR', 'TRP'
        }
        
        self.NONPOLAR_RESIDUES = {
            'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE',
            'PRO', 'GLY', 'TRP', 'TYR', 'CYS'
        }
    
    def calculate_drug_score(self, mol: Chem.Mol) -> float:
        """
        Calculate the drug score using RDKit's QED as a proxy.
        QED is a measure of drug-likeness.
        """
        try:
            qed_score = QED.qed(mol)
            return qed_score
        except:
            return 0.0  # Return a default score if calculation fails
    
    def calculate_clogS(self, mol: Chem.Mol) -> float:
        """
        Placeholder for cLogS calculation.
        RDKit does not have a built-in method to calculate cLogS.
        Implementing a full solubility model is beyond this scope.
        """
        # Implement a simple estimation or use a placeholder
        # For demonstration, we'll use a basic formula based on TPSA and logP
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        logp = Crippen.MolLogP(mol)
        clogS = 0.105 * tpsa - 0.88 * logp  # Example coefficients
        return clogS
    
    def calculate_complexity(self, mol: Chem.Mol) -> int:
        """
        Calculate molecular complexity.
        Here, we define it as the number of unique atom types.
        """
        atom_types = set([atom.GetSymbol() for atom in mol.GetAtoms()])
        return len(atom_types)
    
    def compute_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Compute all necessary descriptors for the molecule.
        """
        descriptors = {}
        descriptors['MolecularWeight'] = Descriptors.MolWt(mol)
        descriptors['HBA'] = Descriptors.NumHAcceptors(mol)
        descriptors['HBD'] = Descriptors.NumHDonors(mol)
        descriptors['TPSA'] = rdMolDescriptors.CalcTPSA(mol)
        descriptors['LogP'] = Crippen.MolLogP(mol)
        descriptors['cLogS'] = self.calculate_clogS(mol)
        descriptors['RotatableBonds'] = Descriptors.NumRotatableBonds(mol)
        descriptors['DrugScore'] = self.calculate_drug_score(mol)
        descriptors['MolecularComplexity'] = self.calculate_complexity(mol)
        return descriptors
    
    def filter_molecule(self, smiles: str) -> Tuple[bool, Dict[str, float]]:
        """
        Determine whether a molecule meets the filtering criteria.
        
        Parameters:
            smiles (str): SMILES string of the molecule.
        
        Returns:
            Tuple[bool, Dict[str, float]]: 
                - Boolean indicating if the molecule passes the filters.
                - Dictionary of computed descriptors.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"Invalid SMILES string: {smiles}")
            return False, {}
        
        descriptors = self.compute_descriptors(mol)
        
        # Apply filtering criteria
        if not (self.mw_min <= descriptors['MolecularWeight'] <= self.mw_max):
            return False, descriptors
        if not (self.hba_min <= descriptors['HBA'] <= self.hba_max):
            return False, descriptors
        if not (self.hbd_min <= descriptors['HBD'] <= self.hbd_max):
            return False, descriptors
        if not (self.tpsa_min <= descriptors['TPSA'] <= self.tpsa_max):
            return False, descriptors
        if not (self.logp_min <= descriptors['LogP'] <= self.logp_max):
            return False, descriptors
        if not (self.clogS_min <= descriptors['cLogS'] <= self.clogS_max):
            return False, descriptors
        if not (self.rotb_min <= descriptors['RotatableBonds'] <= self.rotb_max):
            return False, descriptors
        if not (self.drug_score_min <= descriptors['DrugScore'] <= self.drug_score_max):
            return False, descriptors
        if not (self.complexity_min <= descriptors['MolecularComplexity'] <= self.complexity_max):
            return False, descriptors
        
        return True, descriptors

def parse_arguments():
    """
    Parse command-line arguments for the filtering module.
    
    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Filter molecules based on physicochemical parameters."
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input SMILES file (e.g., .smi, .csv). Each line should contain a SMILES string."
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output file to save filtered SMILES (e.g., filtered.csv)."
    )
    
    # Optional: Allow users to specify thresholds via command-line
    parser.add_argument(
        "--mw_min",
        type=float,
        default=150,
        help="Minimum molecular weight (default: 150)."
    )
    
    parser.add_argument(
        "--mw_max",
        type=float,
        default=500,
        help="Maximum molecular weight (default: 500)."
    )
    
    parser.add_argument(
        "--hba_min",
        type=int,
        default=0,
        help="Minimum number of hydrogen bond acceptors (default: 0)."
    )
    
    parser.add_argument(
        "--hba_max",
        type=int,
        default=10,
        help="Maximum number of hydrogen bond acceptors (default: 10)."
    )
    
    parser.add_argument(
        "--hbd_min",
        type=int,
        default=0,
        help="Minimum number of hydrogen bond donors (default: 0)."
    )
    
    parser.add_argument(
        "--hbd_max",
        type=int,
        default=5,
        help="Maximum number of hydrogen bond donors (default: 5)."
    )
    
    parser.add_argument(
        "--tpsa_min",
        type=float,
        default=0,
        help="Minimum Topological Polar Surface Area (TPSA) (default: 0)."
    )
    
    parser.add_argument(
        "--tpsa_max",
        type=float,
        default=200,
        help="Maximum Topological Polar Surface Area (TPSA) (default: 200)."
    )
    
    parser.add_argument(
        "--logp_min",
        type=float,
        default=-5,
        help="Minimum logP (default: -5)."
    )
    
    parser.add_argument(
        "--logp_max",
        type=float,
        default=10,
        help="Maximum logP (default: 10)."
    )
    
    parser.add_argument(
        "--clogS_min",
        type=float,
        default=-5,
        help="Minimum calculated logS (cLogS) (default: -5)."
    )
    
    parser.add_argument(
        "--clogS_max",
        type=float,
        default=10,
        help="Maximum calculated logS (cLogS) (default: 10)."
    )
    
    parser.add_argument(
        "--rotb_min",
        type=int,
        default=0,
        help="Minimum number of rotatable bonds (default: 0)."
    )
    
    parser.add_argument(
        "--rotb_max",
        type=int,
        default=15,
        help="Maximum number of rotatable bonds (default: 15)."
    )
    
    parser.add_argument(
        "--drugscore_min",
        type=float,
        default=0,
        help="Minimum drug score (QED) (default: 0)."
    )
    
    parser.add_argument(
        "--drugscore_max",
        type=float,
        default=1,
        help="Maximum drug score (QED) (default: 1)."
    )
    
    parser.add_argument(
        "--complexity_min",
        type=int,
        default=10,
        help="Minimum molecular complexity (default: 10)."
    )
    
    parser.add_argument(
        "--complexity_max",
        type=int,
        default=100,
        help="Maximum molecular complexity (default: 100)."
    )
    
    return parser.parse_args()

def load_smiles(input_file: str) -> List[str]:
    """
    Load SMILES strings from an input file.
    
    Args:
        input_file (str): Path to the input SMILES file.
    
    Returns:
        List[str]: List of SMILES strings.
    """
    smiles = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Assuming each line is a SMILES string. Modify if CSV or other format.
                smiles.append(line)
    return smiles

def save_filtered_smiles(filtered: List[Dict[str, float]], output_file: str):
    """
    Save filtered SMILES and their descriptors to a CSV file.
    
    Args:
        filtered (List[Dict[str, float]]): List of dictionaries containing SMILES and descriptors.
        output_file (str): Path to the output CSV file.
    """
    import csv
    
    if not filtered:
        logging.info("No molecules passed the filtering criteria.")
        return
    
    # Extract fieldnames from the first entry
    fieldnames = filtered[0].keys()
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for molecule in filtered:
            writer.writerow(molecule)
    
    logging.info(f"Filtered molecules saved to {output_file}")

def main():
    args = parse_arguments()
    
    # Initialize the MoleculeFilter with user-defined thresholds
    molecule_filter = MoleculeFilter(
        mw_range=(args.mw_min, args.mw_max),
        hba_range=(args.hba_min, args.hba_max),
        hbd_range=(args.hbd_min, args.hbd_max),
        tpsa_range=(args.tpsa_min, args.tpsa_max),
        logp_range=(args.logp_min, args.logp_max),
        clogS_range=(args.clogS_min, args.clogS_max),
        rotb_range=(args.rotb_min, args.rotb_max),
        drug_score_range=(args.drugscore_min, args.drugscore_max),
        complexity_range=(args.complexity_min, args.complexity_max)
    )
    
    # Load SMILES strings
    smiles_list = load_smiles(args.input)
    logging.info(f"Loaded {len(smiles_list)} SMILES from {args.input}")
    
    # Filter molecules
    filtered_molecules = []
    for smi in smiles_list:
        passed, descriptors = molecule_filter.filter_molecule(smi)
        if passed:
            molecule_data = {'SMILES': smi}
            molecule_data.update(descriptors)
            filtered_molecules.append(molecule_data)
    
    logging.info(f"{len(filtered_molecules)} molecules passed the filtering criteria.")
    
    # Save filtered molecules
    save_filtered_smiles(filtered_molecules, args.output)

if __name__ == "__main__":
    main()
