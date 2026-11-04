# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""IR2Vec/MIR2Vec Triplet Generator

Generates IR2Vec or MIR2Vec triplets by applying random optimization levels to
LLVM IR files (or processing MIR files) and extracting triplets using llvm-ir2vec.
Automatically generates preprocessed files (entity2id.txt, relation2id.txt, and
train2id.txt) necessary for training IR2Vec or MIR2Vec vocabularies.

Usage:
    For LLVM IR:
        python generateTriplets.py <llvm_build_dir> <num_optimizations> <ll_file_list> <output_dir>

    For Machine IR:
        python generateTriplets.py --mode=mir <llvm_build_dir> <mir_file_list> <output_dir>
"""

import argparse
import logging
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Set, Tuple

# Configuration
OPT_LEVELS = ["O0", "O1", "O2", "O3", "Os", "Oz"]
DEFAULT_MAX_WORKERS = 100

logger = logging.getLogger(__name__)


# TODO: Change this to a dataclass with slots
# when Python 3.10+ is the minimum version
# https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass
class TripletResult:
    """Result from processing a single LLVM IR file"""

    __slots__ = ["triplets", "max_relation"]

    def __init__(self, triplets: Set[str], max_relation: int):
        self.triplets = triplets
        self.max_relation = max_relation


class IR2VecTripletGenerator:
    """Main class for generating IR2Vec or MIR2Vec triplets"""

    def __init__(
        self,
        llvm_build_dir: Path,
        num_optimizations: int,
        output_dir: Path,
        max_workers: int = DEFAULT_MAX_WORKERS,
        mode: str = "llvm",
    ):
        self.llvm_build_dir = llvm_build_dir
        self.num_optimizations = num_optimizations
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.mode = mode  # "llvm" or "mir"

        # Tool paths
        self.opt_binary = os.path.join(llvm_build_dir, "bin", "opt")
        self.ir2vec_binary = os.path.join(llvm_build_dir, "bin", "llvm-ir2vec")

        self._validate_setup()

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_setup(self):
        """Validate that all required tools and paths exist"""
        if not self.llvm_build_dir.exists():
            raise FileNotFoundError(
                f"LLVM build directory not found: {self.llvm_build_dir}"
            )

        if not os.path.isfile(self.opt_binary) or not os.access(
            self.opt_binary, os.X_OK
        ):
            raise FileNotFoundError(
                f"opt binary not found or not executable: {self.opt_binary}"
            )

        if not os.path.isfile(self.ir2vec_binary) or not os.access(
            self.ir2vec_binary, os.X_OK
        ):
            raise FileNotFoundError(
                f"llvm-ir2vec binary not found or not executable: {self.ir2vec_binary}"
            )

        if self.mode not in ["llvm", "mir"]:
            raise ValueError(f"Mode must be 'llvm' or 'mir', got: {self.mode}")

        # For LLVM IR mode, validate optimization count
        if self.mode == "llvm" and not (1 <= self.num_optimizations <= len(OPT_LEVELS)):
            raise ValueError(
                f"Number of optimizations must be between 1-{len(OPT_LEVELS)}"
            )

    def _select_optimization_levels(self) -> List[str]:
        """Select unique random optimization levels"""
        return random.sample(OPT_LEVELS, self.num_optimizations)

    def _process_single_file(self, input_file: Path) -> TripletResult:
        """Process a single LLVM IR or MIR file"""
        all_triplets = set()
        max_relation = 1

        if self.mode == "mir":
            # For MIR files, process directly without optimization
            triplets, file_max_relation = self._run_mir_pipeline(input_file)
            if triplets:
                all_triplets.update(triplets)
                max_relation = max(max_relation, file_max_relation)
                logger.debug(f"Generated {len(triplets)} triplets for {input_file}")
        else:
            # For LLVM IR files, apply multiple optimization levels
            opt_levels = self._select_optimization_levels()
            for opt_level in opt_levels:
                triplets, file_max_relation = self._run_pipeline(input_file, opt_level)
                if triplets:
                    all_triplets.update(triplets)
                    max_relation = max(max_relation, file_max_relation)
                    logger.debug(
                        f"Generated {len(triplets)} triplets for {input_file} with {opt_level}"
                    )

        return TripletResult(all_triplets, max_relation)

    def _run_pipeline(self, input_file: Path, opt_level: str) -> Tuple[Set[str], int]:
        """Run opt | llvm-ir2vec pipeline using subprocess pipes."""
        try:
            # Run opt first
            opt_proc = subprocess.Popen(
                [self.opt_binary, f"-{opt_level}", str(input_file), "-o", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Run llvm-ir2vec with opt's output as input
            ir2vec_proc = subprocess.Popen(
                [self.ir2vec_binary, "triplets", "--mode=llvm", "-", "-o", "-"],
                stdin=opt_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            opt_proc.stdout.close()
            stdout, _ = ir2vec_proc.communicate()
            opt_proc.wait()

            # Check if either process failed
            if opt_proc.returncode != 0 or ir2vec_proc.returncode != 0:
                return set(), 1

            return self._parse_triplet_output(stdout)
        except (subprocess.SubprocessError, OSError):
            return set(), 1

    def _run_mir_pipeline(self, input_file: Path) -> Tuple[Set[str], int]:
        """Run llvm-ir2vec pipeline for MIR files."""
        try:
            # Run llvm-ir2vec directly on MIR file
            result = subprocess.run(
                [
                    self.ir2vec_binary,
                    "triplets",
                    "--mode=mir",
                    str(input_file),
                    "-o",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                return set(), 1

            return self._parse_triplet_output(result.stdout)
        except (subprocess.SubprocessError, OSError):
            return set(), 1

    def _parse_triplet_output(self, output: str) -> Tuple[Set[str], int]:
        """Parse triplet output and extract max relation"""
        if not output.strip():
            return set(), 1

        lines = output.strip().split("\n")
        max_relation = 1

        # Extract max relation from metadata line
        if lines and lines[0].startswith("MAX_RELATION="):
            max_relation = int(lines[0].split("=")[1])
            lines = lines[1:]

        # Remove duplicate triplets by converting to a set
        return set(lines), max_relation

    def generate_triplets(self, file_list: Path) -> None:
        """Main method to generate triplets from a list of LLVM IR or MIR files"""
        # Store file_list_path for later use in entity generation
        self.file_list_path = file_list

        input_files = self._read_file_list(file_list)

        if self.mode == "mir":
            logger.info(
                f"Processing {len(input_files)} MIR files using {self.max_workers} workers"
            )
        else:
            logger.info(
                f"Processing {len(input_files)} files with {self.num_optimizations} "
                f"optimization levels using {self.max_workers} workers"
            )

        all_triplets = set()
        global_max_relation = 1

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, file): file
                for file in input_files
            }

            for future in as_completed(future_to_file):
                try:
                    result = future.result()
                    all_triplets.update(result.triplets)
                    global_max_relation = max(global_max_relation, result.max_relation)
                except (subprocess.SubprocessError, OSError, ValueError) as e:
                    file_path = future_to_file[future]
                    logger.error(f"Error processing {file_path}: {e}")

        self._generate_output_files(all_triplets, global_max_relation)
        logger.info("Processing completed successfully")

    def _read_file_list(self, file_list: Path) -> List[Path]:
        """Read and validate the list of input files"""
        input_files = []
        with open(file_list, "r") as f:
            for line_num, line in enumerate(f, 1):
                if line := line.strip():
                    file_path = Path(line)
                    if file_path.exists():
                        input_files.append(file_path)
                    else:
                        logger.warning(f"File not found (line {line_num}): {file_path}")

        if not input_files:
            raise ValueError("No valid input files found")
        return input_files

    def _generate_output_files(self, all_triplets: Set[str], max_relation: int) -> None:
        """Generate the final output files"""
        logger.info(f"Generating output files with {len(all_triplets)} unique triplets")

        # Write all output files -- train2id.txt, entity2id.txt, relation2id.txt
        train2id_file = os.path.join(self.output_dir, "train2id.txt")
        entity2id_file = os.path.join(self.output_dir, "entity2id.txt")
        relation2id_file = os.path.join(self.output_dir, "relation2id.txt")

        with open(train2id_file, "w") as f:
            f.write(f"{len(all_triplets)}\n")
            f.writelines(f"{triplet}\n" for triplet in all_triplets)

        self._generate_entity2id(entity2id_file)
        self._generate_relation2id(relation2id_file, max_relation)

    def _generate_entity2id(self, output_file: Path) -> None:
        """Generate entity2id.txt using llvm-ir2vec"""
        if self.mode == "mir":
            # For MIR mode, we need to provide a sample MIR file to determine target
            # Use the first file from the processed list
            input_files = self._read_file_list(self.file_list_path)
            if not input_files:
                raise ValueError("No input files available for entity generation")

            subprocess.run(
                [
                    str(self.ir2vec_binary),
                    "entities",
                    "--mode=mir",
                    str(input_files[0]),
                    "-o",
                    str(output_file),
                ],
                check=True,
                capture_output=True,
            )
        else:
            subprocess.run(
                [
                    str(self.ir2vec_binary),
                    "entities",
                    "--mode=llvm",
                    "-o",
                    str(output_file),
                ],
                check=True,
                capture_output=True,
            )

    def _generate_relation2id(self, output_file: Path, max_relation: int) -> None:
        """Generate relation2id.txt from max relation"""
        max_relation = max(max_relation, 1)  # At least Next relation
        num_relations = max_relation + 1

        with open(output_file, "w") as f:
            f.write(f"{num_relations}\n")
            if self.mode == "llvm":
                # LLVM IR has Type relation at 0
                f.write("Type\t0\n")
                f.write("Next\t1\n")
                f.writelines(f"Arg{i-2}\t{i}\n" for i in range(2, num_relations))
            else:
                # MIR doesn't have Type relation, starts with Next at 0
                f.write("Next\t0\n")
                f.writelines(f"Arg{i-1}\t{i}\n" for i in range(1, num_relations))


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate IR2Vec or MIR2Vec triplets from LLVM IR or Machine IR files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "llvm_build_dir", type=Path, help="Path to LLVM build directory"
    )
    parser.add_argument(
        "num_optimizations",
        type=int,
        nargs="?",
        default=1,
        help="Number of optimization levels to apply (1-6) for LLVM IR mode",
    )
    parser.add_argument(
        "input_file_list",
        type=Path,
        help="File containing list of LLVM IR or MIR files to process",
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for generated files"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["llvm", "mir"],
        default="llvm",
        help="Operation mode: 'llvm' for LLVM IR (default) or 'mir' for Machine IR",
    )
    parser.add_argument(
        "-j",
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum number of parallel workers (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress all output except errors"
    )

    args = parser.parse_args()

    # Configure logging
    level = (
        logging.ERROR
        if args.quiet
        else (logging.DEBUG if args.verbose else logging.INFO)
    )
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    generator = IR2VecTripletGenerator(
        args.llvm_build_dir,
        args.num_optimizations,
        args.output_dir,
        args.max_workers,
        args.mode,
    )
    generator.generate_triplets(args.input_file_list)


if __name__ == "__main__":
    main()
