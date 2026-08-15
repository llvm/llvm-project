import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import classify_tests
from utils import log


def process_single_profraw(profraw_file, source_files, binary, llvm_profdata, llvm_cov):
    results = {}

    profdata_output = os.path.splitext(profraw_file)[0] + ".profdata"

    log("\nProfraw File:", profraw_file)
    log("Profdata File:", profdata_output)

    subprocess.check_call([llvm_profdata, "merge", "-o", profdata_output, profraw_file])
    log(f"Converted {profraw_file} to {profdata_output}")

    for cpp_file in source_files:
        output_file = (
            os.path.splitext(profdata_output)[0] + f"_{os.path.basename(cpp_file)}.txt"
        )

        llvm_cov_cmd = [
            llvm_cov,
            "show",
            "-instr-profile",
            profdata_output,
            binary,
            "--format=text",
            cpp_file,
        ]

        with open(output_file, "w") as output:
            subprocess.check_call(llvm_cov_cmd, stdout=output)

        log(f"Processed file saved as: {output_file}")
        results.setdefault(cpp_file, []).append(output_file)

    return results


def process_coverage_data(
    source_files, inst_build_dir, lit_binary, unit_binary, patch_path
):
    coverage_files = {}

    unit_tests, lit_tests = classify_tests(patch_path)
    if not (unit_tests or lit_tests):
        return coverage_files

    binary = unit_binary if unit_tests else lit_binary
    llvm_profdata = os.path.join(inst_build_dir, "bin", "llvm-profdata")
    llvm_cov = os.path.join(inst_build_dir, "bin", "llvm-cov")
    profiles_dir = os.path.join(inst_build_dir, "profiles")

    profraw_files = [
        os.path.abspath(os.path.join(profiles_dir, f))
        for f in os.listdir(profiles_dir)
        if f.endswith(".profraw")
    ]

    try:
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    process_single_profraw,
                    profraw_file,
                    source_files,
                    binary,
                    llvm_profdata,
                    llvm_cov,
                ): profraw_file
                for profraw_file in profraw_files
            }

            for future in as_completed(futures):
                profraw_file = futures[future]
                try:
                    result = future.result()
                    for cpp_file, files in result.items():
                        coverage_files.setdefault(cpp_file, []).extend(files)
                except subprocess.CalledProcessError as e:
                    log(f"Error processing {profraw_file}: {e}")
                    sys.exit(1)

        log("\nConversion of profraw files to human-readable form is completed.\n")
        return coverage_files

    except Exception as e:
        log("Error during coverage processing:", e)
        sys.exit(1)
