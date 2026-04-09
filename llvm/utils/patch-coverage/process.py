import subprocess
import sys
import os

from utils import classify_tests
from utils import log
from utils import target_name

# TODO: Can we process parallelly


def process_coverage_data(
    source_files, inst_build_dir, lit_binary, unit_binary, patch_path
):
    coverage_files = {}

    try:
        os.chdir(inst_build_dir)

        unit_tests, lit_tests = classify_tests(patch_path)
        if not (unit_tests or lit_tests):
            return coverage_map

        binary = unit_binary if unit_tests else lit_binary

        for root, dirs, files in os.walk("."):
            for file in files:
                if os.path.basename(file) == "default.profraw":
                    continue

                if file.endswith(".profraw"):
                    profraw_file = os.path.abspath(os.path.join(root, file))
                    profdata_output = os.path.abspath(
                        os.path.splitext(profraw_file)[0] + ".profdata"
                    )

                    log("\nProfraw File:", profraw_file)
                    log("Profdata File:", profdata_output)

                    llvm_profdata_cmd = [
                        "./bin/llvm-profdata",
                        "merge",
                        "-o",
                        profdata_output,
                        profraw_file,
                    ]

                    subprocess.check_call(llvm_profdata_cmd)

                    log(f"Converted {profraw_file} to {profdata_output}")

                    for cpp_file in source_files:
                        output_file = (
                            os.path.splitext(profdata_output)[0]
                            + f"_{cpp_file.replace('/', '_')}.txt"
                        )

                        parent_directory = os.path.dirname(os.getcwd())
                        abs_cpp_file = os.path.abspath(
                            os.path.join(parent_directory, cpp_file)
                        )

                        llvm_cov_cmd = [
                            "./bin/llvm-cov",
                            "show",
                            "-instr-profile",
                            profdata_output,
                            binary,
                            "--format=text",
                            abs_cpp_file,
                        ]

                        with open(output_file, "w") as output:
                            subprocess.check_call(llvm_cov_cmd, stdout=output)

                        log(f"Processed file saved as: {output_file}")

                        coverage_files.setdefault(abs_cpp_file, []).append(output_file)

        log("\nConversion of profraw files to human-readable form is completed.\n")

        return coverage_files

    except subprocess.CalledProcessError as e:
        log("Error during profraw to profdata conversion:", e)
        sys.exit(1)
