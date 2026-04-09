import sys
import os

from utils import log


def read_coverage_file(coverage_file):
    uncovered_line_numbers = set()
    covered_line_numbers = set()

    try:
        with open(coverage_file, "r") as cov_file:
            for line in cov_file:
                parts = line.strip().split("|")
                if len(parts) < 3:
                    continue

                try:
                    line_number = int(parts[0].strip())
                    execution_count = int(parts[1].strip())
                except ValueError:
                    continue

                if execution_count == 0:
                    uncovered_line_numbers.add(line_number)
                else:
                    covered_line_numbers.add(line_number)

    except OSError as e:
        raise RuntimeError(f"Failed to read {coverage_file}: {e}")

    return covered_line_numbers, uncovered_line_numbers


def log_coverage_details(file, lines, uncovered_line_numbers, covered_line_numbers):
    log(f"Modified File: {os.path.basename(file)}")

    for line_number_source, line_content in lines:
        if line_number_source in uncovered_line_numbers:
            log(f"  Uncovered Line: {line_number_source} : {line_content.strip()}")
        elif line_number_source in covered_line_numbers:
            log(f"  Covered Line: {line_number_source} : {line_content.strip()}")


def print_common_uncovered_lines(cpp_file, uncovered_lines, modified_lines):
    if cpp_file not in modified_lines:
        return

    print(f"\n[code-coverage] Common uncovered lines for {cpp_file} after all changes:")
    for line_number_source, line_content in modified_lines[cpp_file]:
        if line_number_source in uncovered_lines:
            print(f"  Line {line_number_source}: {line_content.strip()}")


def report_covered_and_uncovered_lines(coverage_files, modified_lines):
    try:
        # Normalize the paths.
        norm_modified = {os.path.normpath(k): v for k, v in modified_lines.items()}

        common_uncovered_results = {}

        for cpp_file, coverage_files_list in coverage_files.items():
            norm_cpp_path = os.path.normpath(cpp_file)
            all_uncovered = set()
            all_covered = set()

            # Filter covered and uncovered lines from each coverage source file.
            for coverage_file in coverage_files_list:
                log(f"Coverage File: {coverage_file}")
                covered, uncovered = read_coverage_file(coverage_file)
                all_uncovered |= uncovered
                all_covered |= covered

            effective_uncovered = all_uncovered - all_covered
            common_uncovered_results[norm_cpp_path] = effective_uncovered

            # Log the covered and uncovered lines to <inst_build>/patch_coverage.log
            if norm_cpp_path in norm_modified:
                log_coverage_details(
                    cpp_file,
                    norm_modified[norm_cpp_path],
                    effective_uncovered,
                    all_covered,
                )

        # Print the common uncovered lines summary on stanadard output.
        for cpp_path, uncovered_set in common_uncovered_results.items():
            if cpp_path in norm_modified:
                print_common_uncovered_lines(cpp_path, uncovered_set, norm_modified)

    except Exception as ex:
        log("Error while reporting covered and uncovered lines:", ex)
        sys.exit(1)
