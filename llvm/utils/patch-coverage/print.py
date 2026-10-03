import os
import re
import sys

from utils import log
from utils import group_contiguous_lines


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


def coverage_percentage(cpp_file, uncovered_lines, modified_lines):
    total = 0
    covered_count = 0

    for line_number_source, line_content in modified_lines[cpp_file]:
        total += 1

        content = line_content.strip()
        if not content:
            covered_count += 1

        elif line_number_source not in uncovered_lines:
            covered_count += 1

    percent = (covered_count * 100 / total) if total else 0
    print(
        f"\n\033[33mFILE COVERAGE: {percent:.1f}% ({covered_count}/{total} lines)\033[0m"
    )
    return total, covered_count


def print_coverage_report(cpp_file, uncovered_lines, modified_lines):
    if cpp_file not in modified_lines:
        return

    rela_path = re.sub(r"^(\.\./)+", "", os.path.relpath(cpp_file))
    print(f"\n\033[1m[code-coverage] Coverage report for {rela_path}:\033[0m")

    with open(cpp_file, "r") as f:
        file_lines = f.readlines()

    patch_lines = sorted(line for line, _ in modified_lines[cpp_file])
    uncovered_lines = set(uncovered_lines)

    groups = group_contiguous_lines(patch_lines)

    # To make use we do not print overlapped context window lines
    printed = set()

    for group in groups:
        start = group[0]
        end = group[-1]

        window_start = max(1, start - 1)
        window_end = min(len(file_lines), end + 1)

        for line in range(window_start, window_end + 1):
            if line in printed:
                continue
            printed.add(line)

            # Syncronise with 0 based indexing that file_lines have.
            content = file_lines[line - 1].rstrip()

            # red 31, green 32, white 0
            if line in patch_lines:
                if line in uncovered_lines:
                    print(f"\033[36m  Line {line:<5}\033[0m: \033[31m{content}\033[0m")
                else:
                    print(f"\033[36m  Line {line:<5}\033[0m: \033[32m{content}\033[0m")
            else:
                print(f"\033[36m  Line {line:<5}\033[0m: {content}")

    return coverage_percentage(cpp_file, uncovered_lines, modified_lines)


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

        # Print the covered, uncovered and context lines on standard output in diff style.
        total = 0
        covered_count = 0
        for cpp_path, uncovered_set in common_uncovered_results.items():
            if cpp_path in norm_modified:
                temp1, temp2 = print_coverage_report(
                    cpp_path, uncovered_set, norm_modified
                )
                total += temp1
                covered_count += temp2
        percent = (covered_count * 100 / total) if total else 0
        print(
            f"\n\033[33mPATCH COVERAGE: {percent:.1f}% ({covered_count}/{total} lines)\033[0m"
        )

    except Exception as ex:
        log("Error while reporting covered and uncovered lines:", ex)
        sys.exit(1)
