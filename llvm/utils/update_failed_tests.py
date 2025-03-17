#!/usr/bin/env python3

"""A test case update orchestrator script.

This script is a utility to automate the updating of failed LLVM test cases by
invoking other `update_*_test_checkes.py` scripts. It provides a one-click
solution to update test expectations for previously failed tests.
"""

import subprocess
import os


def get_llvm_path():
    util_path = os.path.dirname(os.path.realpath(__file__))
    llvm_path = os.path.dirname(util_path)
    print("The LLVM path is", llvm_path)
    return llvm_path


def get_build_path():
    if os.path.basename(os.getcwd()).startswith("build"):
        build_path = os.getcwd()
    else:
        dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
        build_dirs = [d for d in dirs if d.startswith('build')]

        if len(build_dirs) != 1:
            print(
                "Cannot find the build directory. Please run this script in the build directory.")
            exit(1)
        build_path = build_dirs[0]

    print("The BUILD path is", build_path)
    return build_path


def run_tool(tool_path, tool_name, tool_bin, build_path, file_path):
    print(tool_name.upper() + " updating: ", file_path)
    result = subprocess.run(
        ["python3", tool_path, "--"+tool_name+"="+build_path+"/bin/"+tool_bin, file_path])
    return result


def run(build_path, projetct_name, project_path, test_times_path):
    if not os.path.exists(test_times_path):
        print("No tests found for", projetct_name)
        return

    # read lit test records:
    with open(test_times_path, 'r') as f:
        rest_tests = []

        for line in f:
            # split line into Time and Path
            parts = line.strip().split(' ', 1)
            run_time = float(parts[0])
            file_path = project_path + "/test/" + parts[1]

            # If time is negative, then it is a failed test
            if run_time < 0:
                if not os.path.exists(file_path):
                    print("NOT FOUND:", file_path)
                    continue

                # open file, read first line
                with open(file_path, 'r') as target_file:
                    first_line = target_file.readline().strip()
                if not first_line.startswith("; NOTE: Assertions") and not first_line.startswith("# NOTE: Assertions"):
                    print("\nSKIP: ", file_path)
                    continue

                tool_name = first_line.split(" ")[7]
                tool_path = llvm_path + "/" + tool_name

                # call update tools
                if "update_llc_test_checks" in tool_name:
                    result = run_tool(tool_path, "llc", "llc",
                                      build_path, file_path)
                elif "update_cc_test_checks" in tool_name:
                    result = run_tool(tool_path, "cc", "clang",
                                      build_path, file_path)
                elif "update_test_checks" in tool_name or "update_analyze_test_checks" in tool_name:
                    result = run_tool(tool_path, "opt", "opt",
                                      build_path, file_path)
                elif "update_mir_test_checks" in tool_name:
                    result = run_tool(tool_path, "llc", "llc",
                                      build_path, file_path)
                else:
                    print("\nUNHANDLED: ", file_path)
                    continue

                if result.returncode != 0:
                    rest_tests.append(file_path)

        if len(rest_tests) != 0:
            for failed in rest_tests:
                print("FAILED: ", failed)


if __name__ == "__main__":
    llvm_path = get_llvm_path()
    build_path = get_build_path()

    llvm_test_times_path = build_path + "/test/.lit_test_times.txt"
    clang_test_times_path = build_path + "/tools/clang/test/.lit_test_times.txt"
    clang_path = llvm_path + "/clang"

    run(build_path, "llvm", llvm_path, llvm_test_times_path)
    run(build_path, "clang", clang_path, clang_test_times_path)
