# Patch based Code Coverage git integration

git-check-coverage is a Python script designed to facilitate code coverage integration with Git. This tool allows developers to track code coverage for their commits, providing insights into which parts of the codebase are tested and which are not. One of the Future plan is to also integrate into github ci and provide code coverage data for each pull request to help reviewer in reviewing the changes.

### Requirements

    Python 3.x
    Git
    LLVM build tools (CMake, Ninja)
    unidiff library (install via pip)
    A recent version of clang/clang++ to build intrumented build directory
    LLVM tools (llvm-cov, llvm-profdata, llvm-lit)

### Setup & Usage

```shell
$ cd llvm-project
$ export PATH=/home/user/local-llvm/bin:$PATH
$ mkdir build
$ cmake -G Ninja -Bbuild -Sllvm -DCMAKE_BUILD_TYPE=Release -DLLVM_USE_LINKER=lld -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_ASSERTIONS=ON
$ ninja -C build
$ cp llvm/utils/git-check-coverage .
$ chmod +x git-check-coverage
$ git check-coverage -b build bin/opt llvm/test
```

Parameters

    -b or --build-dir: Specify the build directory (default is build).
    <binary_name>: Provide the name of the binary to analyze.
    <test_suite_path>: Path to the test suite that will be executed.

Optional Arguments

    -n or --num-commits: Specify the number of recent commits to analyze.
    -i or --instrumented-build-dir: Provide a separate directory for instrumented builds.

> Note : clang and clang++ we are using is of recent version and install in path /home/user/local-llvm/bin.

### Functions Overview

The script contains several key functions:

```
extract_source_files_from_patch(patch_path): Extracts modified and test files from a patch.
write_source_file_allowlist(source_files, allowlist_path): Writes an allowlist for source files.
build_llvm(build_dir): Configures and builds LLVM in the specified build directory.
run_modified_lit_tests(llvm_lit_path, patch_path, tests): Runs modified lit tests identified from the patch file.
process_coverage_data(cpp_files, build_dir, binary): Converts profraw coverage data to profdata format and generates human-readable reports.
read_coverage_file(coverage_file): Reads a coverage file to return covered and uncovered line numbers.
print_common_uncovered_lines(cpp_file, uncovered_lines, modified_lines): Displays common uncovered lines for each modified source file.
find_lit_tests(lit_path, test_paths): Identifies lit tests based on the provided paths.
```

### Known Issues

1. Unit test
2. LLDB tests(.py) coverage

