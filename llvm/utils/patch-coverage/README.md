# Patch-Based Code Coverage (LLVM Git Integration)

This tool provides patch-level code coverage for LLVM development workflows.
It helps developers and reviewers understand which lines introduced by a patch
are covered by tests and which are not.

## Overview

Given recent commits, the tool:

1. Generates a patch (`git diff`)
2. Identifies modified source files and lines
3. Detects modified lit tests and unit tests
4. Builds LLVM with coverage instrumentation
5. Runs only modified tests
6. Collects coverage data using `llvm-profdata` and `llvm-cov`
7. Reports:

   * Covered lines introduced by the patch
   * Uncovered lines introduced by the patch

## Setup

### 1. Build LLVM (standard workflow)

If you have an existing LLVM build, point to it with `-b`. If not, skip this step —
the tool will automatically configure and build LLVM in the specified build directory
with a minimal configuration when first run.

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

cmake -G Ninja -B build -S llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INCLUDE_TESTS=ON \
  -DLLVM_BUILD_TESTS=ON

ninja -C build
```

### 2. Install Python dependencies

```bash
pip install -r llvm/utils/patch-coverage/requirements.txt
```
or
```bash
python3 -m venv .venv                                                                                                                                                                      
source .venv/bin/activate
pip install unidiff
```

### 3. Add tool to PATH

```bash
export PATH=$PATH:llvm/utils/patch-coverage/
```

This enables:

```bash
git patch-coverage ...
```

## Usage

### Basic command

```bash
git patch-coverage -b <build-dir> <binary> <test-path>
```

### Example

```bash
git patch-coverage -b build -n 3 opt llvm/test
```

## Arguments

### Required

| Argument          | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `<binary>`        | LLVM binary to analyze (e.g., `opt`)             |


### Optional

| Argument                       | Description                                                 |
| ------------------------------ | ----------------------------------------------------------- |
| `-n, --num-commits`            | Number of recent commits to analyze (default: 1)            |
| `--projects`                   | LLVM projects to enable (e.g. clang;mlir (default: None)    |
| `-i, --instrumented-build-dir` | Use pre-built instrumented directory (Dafault: `build_inst`)| 
| `-b, --build-dir`              | Path to LLVM build directory (default: `build`)             |
| `<test-path>`                  | One or more test suite paths (default: `build/test`)        |



## How It Works
### 1. Patch Extraction
The tool runs `git diff HEAD~N HEAD` to generate a patch from the last N commits.
It parses the diff to extract:
- **Source files** — modified `.cpp`, `.c`, or `.mm` files outside test directories
- **Test files** — modified lit tests and unit tests within the patch
- **Modified lines** — the exact line numbers added or changed in each source file

### 2. Allowlist Generation
To avoid instrumenting the entire codebase, the tool generates an allowlist
containing only the source files present in the patch:   
source:/path/to/modified/file.cpp=allow  
default:skip 

This is passed to the compiler via `-fprofile-list`, so profiling overhead is
scoped only to what changed.

### 3. Instrumented Build
The tool configures a second build directory (`build_inst`) with:
- `LLVM_BUILD_INSTRUMENTED_COVERAGE=ON` — enables source-based coverage
- `LLVM_INDIVIDUAL_TEST_COVERAGE=ON` — generates per-test `.profraw` files
- The allowlist flags from step 2

Rebuilds are skipped if the patch hash (SHA-256 of the diff) matches the hash
from the last successful build, stored in `build_inst/.last_patch_hash`.

### 4. Test Execution
Only tests modified in the patch are run, against the instrumented binary.
Each test writes its coverage data to a dedicated `.profraw` file under
`build_inst/profiles/`, named by a hash of the test path to avoid collisions:

- **Lit tests** — run via `llvm-lit` with `LLVM_PROFILE_FILE` set per test
- **Unit tests** — run directly as binaries with `LLVM_PROFILE_FILE` set

### 5. Coverage Processing
For each `.profraw` file collected during test execution:
1. `llvm-profdata merge` converts it to a `.profdata` file
2. `llvm-cov show` produces a line-level coverage report for each patched
   source file, using the instrumented binary as the symbol source

### 6. Reporting
The tool cross-references coverage results against the modified lines extracted
in step 1. A line is considered:
- **Covered** — if it was executed by at least one test
- **Uncovered** — if it appears in the patch but was never executed
- **Effectively covered** — if it was uncovered in some tests but covered in
  others (union of all runs wins)

Results are printed to the terminal in diff style with color coding, and the
full details are written to `build_inst/patch_coverage.log`.

## Output

### `<build_inst>/patch_coverage.log`

```
...
Modified File: llvm/lib/IR/Example.cpp
  Covered Line: 42 : foo();
  Uncovered Line: 45 : bar();
```

### Summary on Standard Output

```
[code-coverage] Coverage report for llvm/lib/IR/Example.cpp:
  Line 42  :   foo();
  Line 45  :   bar();

```

## Troubleshooting

### No coverage generated

* Ensure instrumented build succeeded and tests run.
* Check `.profraw` files in `build_inst` directory.

### Rebuild skipped unexpectedly

* Delete `binary` to force rebuild

### For any other doubt
* Check `<build_inst>/patch_coverage.log`. It has full logging of tool.
