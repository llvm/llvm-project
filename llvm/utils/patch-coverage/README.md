# Patch-Based Code Coverage (LLVM Git Integration)

This tool provides patch-level code coverage for LLVM development workflows.
It helps developers and reviewers understand which lines introduced by a patch are covered by tests and which are not.

It integrates with LLVM’s infrastructure (`llvm-lit`, coverage instrumentation, and CI) and works both locally and in CI environments.


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


## Key Features

* Patch-aware coverage (only modified lines)
* Selective test execution (only affected tests)
* Supports:

  * `llvm-lit` tests
  * LLVM unit tests
* Uses allowlist-based instrumentation for faster builds
* Avoids unnecessary rebuilds via patch hashing


## Setup

### 1. Build LLVM (standard workflow)

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
git patch-coverage -b build -n 3 bin/opt llvm/test
```

## Arguments

### Required

| Argument          | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `<binary>`        | LLVM binary to analyze (e.g., `bin/opt`)             |


### Optional

| Argument                       | Description                                                 |
| ------------------------------ | ----------------------------------------------------------- |
| `-n, --num-commits`            | Number of recent commits to analyze (default: 1)            |
| `-i, --instrumented-build-dir` | Use pre-built instrumented directory (Deafult: `build_inst`)|               
| `-b, --build-dir`              | Path to LLVM build directory (default: `build`)             |
| `<test-path>`                  | One or more test suite paths (default: `build/test`)        |


## How It Works

To be written


## Output

### `<build>/patch_coverage.log`

```
Modified File: llvm/lib/IR/Example.cpp
  Covered Line: 42 : foo();
  Uncovered Line: 45 : bar();
```

### Summary on Standard Output

```
[code-coverage] Common uncovered lines for llvm/lib/IR/Example.cpp:
  Line 45: bar();
```

## Troubleshooting

### No coverage generated

* Ensure instrumented build succeeded
* Check `.profraw` files in build directory

### Rebuild skipped unexpectedly

* Delete `.last_patch_hash` to force rebuild

