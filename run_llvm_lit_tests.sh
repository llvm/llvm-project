#!/bin/bash -eu

# Run llvm lit tests for Next32
# Arguments: $1 = toolchain build directory (or next32 home)
TOOLCHAIN_DIR=$1

# Add Next32 test directories.
NEXT32_LLVM_LIT_TEST_DIRS=$(find llvm/test -type d -name Next32)
NEXT32_CLANG_LIT_TEST_DIRS=$(find clang/test -type d -name Next32)

# Add tests that are not under a Next32 directory.
# TODO: Enable clang/test/Parser/pragma-ns-mark.cpp once it is fixed.
NEXT32_LLVM_LIT_TESTS="${NEXT32_LLVM_LIT_TEST_DIRS}"
NEXT32_CLANG_LIT_TESTS="${NEXT32_CLANG_LIT_TEST_DIRS}"

${TOOLCHAIN_DIR}/bin/llvm-lit $NEXT32_LLVM_LIT_TESTS -v
${TOOLCHAIN_DIR}/bin/llvm-lit $NEXT32_CLANG_LIT_TESTS -v
