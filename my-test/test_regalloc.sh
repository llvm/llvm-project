#!/bin/bash

# Register Allocator Comparison Test Framework
# Usage: ./test_regalloc.sh <test_file.c>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <test_file.c>"
    exit 1
fi

TEST_FILE=$1
BASE_NAME=$(basename "$TEST_FILE" .c)
CLANG_PATH="~/work/llvm-project/build/bin/clang"
LLC_PATH="~/work/llvm-project/build/bin/llc"

echo "================================="
echo "Testing Register Allocators"
echo "File: $TEST_FILE"
echo "================================="

# Step 1: Compile to LLVM IR
echo "Step 1: Compiling to LLVM IR..."
$CLANG_PATH -O0 -emit-llvm -S "$TEST_FILE" -o "${BASE_NAME}.ll"

if [ $? -ne 0 ]; then
    echo "Error: Failed to compile to LLVM IR"
    exit 1
fi

echo "Generated: ${BASE_NAME}.ll"

# Step 2: Test with Segment Tree Allocator
echo ""
echo "Step 2: Testing Segment Tree Allocator..."
echo "----------------------------------------"
$LLC_PATH -O3 -mtriple=aarch64-unknown-linux-gnu \
    -regalloc=segtre \
    -time-passes \
    -stats \
    "${BASE_NAME}.ll" \
    -o "${BASE_NAME}_segtre.s" 2> "${BASE_NAME}_segtre_stats.txt"

echo "Generated: ${BASE_NAME}_segtre.s"
echo "Stats saved to: ${BASE_NAME}_segtre_stats.txt"

# Step 3: Test with Greedy Allocator
echo ""
echo "Step 3: Testing Greedy Allocator..."
echo "-----------------------------------"
$LLC_PATH -O3 -mtriple=aarch64-unknown-linux-gnu \
    -regalloc=greedy \
    -time-passes \
    -stats \
    "${BASE_NAME}.ll" \
    -o "${BASE_NAME}_greedy.s" 2> "${BASE_NAME}_greedy_stats.txt"

echo "Generated: ${BASE_NAME}_greedy.s"
echo "Stats saved to: ${BASE_NAME}_greedy_stats.txt"

# Step 4: Analysis
echo ""
echo "Step 4: Analyzing Results..."
echo "============================"

# Count instructions
SEGTRE_INSTR=$(grep -c '^\s*[a-zA-Z]' "${BASE_NAME}_segtre.s")
GREEDY_INSTR=$(grep -c '^\s*[a-zA-Z]' "${BASE_NAME}_greedy.s")

echo "Instruction Count:"
echo "  Segment Tree: $SEGTRE_INSTR"
echo "  Greedy:       $GREEDY_INSTR"

# Count load/store instructions (potential spills)
SEGTRE_LOADSTORE=$(grep -c -E '^\s*(ldr|str|ldp|stp)' "${BASE_NAME}_segtre.s")
GREEDY_LOADSTORE=$(grep -c -E '^\s*(ldr|str|ldp|stp)' "${BASE_NAME}_greedy.s")

echo ""
echo "Load/Store Instructions (potential spills):"
echo "  Segment Tree: $SEGTRE_LOADSTORE"
echo "  Greedy:       $GREEDY_LOADSTORE"

# Extract compilation time
SEGTRE_TIME=$(grep "Total Execution Time" "${BASE_NAME}_segtre_stats.txt" | head -1 | awk '{print $4}')
GREEDY_TIME=$(grep "Total Execution Time" "${BASE_NAME}_greedy_stats.txt" | head -1 | awk '{print $4}')

echo ""
echo "Compilation Time:"
echo "  Segment Tree: $SEGTRE_TIME seconds"
echo "  Greedy:       $GREEDY_TIME seconds"

# Check for register allocator specific stats
echo ""
echo "Register Allocator Specific Stats:"
echo "==================================="

if grep -q "Segment Tree Allocator Cumulative Statistics" "${BASE_NAME}_segtre_stats.txt"; then
    echo "Segment Tree Stats:"
    grep -A 15 "Segment Tree Allocator Cumulative Statistics" "${BASE_NAME}_segtre_stats.txt"
fi

if grep -q "Register Allocation" "${BASE_NAME}_greedy_stats.txt"; then
    echo ""
    echo "Greedy Allocator Stats:"
    grep -A 10 "Register Allocation" "${BASE_NAME}_greedy_stats.txt"
fi

# Compare assembly differences
echo ""
echo "Assembly Code Differences:"
echo "========================="
diff -u "${BASE_NAME}_greedy.s" "${BASE_NAME}_segtre.s" > "${BASE_NAME}_diff.txt"

if [ -s "${BASE_NAME}_diff.txt" ]; then
    echo "Differences found! See ${BASE_NAME}_diff.txt"
    echo "First 20 lines of differences:"
    head -20 "${BASE_NAME}_diff.txt"
else
    echo "No differences in generated assembly code."
fi

echo ""
echo "Testing completed!"
echo "Files generated:"
echo "  - ${BASE_NAME}.ll (LLVM IR)"
echo "  - ${BASE_NAME}_segtre.s (Segment Tree assembly)"
echo "  - ${BASE_NAME}_greedy.s (Greedy assembly)" 
echo "  - ${BASE_NAME}_segtre_stats.txt (Segment Tree stats)"
echo "  - ${BASE_NAME}_greedy_stats.txt (Greedy stats)"
echo "  - ${BASE_NAME}_diff.txt (Assembly differences)"