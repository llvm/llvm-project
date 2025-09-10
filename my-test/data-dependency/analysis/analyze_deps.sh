#!/bin/bash

echo "Data Dependency Analysis Report"
echo "==============================="

for file in *.c; do
    echo "Analyzing $file..."
    base=$(basename $file .c)
    
    # 生成 LLVM IR
    clang -O0 -emit-llvm -S $file -o ${base}.ll
    
    # 分析記憶體相依
    echo "Memory Dependencies:"
    opt -passes=print-memoryssa ${base}.ll -disable-output 2>&1 | head -10
    
    # 分析循環
    echo "Loop Analysis:"
    opt -passes=print-loops ${base}.ll -disable-output 2>&1 | head -5
    
    echo "---"
done
