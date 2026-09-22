; RUN: not llc -mtriple=pisa -filetype=asm %s -o /dev/null 2>&1 | FileCheck %s

target triple = "pisa"

@target = addrspace(1) global i8 0
@bad = addrspace(1) global <2 x i24> <i24 ptrtoint (ptr addrspace(1) @target to i24), i24 1>

; CHECK: LLVM ERROR: cannot lower vector global with unusual element type
