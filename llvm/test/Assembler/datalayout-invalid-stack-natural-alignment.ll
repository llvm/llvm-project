; RUN: not --crash llvm-as %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Alignment is neither 0 nor a power of 2

target datalayout = "S24"
