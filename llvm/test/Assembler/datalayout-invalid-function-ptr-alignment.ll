; RUN: not llvm-as %s 2>&1 | FileCheck %s

; CHECK: error: Alignment is neither 0 nor a power of 2

target datalayout = "Fi24"
