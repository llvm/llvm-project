; RUN: opt -o %t %s
; RUN: not llvm-isel-fuzzer %t 2>&1 | FileCheck %s

; CHECK: -mtriple must be specified
