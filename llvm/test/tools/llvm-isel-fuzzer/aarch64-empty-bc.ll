; REQUIRES: aarch64-registered-target

; RUN: opt -o %t %s
; RUN: llvm-isel-fuzzer %t -ignore_remaining_args=1 -mtriple aarch64 2>&1 | FileCheck %s

; CHECK: Running
