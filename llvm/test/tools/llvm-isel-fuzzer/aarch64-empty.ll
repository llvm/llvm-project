; REQUIRES: aarch64-registered-target

; RUN: echo > %t
; RUN: llvm-isel-fuzzer %t -ignore_remaining_args=1 -mtriple aarch64 2>&1 | FileCheck %s

; CHECK: Running
