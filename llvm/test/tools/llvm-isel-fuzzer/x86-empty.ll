; REQUIRES: x86-registered-target

; RUN: echo > %t
; RUN: llvm-isel-fuzzer %t -ignore_remaining_args=1 -mtriple x86_64 2>&1 | FileCheck %s

; CHECK: Running
