; Check that fuzzer will fail on invalid input
; REQUIRES: x86-registered-target

; RUN: llvm-opt-fuzzer %s -ignore_remaining_args=1 -mtriple x86_64 -passes instcombine 2>&1 | FileCheck %s
; CHECK: input module is broken

invalid input
