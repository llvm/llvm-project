; REQUIRES: x86-registered-target

; Temporary bitcode file
; RUN: opt -o %t %s

; Don't start without target triple
; RUN: not llvm-opt-fuzzer %t 2>&1 | FileCheck -check-prefix=TRIPLE %s
; TRIPLE: -mtriple must be specified

; Don't start without passes specified
; RUN: not llvm-opt-fuzzer %t -ignore_remaining_args=1 -mtriple x86_64 2>&1 | FileCheck -check-prefix=PASSES %s
; PASSES: at least one pass should be specified

; Don't start with incorrect passes specified
; RUN: not llvm-opt-fuzzer %t -ignore_remaining_args=1 -mtriple x86_64 -passes no-pass 2>&1 | FileCheck -check-prefix=PIPELINE %s
; PIPELINE: unknown pass name 'no-pass'

; Correct command line
; RUN: llvm-opt-fuzzer %t -ignore_remaining_args=1 -mtriple x86_64 -passes instcombine 2>&1 | FileCheck -check-prefix=CORRECT %s
; CORRECT: Running
