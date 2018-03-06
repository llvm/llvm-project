; If the binary looks up libraries using an rpath, we can't test this
; without copying the whole lib dir or polluting the build dir.
; REQUIRES: static-libs
; REQUIRES: x86-registered-target

; This test is really flaky on Windows. On Windows, executables and DLLs cannot
; be deleted or written while they are loaded. The OS unlocks the file some
; time after the process terminates, so if 'rm' runs too quickly, it will fail
; with "access denied".
; UNSUPPORTED: system-windows

; Temporary bitcode file
; RUN: opt -o %t.input %s

; RUN: cp llvm-opt-fuzzer %t.bin--
; RUN: not %t.bin-- %t.input 2>&1 | FileCheck -check-prefix=EMPTY %s
; RUN: rm %t.bin--
; EMPTY: -mtriple must be specified

; RUN: cp llvm-opt-fuzzer %t.bin--x86_64
; RUN: not %t.bin--x86_64 %t.input 2>&1 | FileCheck -check-prefix=PASSES %s
; RUN: rm %t.bin--x86_64
; PASSES: at least one pass should be specified

; RUN: cp llvm-opt-fuzzer %t.bin--x86_64-unknown
; RUN: not %t.bin--x86_64-unknown %t.input 2>&1 | FileCheck -check-prefix=UNKNOWN %s
; RUN: rm %t.bin--x86_64-unknown
; UNKNOWN: Unknown option: unknown

; RUN: cp llvm-opt-fuzzer %t.bin--x86_64-instcombine
; RUN: %t.bin--x86_64-instcombine %t.input 2>&1 | FileCheck -check-prefix=CORRECT %s
; RUN: rm %t.bin--x86_64-instcombine
; CORRECT: Injected args: -mtriple=x86_64 -passes=instcombine
; CORRECT: Running
