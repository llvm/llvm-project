; If the binary looks up libraries using an rpath, we can't test this
; without copying the whole lib dir or polluting the build dir.
; REQUIRES: static-libs
; REQUIRES: x86-registered-target

; The above also applies if the binary is built with libc++.
; UNSUPPORTED: libcxx-used

; This test is really flaky on Windows. On Windows, executables and DLLs cannot
; be deleted or written while they are loaded. The OS unlocks the file some
; time after the process terminates, so if 'rm' runs too quickly, it will fail
; with "access denied".
; UNSUPPORTED: system-windows

; Temporary bitcode file
; RUN: opt -o %t.input %s

; workaround for https://openradar.appspot.com/FB8914243
; RUN: rm -f %t.bin--
; RUN: rm -f %t.bin--x86_64
; RUN: rm -f %t.bin--x86_64-unknown
; RUN: rm -f %t.bin--x86_64-instcombine

; RUN: cp llvm-opt-fuzzer %t.bin--
; RUN: not %t.bin-- %t.input 2>&1 | FileCheck -check-prefix=EMPTY %s
; EMPTY: -mtriple must be specified

; RUN: cp llvm-opt-fuzzer %t.bin--x86_64
; RUN: not %t.bin--x86_64 %t.input 2>&1 | FileCheck -check-prefix=PASSES %s
; PASSES: at least one pass should be specified

; RUN: cp llvm-opt-fuzzer %t.bin--x86_64-unknown
; RUN: not %t.bin--x86_64-unknown %t.input 2>&1 | FileCheck -check-prefix=UNKNOWN %s
; UNKNOWN: Unknown option: unknown

; RUN: cp llvm-opt-fuzzer %t.bin--x86_64-instcombine
; RUN: %t.bin--x86_64-instcombine %t.input 2>&1 | FileCheck -check-prefix=CORRECT %s
; CORRECT: Injected args: -mtriple=x86_64 -passes=instcombine
; CORRECT: Running
