; RUN: rm -rf %t && mkdir -p %t
; RUN: llvm-as -o %t/1.bc %s
; RUN: llvm-lto -print-macho-cpu-only %t/1.bc | FileCheck %s

target triple = "arm64e-apple-darwin"
; CHECK: 1.bc:
; CHECK-NEXT: cputype: 16777228
; CHECK-NEXT: cpusubtype: 2
