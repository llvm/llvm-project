; RUN: rm -rf %t && mkdir -p %t
; RUN: llvm-as -o %t/1.bc %s
; RUN: llvm-lto -print-macho-cpu-only %t/1.bc | FileCheck %s

target triple = "x86_64-apple-darwin"
; CHECK: 1.bc:
; CHECK-NEXT: cputype: 16777223
; CHECK-NEXT: cpusubtype: 3
