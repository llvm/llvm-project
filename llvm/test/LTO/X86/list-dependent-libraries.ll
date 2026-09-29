; RUN: rm -rf %t && mkdir -p %t
; RUN: llvm-as -o %t/1.bc %s
; RUN: llvm-as -o %t/2.bc %S/Inputs/list-dependent-libraries.ll
; RUN: llvm-lto -list-dependent-libraries-only %t/1.bc %t/2.bc | FileCheck %s
; REQUIRES: default_triple

; CHECK: 1.bc:
; CHECK-NEXT: wibble
; CHECK: 2.bc:
; CHECK-NEXT: foo
; CHECK-NEXT: b a r
; CHECK-NEXT: foo

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

!llvm.dependent-libraries = !{!0}

!0 = !{!"wibble"}
