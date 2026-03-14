; RUN: llvm-link %S/Inputs/ident.a.ll %S/Inputs/ident.b.ll -S | FileCheck %s

; Verify that multiple input llvm.ident metadata are linked together, and
; duplicate "Compiler V2" metadata from the two inputs are unified.

; CHECK-DAG: !llvm.ident = !{!0, !1, !2}
; CHECK-DAG: "Compiler V1"
; CHECK-DAG: "Compiler V2"
; CHECK-DAG: "Compiler V3"

