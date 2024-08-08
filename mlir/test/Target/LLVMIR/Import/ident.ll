; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK: module attributes {
; CHECK-SAME: "flang version 61.7.4"
!llvm.ident = !{!0}
!0 = !{!"flang version 61.7.4"}
