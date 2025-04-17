; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK: module attributes {
; CHECK-SAME: llvm.commandline = "exec -o infile"
!llvm.commandline = !{!0}
!0 = !{!"exec -o infile"}
