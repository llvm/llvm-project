; RUN: mlir-translate -import-llvm %s | FileCheck %s

; CHECK: llvm.dependent_libraries = ["foo", "bar"]
!llvm.dependent-libraries = !{!0, !1}
!0 = !{!"foo"}
!1 = !{!"bar"}
