;; This test ensures that the backend does not crash when CodeView is enabled
;; through module flags but llvm.dbg.cu is missing.

; RUN: llc < %s -mtriple=x86_64-unknown-uefi | FileCheck %s

define void @foo() {
entry:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"CodeView", i32 1}

; CHECK-LABEL: foo:
; CHECK: retq
