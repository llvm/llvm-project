; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

define void @foo() nounwind {
; CHECK-LABEL: foo
; CHECK: hvc #0x2
  tail call void @llvm.aarch64.hvc(i32 2)
  ret void
}

declare void @llvm.aarch64.hvc(i32 immarg) nounwind
