; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

define void @foo() nounwind {
; CHECK-LABEL: foo
; CHECK: brk #0x2
  tail call void @llvm.aarch64.break(i32 2)
  ret void
}

declare void @llvm.aarch64.break(i32 immarg) nounwind
