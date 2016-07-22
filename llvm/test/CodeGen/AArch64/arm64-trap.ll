; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s
define void @foo() nounwind {
; CHECK: foo
; CHECK: brk #0x1
  tail call void @llvm.trap()
  ret void
}
declare void @llvm.trap() nounwind
