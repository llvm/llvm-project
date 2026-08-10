; RUN: llc -mtriple=thumb-eabi %s -o - | FileCheck %s
; rdar://7961298

define void @t() nounwind {
entry:
; CHECK-LABEL: t:
; CHECK: trap
  call void @llvm.trap()
  unreachable
}

declare void @llvm.trap() nounwind
