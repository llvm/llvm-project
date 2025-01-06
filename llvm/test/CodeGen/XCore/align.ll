; RUN: llc < %s -mtriple=xcore | FileCheck %s

; CHECK: .p2align 2
; CHECK-LABEL: f:
define void @f() nounwind {
entry:
  ret void
}

; CHECK: .p2align 1
; CHECK-LABEL: g:
define void @g() nounwind optsize {
entry:
  ret void
}
