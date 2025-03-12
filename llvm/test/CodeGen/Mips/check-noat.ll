; RUN: llc -mtriple=mipsel < %s | FileCheck %s 

define void @f() nounwind readnone {
entry:
; CHECK-LABEL: f:
; CHECK: .set  noat
; CHECK: .set  at

  ret void
}

