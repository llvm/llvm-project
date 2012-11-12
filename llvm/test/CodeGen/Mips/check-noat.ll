; RUN: llc -march=mipsel < %s | FileCheck %s 

define void @f() nounwind readnone {
entry:
; CHECK: f:
; CHECK: .set  noat
; CHECK: .set  at

  ret void
}

