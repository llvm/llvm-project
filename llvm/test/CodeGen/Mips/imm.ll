; RUN: llc -march=mipsel < %s | FileCheck %s

define i32 @foo0() nounwind readnone {
entry:
; CHECK: foo0
; CHECK: lui $[[R0:[0-9]+]], 4660
; CHECK: ori ${{[0-9]+}}, $[[R0]], 22136
  ret i32 305419896
}

define i32 @foo1() nounwind readnone {
entry:
; CHECK: foo1
; CHECK: lui ${{[0-9]+}}, 4660
; CHECK-NOT: ori
  ret i32 305397760
}

define i32 @foo2() nounwind readnone {
entry:
; CHECK: foo2
; CHECK: addiu ${{[0-9]+}}, $zero, 4660
  ret i32 4660
}

define i32 @foo17() nounwind readnone {
entry:
; CHECK: foo17
; CHECK: addiu ${{[0-9]+}}, $zero, -32204
  ret i32 -32204
}

define i32 @foo18() nounwind readnone {
entry:
; CHECK: foo18
; CHECK: ori ${{[0-9]+}}, $zero, 33332
  ret i32 33332
}
