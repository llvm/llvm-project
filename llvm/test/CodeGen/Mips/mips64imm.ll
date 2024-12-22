; RUN: llc -march=mips64el -mcpu=mips4 < %s | FileCheck %s
; RUN: llc -march=mips64el -mcpu=mips64 < %s | FileCheck %s

define i32 @foo1() nounwind readnone {
entry:
; CHECK: foo1
; CHECK: lui ${{[0-9]+}}, 4660
; CHECK-NOT: ori
  ret i32 305397760
}

define i64 @foo3() nounwind readnone {
entry:
; CHECK: foo3
; CHECK: lui $[[R0:[0-9]+]], 4660
; CHECK: ori ${{[0-9]+}}, $[[R0]], 22136
  ret i64 305419896
}

define i64 @foo6() nounwind readnone {
entry:
; CHECK: foo6
; CHECK: ori ${{[0-9]+}}, $zero, 33332
  ret i64 33332
}

define i64 @foo7() nounwind readnone {
entry:
; CHECK: foo7
; CHECK: daddiu ${{[0-9]+}}, $zero, -32204
  ret i64 -32204
}

define i64 @foo9() nounwind readnone {
entry:
; CHECK: foo9
; CHECK: lui $[[R0:[0-9]+]], 583
; CHECK: daddiu $[[R1:[0-9]+]], $[[R0]], -30001
; CHECK: dsll $[[R2:[0-9]+]], $[[R1]], 18
; CHECK: daddiu $[[R3:[0-9]+]], $[[R2]], 18441
; CHECK: dsll $[[R4:[0-9]+]], $[[R3]], 17
; CHECK: daddiu ${{[0-9]+}}, $[[R4]], 13398
  ret i64 1311768467284833366
}

define i64 @foo10() nounwind readnone {
entry:
; CHECK: foo10
; CHECK: lui $[[R0:[0-9]+]], 34661
; CHECK: daddiu  ${{[0-9]+}}, $[[R0]], 17185
  ret i64 -8690466096928522240
}

