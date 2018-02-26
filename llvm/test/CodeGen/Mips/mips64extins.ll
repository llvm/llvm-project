; RUN: llc < %s -march=mips64el -mcpu=mips64r2 -target-abi=n64 | FileCheck %s

define i64 @dext(i64 %i) nounwind readnone {
entry:
; CHECK-LABEL: dext:
; CHECK: dext ${{[0-9]+}}, ${{[0-9]+}}, 5, 10
  %shr = lshr i64 %i, 5
  %and = and i64 %shr, 1023
  ret i64 %and
}

define i64 @dextm(i64 %i) nounwind readnone {
entry:
; CHECK-LABEL: dextm:
; CHECK: dextm ${{[0-9]+}}, ${{[0-9]+}}, 5, 34
  %shr = lshr i64 %i, 5
  %and = and i64 %shr, 17179869183
  ret i64 %and
}

define i64 @dextu(i64 %i) nounwind readnone {
entry:
; CHECK-LABEL: dextu:
; CHECK: dextu ${{[0-9]+}}, ${{[0-9]+}}, 34, 6
  %shr = lshr i64 %i, 34
  %and = and i64 %shr, 63
  ret i64 %and
}

define i64 @dins(i64 %i, i64 %j) nounwind readnone {
entry:
; CHECK-LABEL: dins:
; CHECK: dins ${{[0-9]+}}, ${{[0-9]+}}, 8, 10
  %shl2 = shl i64 %j, 8
  %and = and i64 %shl2, 261888
  %and3 = and i64 %i, -261889
  %or = or i64 %and3, %and
  ret i64 %or
}

define i64 @dinsm(i64 %i, i64 %j) nounwind readnone {
entry:
; CHECK-LABEL: dinsm:
; CHECK: dinsm ${{[0-9]+}}, ${{[0-9]+}}, 10, 33
  %shl4 = shl i64 %j, 10
  %and = and i64 %shl4, 8796093021184
  %and5 = and i64 %i, -8796093021185
  %or = or i64 %and5, %and
  ret i64 %or
}

define i64 @dinsu(i64 %i, i64 %j) nounwind readnone {
entry:
; CHECK-LABEL: dinsu:
; CHECK: dinsu ${{[0-9]+}}, ${{[0-9]+}}, 40, 13
  %shl4 = shl i64 %j, 40
  %and = and i64 %shl4, 9006099743113216
  %and5 = and i64 %i, -9006099743113217
  %or = or i64 %and5, %and
  ret i64 %or
}
