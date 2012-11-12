; RUN: llc -march=mips64el -filetype=obj -mcpu=mips64r2 -mattr=n64 %s -o - \
; RUN: | llvm-objdump -disassemble -triple mips64el -mattr +mips64r2 - \
; RUN: | FileCheck %s

define i64 @dext(i64 %i) nounwind readnone {
entry:
; CHECK: dext ${{[0-9]+}}, ${{[0-9]+}}, 5, 10
  %shr = lshr i64 %i, 5
  %and = and i64 %shr, 1023
  ret i64 %and
}

define i64 @dextu(i64 %i) nounwind readnone {
entry:
; CHECK: dextu ${{[0-9]+}}, ${{[0-9]+}}, 2, 6
  %shr = lshr i64 %i, 34
  %and = and i64 %shr, 63
  ret i64 %and
}

define i64 @dextm(i64 %i) nounwind readnone {
entry:
; CHECK: dextm ${{[0-9]+}}, ${{[0-9]+}}, 5, 2
  %shr = lshr i64 %i, 5
  %and = and i64 %shr, 17179869183
  ret i64 %and
}

define i64 @dins(i64 %i, i64 %j) nounwind readnone {
entry:
; CHECK: dins ${{[0-9]+}}, ${{[0-9]+}}, 8, 10
  %shl2 = shl i64 %j, 8
  %and = and i64 %shl2, 261888
  %and3 = and i64 %i, -261889
  %or = or i64 %and3, %and
  ret i64 %or
}

define i64 @dinsm(i64 %i, i64 %j) nounwind readnone {
entry:
; CHECK: dinsm ${{[0-9]+}}, ${{[0-9]+}}, 10, 1
  %shl4 = shl i64 %j, 10
  %and = and i64 %shl4, 8796093021184
  %and5 = and i64 %i, -8796093021185
  %or = or i64 %and5, %and
  ret i64 %or
}

define i64 @dinsu(i64 %i, i64 %j) nounwind readnone {
entry:
; CHECK: dinsu ${{[0-9]+}}, ${{[0-9]+}}, 8, 13
  %shl4 = shl i64 %j, 40
  %and = and i64 %shl4, 9006099743113216
  %and5 = and i64 %i, -9006099743113217
  %or = or i64 %and5, %and
  ret i64 %or
}
