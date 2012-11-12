; RUN: llc -march=mips64el -mcpu=mips64r2 < %s | FileCheck %s

define i64 @f0(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: dsllv
  %shl = shl i64 %a0, %a1
  ret i64 %shl
}

define i64 @f1(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: dsrav
  %shr = ashr i64 %a0, %a1
  ret i64 %shr
}

define i64 @f2(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: dsrlv
  %shr = lshr i64 %a0, %a1
  ret i64 %shr
}

define i64 @f3(i64 %a0) nounwind readnone {
entry:
; CHECK: dsll ${{[0-9]+}}, ${{[0-9]+}}, 10
  %shl = shl i64 %a0, 10
  ret i64 %shl
}

define i64 @f4(i64 %a0) nounwind readnone {
entry:
; CHECK: dsra ${{[0-9]+}}, ${{[0-9]+}}, 10
  %shr = ashr i64 %a0, 10
  ret i64 %shr
}

define i64 @f5(i64 %a0) nounwind readnone {
entry:
; CHECK: dsrl ${{[0-9]+}}, ${{[0-9]+}}, 10
  %shr = lshr i64 %a0, 10
  ret i64 %shr
}

define i64 @f6(i64 %a0) nounwind readnone {
entry:
; CHECK: dsll ${{[0-9]+}}, ${{[0-9]+}}, 40
  %shl = shl i64 %a0, 40
  ret i64 %shl
}

define i64 @f7(i64 %a0) nounwind readnone {
entry:
; CHECK: dsra ${{[0-9]+}}, ${{[0-9]+}}, 40
  %shr = ashr i64 %a0, 40
  ret i64 %shr
}

define i64 @f8(i64 %a0) nounwind readnone {
entry:
; CHECK: dsrl ${{[0-9]+}}, ${{[0-9]+}}, 40
  %shr = lshr i64 %a0, 40
  ret i64 %shr
}

define i64 @f9(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: drotrv
  %shr = lshr i64 %a0, %a1
  %sub = sub i64 64, %a1
  %shl = shl i64 %a0, %sub
  %or = or i64 %shl, %shr
  ret i64 %or
}

define i64 @f10(i64 %a0, i64 %a1) nounwind readnone {
entry:
; CHECK: drotrv
  %shl = shl i64 %a0, %a1
  %sub = sub i64 64, %a1
  %shr = lshr i64 %a0, %sub
  %or = or i64 %shr, %shl
  ret i64 %or
}

define i64 @f11(i64 %a0) nounwind readnone {
entry:
; CHECK: drotr ${{[0-9]+}}, ${{[0-9]+}}, 10
  %shr = lshr i64 %a0, 10
  %shl = shl i64 %a0, 54
  %or = or i64 %shr, %shl
  ret i64 %or
}

define i64 @f12(i64 %a0) nounwind readnone {
entry:
; CHECK: drotr ${{[0-9]+}}, ${{[0-9]+}}, 54
  %shl = shl i64 %a0, 10
  %shr = lshr i64 %a0, 54
  %or = or i64 %shl, %shr
  ret i64 %or
}


