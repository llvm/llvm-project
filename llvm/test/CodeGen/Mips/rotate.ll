; RUN: llc -march=mips -mcpu=mips32r2 < %s | FileCheck %s

; CHECK:  rotrv $2, $4
define i32 @rot0(i32 %a, i32 %b) nounwind readnone {
entry:
  %shl = shl i32 %a, %b
  %sub = sub i32 32, %b
  %shr = lshr i32 %a, %sub
  %or = or i32 %shr, %shl
  ret i32 %or
}

; CHECK:  rotr  $2, $4, 22
define i32 @rot1(i32 %a) nounwind readnone {
entry:
  %shl = shl i32 %a, 10
  %shr = lshr i32 %a, 22
  %or = or i32 %shl, %shr
  ret i32 %or
}

; CHECK:  rotrv $2, $4, $5
define i32 @rot2(i32 %a, i32 %b) nounwind readnone {
entry:
  %shr = lshr i32 %a, %b
  %sub = sub i32 32, %b
  %shl = shl i32 %a, %sub
  %or = or i32 %shl, %shr
  ret i32 %or
}

; CHECK:  rotr  $2, $4, 10
define i32 @rot3(i32 %a) nounwind readnone {
entry:
  %shr = lshr i32 %a, 10
  %shl = shl i32 %a, 22
  %or = or i32 %shr, %shl
  ret i32 %or
}

