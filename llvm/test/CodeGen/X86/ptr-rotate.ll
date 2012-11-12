; RUN: llc -mtriple=i386-apple-darwin -mcpu=corei7 -o - < %s | FileCheck %s

define i32 @func(i8* %A) nounwind readnone {
entry:
  %tmp = ptrtoint i8* %A to i32
  %shr = lshr i32 %tmp, 5
  %shl = shl i32 %tmp, 27
  %or = or i32 %shr, %shl
; CHECK: roll  $27
  ret i32 %or
}
