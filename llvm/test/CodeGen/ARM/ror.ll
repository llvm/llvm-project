; RUN: llc -mtriple=arm-eabi -mattr=+v6 %s -o - | FileCheck %s

; rotr (rotr x, 4), 6 -> rotr x, 10 -> ror r0, r0, #10
define i32 @test1(i32 %x) nounwind readnone {
; CHECK-LABEL: test1:
; CHECK: ror  r0, r0, #4
; CHECK: ror  r0, r0, #6
; CHECK: bx  lr
entry:
  %high_part.i = shl i32 %x, 28
  %low_part.i = lshr i32 %x, 4
  %result.i = or i32 %high_part.i, %low_part.i
  %high_part.i.1 = shl i32 %result.i, 26
  %low_part.i.2 = lshr i32 %result.i, 6
  %result.i.3 = or i32 %low_part.i.2, %high_part.i.1
  ret i32 %result.i.3
}

; the same vector test
define <2 x i32> @test2(<2 x i32> %x) nounwind readnone {
; CHECK-LABEL: test2:
; CHECK: ror  r0, r0, #4
; CHECK: ror  r1, r1, #4
; CHECK: ror  r0, r0, #6
; CHECK: ror  r1, r1, #6
; CHECK: bx  lr
entry:
  %high_part.i = shl <2 x i32> %x, <i32 28, i32 28>
  %low_part.i = lshr <2 x i32> %x, <i32 4, i32 4>
  %result.i = or <2 x i32> %high_part.i, %low_part.i
  %high_part.i.1 = shl <2 x i32> %result.i, <i32 26, i32 26>
  %low_part.i.2 = lshr <2 x i32> %result.i, <i32 6, i32 6>
  %result.i.3 = or <2 x i32> %low_part.i.2, %high_part.i.1
  ret <2 x i32> %result.i.3
}

