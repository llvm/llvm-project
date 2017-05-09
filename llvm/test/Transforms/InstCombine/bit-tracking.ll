; RUN: opt < %s -instcombine -S | FileCheck %s

; This file contains various testcases that require tracking whether bits are
; set or cleared by various instructions.

; Reduce down to a single XOR
define i32 @test3(i32 %B) {
; CHECK-LABEL: @test3(
; CHECK-NEXT:    [[TMP_8:%.*]] = xor i32 %B, 1
; CHECK-NEXT:    ret i32 [[TMP_8]]
;
  %ELIMinc = and i32 %B, 1
  %tmp.5 = xor i32 %ELIMinc, 1
  %ELIM7 = and i32 %B, -2
  %tmp.8 = or i32 %tmp.5, %ELIM7
  ret i32 %tmp.8
}

; Finally, a bigger case where we chain things together.  This corresponds to
; incrementing a single-bit bitfield, which should become just an xor.
define i32 @test4(i32 %B) {
; CHECK-LABEL: @test4(
; CHECK-NEXT:    [[TMP_8:%.*]] = xor i32 %B, 1
; CHECK-NEXT:    ret i32 [[TMP_8]]
;
  %ELIM3 = shl i32 %B, 31
  %ELIM4 = ashr i32 %ELIM3, 31
  %inc = add i32 %ELIM4, 1
  %ELIM5 = and i32 %inc, 1
  %ELIM7 = and i32 %B, -2
  %tmp.8 = or i32 %ELIM5, %ELIM7
  ret i32 %tmp.8
}

