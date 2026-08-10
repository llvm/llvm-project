; RUN: llc -mtriple=thumb-eabi -mcpu=cortex-m3 %s -o - | FileCheck %s --check-prefix=CHECK-NO-DSP
; RUN: llc -mtriple=thumb-eabi -mcpu=cortex-m4 %s -o - | FileCheck %s --check-prefix=CHECK-DSP
; RUN: llc -mtriple=thumbv7em-eabi %s -o - | FileCheck %s -check-prefix=CHECK-DSP
; RUN: llc -mtriple=thumbv8m.main-none-eabi %s -o - | FileCheck %s -check-prefix=CHECK-NO-DSP
; RUN: llc -mtriple=thumbv8m.main-none-eabi -mattr=+dsp %s -o - | FileCheck %s -check-prefix=CHECK-DSP

define i32 @test1(i16 zeroext %z) nounwind {
; CHECK-LABEL: test1:
; CHECK-DSP: sxth
; CHECK-NO-DSP: sxth
  %r = sext i16 %z to i32
  ret i32 %r
}

define i32 @test2(i8 zeroext %z) nounwind {
; CHECK-LABEL: test2:
; CHECK-DSP: sxtb
; CHECK-NO-DSP: sxtb
  %r = sext i8 %z to i32
  ret i32 %r
}

define i32 @test3(i16 signext %z) nounwind {
; CHECK-LABEL: test3:
; CHECK-DSP: uxth
; CHECK-NO-DSP: uxth
  %r = zext i16 %z to i32
  ret i32 %r
}

define i32 @test4(i8 signext %z) nounwind {
; CHECK-LABEL: test4:
; CHECK-DSP: uxtb
; CHECK-NO-DSP: uxtb
  %r = zext i8 %z to i32
  ret i32 %r
}

define i32 @test5(i32 %a, i8 %b) {
; CHECK-LABEL: test5:
; CHECK-DSP: sxtab r0, r0, r1
; CHECK-NO-DSP-NOT: sxtab
  %sext = sext i8 %b to i32
  %add = add i32 %a, %sext
  ret i32 %add
}

define i32 @test6(i32 %a, i32 %b) {
; CHECK-LABEL: test6:
; CHECK-DSP: sxtab r0, r0, r1
; CHECK-NO-DSP-NOT: sxtab
  %shl = shl i32 %b, 24
  %ashr = ashr i32 %shl, 24
  %add = add i32 %a, %ashr
  ret i32 %add
}

define i32 @test7(i32 %a, i16 %b) {
; CHECK-LABEL: test7:
; CHECK-DSP: sxtah r0, r0, r1
; CHECK-NO-DSPNOT: sxtah
  %sext = sext i16 %b to i32
  %add = add i32 %a, %sext
  ret i32 %add
}

define i32 @test8(i32 %a, i32 %b) {
; CHECK-LABEL: test8:
; CHECK-DSP: sxtah r0, r0, r1
; CHECK-NO-DSP-NOT: sxtah
  %shl = shl i32 %b, 16
  %ashr = ashr i32 %shl, 16
  %add = add i32 %a, %ashr
  ret i32 %add
}

define i32 @test9(i32 %a, i8 %b) {
; CHECK-LABEL: test9:
; CHECK-DSP: uxtab r0, r0, r1
; CHECK-NO-DSP-NOT: uxtab
  %zext = zext i8 %b to i32
  %add = add i32 %a, %zext
  ret i32 %add
}

define i32 @test10(i32 %a, i32 %b) {
;CHECK-LABEL: test10:
;CHECK-DSP: uxtab r0, r0, r1
;CHECK-NO-DSP-NOT: uxtab
  %and = and i32 %b, 255
  %add = add i32 %a, %and
  ret i32 %add
}

define i32 @test11(i32 %a, i16 %b) {
; CHECK-LABEL: test11:
; CHECK-DSP: uxtah r0, r0, r1
; CHECK-NO-DSP-NOT: uxtah
  %zext = zext i16 %b to i32
  %add = add i32 %a, %zext
  ret i32 %add
}

define i32 @test12(i32 %a, i32 %b) {
;CHECK-LABEL: test12:
;CHECK-DSP: uxtah r0, r0, r1
;CHECK-NO-DSP-NOT: uxtah
  %and = and i32 %b, 65535
  %add = add i32 %a, %and
  ret i32 %add
}

