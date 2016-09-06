; RUN: llc -mtriple=thumb-eabi -mcpu=cortex-m3 %s -o - | FileCheck %s
; RUN: llc -mtriple=thumb-eabi -mcpu=cortex-m4 %s -o - | FileCheck %s --check-prefix=CHECK-M4

define i32 @test1(i16 zeroext %z) nounwind {
; CHECK-LABEL: test1:
; CHECK: sxth
  %r = sext i16 %z to i32
  ret i32 %r
}

define i32 @test2(i8 zeroext %z) nounwind {
; CHECK-LABEL: test2:
; CHECK: sxtb
  %r = sext i8 %z to i32
  ret i32 %r
}

define i32 @test3(i16 signext %z) nounwind {
; CHECK-LABEL: test3:
; CHECK: uxth
  %r = zext i16 %z to i32
  ret i32 %r
}

define i32 @test4(i8 signext %z) nounwind {
; CHECK-LABEL: test4:
; CHECK: uxtb
  %r = zext i8 %z to i32
  ret i32 %r
}

define i32 @test5(i32 %a, i8 %b) {
; CHECK-LABEL: test5:
; CHECK-NOT: sxtab
; CHECK-M4: sxtab r0, r0, r1
  %sext = sext i8 %b to i32
  %add = add i32 %a, %sext
  ret i32 %add
}

define i32 @test6(i32 %a, i32 %b) {
; CHECK-LABEL: test6:
; CHECK-NOT: sxtab
; CHECK-M4: sxtab r0, r0, r1
  %shl = shl i32 %b, 24
  %ashr = ashr i32 %shl, 24
  %add = add i32 %a, %ashr
  ret i32 %add
}

define i32 @test7(i32 %a, i16 %b) {
; CHECK-LABEL: test7:
; CHECK-NOT: sxtah
; CHECK-M4: sxtah r0, r0, r1
  %sext = sext i16 %b to i32
  %add = add i32 %a, %sext
  ret i32 %add
}

define i32 @test8(i32 %a, i32 %b) {
; CHECK-LABEL: test8:
; CHECK-NOT: sxtah
; CHECK-M4: sxtah r0, r0, r1
  %shl = shl i32 %b, 16
  %ashr = ashr i32 %shl, 16
  %add = add i32 %a, %ashr
  ret i32 %add
}

define i32 @test9(i32 %a, i8 %b) {
; CHECK-LABEL: test9:
; CHECK-NOT: uxtab
; CHECK-M4: uxtab r0, r0, r1
  %zext = zext i8 %b to i32
  %add = add i32 %a, %zext
  ret i32 %add
}

define i32 @test10(i32 %a, i32 %b) {
;CHECK-LABEL: test10:
;CHECK-NOT: uxtab
;CHECK-M4: uxtab r0, r0, r1
  %and = and i32 %b, 255
  %add = add i32 %a, %and
  ret i32 %add
}

define i32 @test11(i32 %a, i16 %b) {
; CHECK-LABEL: test11:
; CHECK-NOT: uxtah
; CHECK-M4: uxtah r0, r0, r1
  %zext = zext i16 %b to i32
  %add = add i32 %a, %zext
  ret i32 %add
}

define i32 @test12(i32 %a, i32 %b) {
;CHECK-LABEL: test12:
;CHECK-NOT: uxtah
;CHECK-M4: uxtah r0, r0, r1
  %and = and i32 %b, 65535
  %add = add i32 %a, %and
  ret i32 %add
}

