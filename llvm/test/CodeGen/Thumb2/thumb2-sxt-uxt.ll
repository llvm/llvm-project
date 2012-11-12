; RUN: llc < %s -march=thumb -mcpu=cortex-m3 | FileCheck %s

define i32 @test1(i16 zeroext %z) nounwind {
; CHECK: test1:
; CHECK: sxth
  %r = sext i16 %z to i32
  ret i32 %r
}

define i32 @test2(i8 zeroext %z) nounwind {
; CHECK: test2:
; CHECK: sxtb
  %r = sext i8 %z to i32
  ret i32 %r
}

define i32 @test3(i16 signext %z) nounwind {
; CHECK: test3:
; CHECK: uxth
  %r = zext i16 %z to i32
  ret i32 %r
}

define i32 @test4(i8 signext %z) nounwind {
; CHECK: test4:
; CHECK: uxtb
  %r = zext i8 %z to i32
  ret i32 %r
}
