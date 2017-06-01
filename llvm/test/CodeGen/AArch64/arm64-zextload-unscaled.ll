; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

@var32 = global i32 0

define void @test_zextloadi1_unscaled(i1* %base) {
; CHECK-LABEL: test_zextloadi1_unscaled:
; CHECK: ldurb {{w[0-9]+}}, [{{x[0-9]+}}, #-7]

  %addr = getelementptr i1, i1* %base, i32 -7
  %val = load i1, i1* %addr, align 1

  %extended = zext i1 %val to i32
  store i32 %extended, i32* @var32, align 4
  ret void
}

define void @test_zextloadi8_unscaled(i8* %base) {
; CHECK-LABEL: test_zextloadi8_unscaled:
; CHECK: ldurb {{w[0-9]+}}, [{{x[0-9]+}}, #-7]

  %addr = getelementptr i8, i8* %base, i32 -7
  %val = load i8, i8* %addr, align 1

  %extended = zext i8 %val to i32
  store i32 %extended, i32* @var32, align 4
  ret void
}

define void @test_zextloadi16_unscaled(i16* %base) {
; CHECK-LABEL: test_zextloadi16_unscaled:
; CHECK: ldurh {{w[0-9]+}}, [{{x[0-9]+}}, #-14]

  %addr = getelementptr i16, i16* %base, i32 -7
  %val = load i16, i16* %addr, align 2

  %extended = zext i16 %val to i32
  store i32 %extended, i32* @var32, align 4
  ret void
}

