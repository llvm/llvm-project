; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

@var32 = global i32 0

define void @test_zextloadi1_unscaled(ptr %base) {
; CHECK-LABEL: test_zextloadi1_unscaled:
; CHECK: ldurb {{w[0-9]+}}, [{{x[0-9]+}}, #-7]

  %addr = getelementptr i1, ptr %base, i32 -7
  %val = load i1, ptr %addr, align 1

  %extended = zext i1 %val to i32
  store i32 %extended, ptr @var32, align 4
  ret void
}

define void @test_zextloadi8_unscaled(ptr %base) {
; CHECK-LABEL: test_zextloadi8_unscaled:
; CHECK: ldurb {{w[0-9]+}}, [{{x[0-9]+}}, #-7]

  %addr = getelementptr i8, ptr %base, i32 -7
  %val = load i8, ptr %addr, align 1

  %extended = zext i8 %val to i32
  store i32 %extended, ptr @var32, align 4
  ret void
}

define void @test_zextloadi16_unscaled(ptr %base) {
; CHECK-LABEL: test_zextloadi16_unscaled:
; CHECK: ldurh {{w[0-9]+}}, [{{x[0-9]+}}, #-14]

  %addr = getelementptr i16, ptr %base, i32 -7
  %val = load i16, ptr %addr, align 2

  %extended = zext i16 %val to i32
  store i32 %extended, ptr @var32, align 4
  ret void
}

