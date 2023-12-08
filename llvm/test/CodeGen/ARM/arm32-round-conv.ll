; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mattr=+fp-armv8 | FileCheck %s
; RUN: llc < %s -mtriple=armv8-linux-gnueabihf -mattr=+fp-armv8 | FileCheck %s

; CHECK-LABEL: test1
; CHECK: vcvtm.s32.f32
define i32 @test1(float %a) {
entry:
  %call = call float @floorf(float %a) nounwind readnone
  %conv = fptosi float %call to i32
  ret i32 %conv
}

; CHECK-LABEL: test2
; CHECK: vcvtm.u32.f32
define i32 @test2(float %a) {
entry:
  %call = call float @floorf(float %a) nounwind readnone
  %conv = fptoui float %call to i32
  ret i32 %conv
}

; CHECK-LABEL: test3
; CHECK: vcvtm.s32.f64
define i32 @test3(double %a) {
entry:
  %call = call double @floor(double %a) nounwind readnone
  %conv = fptosi double %call to i32
  ret i32 %conv
}

; CHECK-LABEL: test4
; CHECK: vcvtm.u32.f64
define i32 @test4(double %a) {
entry:
  %call = call double @floor(double %a) nounwind readnone
  %conv = fptoui double %call to i32
  ret i32 %conv
}

; CHECK-LABEL: test5
; CHECK: vcvtp.s32.f32
define i32 @test5(float %a) {
entry:
  %call = call float @ceilf(float %a) nounwind readnone
  %conv = fptosi float %call to i32
  ret i32 %conv
}

; CHECK-LABEL: test6
; CHECK: vcvtp.u32.f32
define i32 @test6(float %a) {
entry:
  %call = call float @ceilf(float %a) nounwind readnone
  %conv = fptoui float %call to i32
  ret i32 %conv
}

; CHECK-LABEL: test7
; CHECK: vcvtp.s32.f64
define i32 @test7(double %a) {
entry:
  %call = call double @ceil(double %a) nounwind readnone
  %conv = fptosi double %call to i32
  ret i32 %conv
}

; CHECK-LABEL: test8
; CHECK: vcvtp.u32.f64
define i32 @test8(double %a) {
entry:
  %call = call double @ceil(double %a) nounwind readnone
  %conv = fptoui double %call to i32
  ret i32 %conv
}

; CHECK-LABEL: test9
; CHECK: vcvta.s32.f32
define i32 @test9(float %a) {
entry:
  %call = call float @roundf(float %a) nounwind readnone
  %conv = fptosi float %call to i32
  ret i32 %conv
}

; CHECK-LABEL: test10
; CHECK: vcvta.u32.f32
define i32 @test10(float %a) {
entry:
  %call = call float @roundf(float %a) nounwind readnone
  %conv = fptoui float %call to i32
  ret i32 %conv
}

; CHECK-LABEL: test11
; CHECK: vcvta.s32.f64
define i32 @test11(double %a) {
entry:
  %call = call double @round(double %a) nounwind readnone
  %conv = fptosi double %call to i32
  ret i32 %conv
}

; CHECK-LABEL: test12
; CHECK: vcvta.u32.f64
define i32 @test12(double %a) {
entry:
  %call = call double @round(double %a) nounwind readnone
  %conv = fptoui double %call to i32
  ret i32 %conv
}

declare float @floorf(float) nounwind readnone
declare double @floor(double) nounwind readnone
declare float @ceilf(float) nounwind readnone
declare double @ceil(double) nounwind readnone
declare float @roundf(float) nounwind readnone
declare double @round(double) nounwind readnone
