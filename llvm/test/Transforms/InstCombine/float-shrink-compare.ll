; RUN: opt -S -instcombine < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define i32 @test1(float %x, float %y) nounwind uwtable {
  %1 = fpext float %x to double
  %2 = call double @ceil(double %1) nounwind readnone
  %3 = fpext float %y to double
  %4 = fcmp oeq double %2, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @test1(
; CHECK-NEXT: %ceilf = call float @ceilf(float %x)
; CHECK-NEXT: fcmp oeq float %ceilf, %y
}

define i32 @test2(float %x, float %y) nounwind uwtable {
  %1 = fpext float %x to double
  %2 = call double @fabs(double %1) nounwind readnone
  %3 = fpext float %y to double
  %4 = fcmp oeq double %2, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @test2(
; CHECK-NEXT: [[FABS:%[0-9]+]] = call float @llvm.fabs.f32(float %x)
; CHECK-NEXT: fcmp oeq float [[FABS]], %y
}

define i32 @fmf_test2(float %x, float %y) nounwind uwtable {
  %1 = fpext float %x to double
  %2 = call nnan double @fabs(double %1) nounwind readnone
  %3 = fpext float %y to double
  %4 = fcmp oeq double %2, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @fmf_test2(
; CHECK-NEXT: [[FABS:%[0-9]+]] = call nnan float @llvm.fabs.f32(float %x)
; CHECK-NEXT: fcmp oeq float [[FABS]], %y
}

define i32 @test3(float %x, float %y) nounwind uwtable {
  %1 = fpext float %x to double
  %2 = call double @floor(double %1) nounwind readnone
  %3 = fpext float %y to double
  %4 = fcmp oeq double %2, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @test3(
; CHECK-NEXT: %floorf = call float @floorf(float %x)
; CHECK-NEXT: fcmp oeq float %floorf, %y
}

define i32 @test4(float %x, float %y) nounwind uwtable {
  %1 = fpext float %x to double
  %2 = call double @nearbyint(double %1) nounwind
  %3 = fpext float %y to double
  %4 = fcmp oeq double %2, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @test4(
; CHECK-NEXT: %nearbyintf = call float @nearbyintf(float %x)
; CHECK-NEXT: fcmp oeq float %nearbyintf, %y
}

define i32 @test5(float %x, float %y) nounwind uwtable {
  %1 = fpext float %x to double
  %2 = call double @rint(double %1) nounwind
  %3 = fpext float %y to double
  %4 = fcmp oeq double %2, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @test5(
; CHECK-NEXT: %rintf = call float @rintf(float %x)
; CHECK-NEXT: fcmp oeq float %rintf, %y
}

define i32 @test6(float %x, float %y) nounwind uwtable {
  %1 = fpext float %x to double
  %2 = call double @round(double %1) nounwind readnone
  %3 = fpext float %y to double
  %4 = fcmp oeq double %2, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @test6(
; CHECK-NEXT: %roundf = call float @roundf(float %x)
; CHECK-NEXT: fcmp oeq float %roundf, %y
}

define i32 @test7(float %x, float %y) nounwind uwtable {
  %1 = fpext float %x to double
  %2 = call double @trunc(double %1) nounwind
  %3 = fpext float %y to double
  %4 = fcmp oeq double %2, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @test7(
; CHECK-NEXT: %truncf = call float @truncf(float %x)
; CHECK-NEXT: fcmp oeq float %truncf, %y
}

define i32 @test8(float %x, float %y) nounwind uwtable {
  %1 = fpext float %y to double
  %2 = fpext float %x to double
  %3 = call double @ceil(double %2) nounwind readnone
  %4 = fcmp oeq double %1, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @test8(
; CHECK-NEXT: %ceilf = call float @ceilf(float %x)
; CHECK-NEXT: fcmp oeq float %ceilf, %y
}

define i32 @test9(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %y.ext = fpext float %y to double
  %fabs = call double @fabs(double %x.ext) nounwind readnone
  %cmp = fcmp oeq double %y.ext, %fabs
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test9(
; CHECK-NEXT: %fabs = call float @llvm.fabs.f32(float %x)
; CHECK-NEXT: fcmp oeq float %fabs, %y
}

define i32 @test10(float %x, float %y) nounwind uwtable {
  %1 = fpext float %y to double
  %2 = fpext float %x to double
  %3 = call double @floor(double %2) nounwind readnone
  %4 = fcmp oeq double %1, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @test10(
; CHECK-NEXT: %floorf = call float @floorf(float %x)
; CHECK-NEXT: fcmp oeq float %floorf, %y
}

define i32 @test11(float %x, float %y) nounwind uwtable {
  %1 = fpext float %y to double
  %2 = fpext float %x to double
  %3 = call double @nearbyint(double %2) nounwind
  %4 = fcmp oeq double %1, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @test11(
; CHECK-NEXT: %nearbyintf = call float @nearbyintf(float %x)
; CHECK-NEXT: fcmp oeq float %nearbyintf, %y
}

define i32 @test12(float %x, float %y) nounwind uwtable {
  %1 = fpext float %y to double
  %2 = fpext float %x to double
  %3 = call double @rint(double %2) nounwind
  %4 = fcmp oeq double %1, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @test12(
; CHECK-NEXT: %rintf = call float @rintf(float %x)
; CHECK-NEXT: fcmp oeq float %rintf, %y
}

define i32 @test13(float %x, float %y) nounwind uwtable {
  %1 = fpext float %y to double
  %2 = fpext float %x to double
  %3 = call double @round(double %2) nounwind readnone
  %4 = fcmp oeq double %1, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @test13(
; CHECK-NEXT: %roundf = call float @roundf(float %x)
; CHECK-NEXT: fcmp oeq float %roundf, %y
}

define i32 @test14(float %x, float %y) nounwind uwtable {
  %1 = fpext float %y to double
  %2 = fpext float %x to double
  %3 = call double @trunc(double %2) nounwind
  %4 = fcmp oeq double %1, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @test14(
; CHECK-NEXT: %truncf = call float @truncf(float %x)
; CHECK-NEXT: fcmp oeq float %truncf, %y
}

define i32 @test15(float %x, float %y, float %z) nounwind uwtable {
  %1 = fpext float %x to double
  %2 = fpext float %y to double
  %3 = call double @fmin(double %1, double %2) nounwind
  %4 = fpext float %z to double
  %5 = fcmp oeq double %3, %4
  %6 = zext i1 %5 to i32
  ret i32 %6
; CHECK-LABEL: @test15(
; CHECK-NEXT: %fminf = call float @fminf(float %x, float %y)
; CHECK-NEXT: fcmp oeq float %fminf, %z
}

define i32 @test16(float %x, float %y, float %z) nounwind uwtable {
  %1 = fpext float %z to double
  %2 = fpext float %x to double
  %3 = fpext float %y to double
  %4 = call double @fmin(double %2, double %3) nounwind
  %5 = fcmp oeq double %1, %4
  %6 = zext i1 %5 to i32
  ret i32 %6
; CHECK-LABEL: @test16(
; CHECK-NEXT: %fminf = call float @fminf(float %x, float %y)
; CHECK-NEXT: fcmp oeq float %fminf, %z
}

define i32 @test17(float %x, float %y, float %z) nounwind uwtable {
  %1 = fpext float %x to double
  %2 = fpext float %y to double
  %3 = call double @fmax(double %1, double %2) nounwind
  %4 = fpext float %z to double
  %5 = fcmp oeq double %3, %4
  %6 = zext i1 %5 to i32
  ret i32 %6
; CHECK-LABEL: @test17(
; CHECK-NEXT: %fmaxf = call float @fmaxf(float %x, float %y)
; CHECK-NEXT: fcmp oeq float %fmaxf, %z
}

define i32 @test18(float %x, float %y, float %z) nounwind uwtable {
  %1 = fpext float %z to double
  %2 = fpext float %x to double
  %3 = fpext float %y to double
  %4 = call double @fmax(double %2, double %3) nounwind
  %5 = fcmp oeq double %1, %4
  %6 = zext i1 %5 to i32
  ret i32 %6
; CHECK-LABEL: @test18(
; CHECK-NEXT: %fmaxf = call float @fmaxf(float %x, float %y)
; CHECK-NEXT: fcmp oeq float %fmaxf, %z
}

define i32 @test19(float %x, float %y, float %z) nounwind uwtable {
  %1 = fpext float %x to double
  %2 = fpext float %y to double
  %3 = call double @copysign(double %1, double %2) nounwind
  %4 = fpext float %z to double
  %5 = fcmp oeq double %3, %4
  %6 = zext i1 %5 to i32
  ret i32 %6
; CHECK-LABEL: @test19(
; CHECK-NEXT: %copysignf = call float @copysignf(float %x, float %y)
; CHECK-NEXT: fcmp oeq float %copysignf, %z
}

define i32 @test20(float %x, float %y) nounwind uwtable {
  %1 = fpext float %y to double
  %2 = fpext float %x to double
  %3 = call double @fmin(double 1.000000e+00, double %2) nounwind
  %4 = fcmp oeq double %1, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK-LABEL: @test20(
; CHECK-NEXT: %fminf = call float @fminf(float 1.000000e+00, float %x)
; CHECK-NEXT: fcmp oeq float %fminf, %y
}

define i32 @test21(float %x, float %y) nounwind uwtable {
  %1 = fpext float %y to double
  %2 = fpext float %x to double
  %3 = call double @fmin(double 1.300000e+00, double %2) nounwind
  %4 = fcmp oeq double %1, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; should not be changed to fminf as the constant would loose precision
; CHECK-LABEL: @test21(
; CHECK: %3 = call double @fmin(double 1.300000e+00, double %2)
}

declare double @fabs(double) nounwind readnone
declare double @ceil(double) nounwind readnone
declare double @copysign(double, double) nounwind readnone
declare double @floor(double) nounwind readnone
declare double @nearbyint(double) nounwind readnone
declare double @rint(double) nounwind readnone
declare double @round(double) nounwind readnone
declare double @trunc(double) nounwind readnone
declare double @fmin(double, double) nounwind readnone
declare double @fmax(double, double) nounwind readnone
