; RUN: opt -S -instcombine < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define i32 @test1(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %ceil = call double @ceil(double %x.ext) nounwind readnone
  %ext.y = fpext float %y to double
  %cmp = fcmp oeq double %ceil, %ext.y
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test1(
; CHECK-NEXT: %ceil = call float @llvm.ceil.f32(float %x)
; CHECK-NEXT: fcmp oeq float %ceil, %y
}

define i32 @test1_intrin(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %ceil = call double @llvm.ceil.f64(double %x.ext) nounwind readnone
  %ext.y = fpext float %y to double
  %cmp = fcmp oeq double %ceil, %ext.y
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test1_intrin(
; CHECK-NEXT: %ceil = call float @llvm.ceil.f32(float %x)
; CHECK-NEXT: fcmp oeq float %ceil, %y
}

define i32 @test2(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %fabs = call double @fabs(double %x.ext) nounwind readnone
  %y.ext = fpext float %y to double
  %cmp = fcmp oeq double %fabs, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test2(
; CHECK-NEXT: %fabs = call float @llvm.fabs.f32(float %x)
; CHECK-NEXT: fcmp oeq float %fabs, %y
}

define i32 @test2_intrin(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %fabs = call double @llvm.fabs.f64(double %x.ext) nounwind readnone
  %y.ext = fpext float %y to double
  %cmp = fcmp oeq double %fabs, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test2_intrin(
; CHECK-NEXT: %fabs = call float @llvm.fabs.f32(float %x)
; CHECK-NEXT: fcmp oeq float %fabs, %y
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
  %x.ext = fpext float %x to double
  %floor = call double @floor(double %x.ext) nounwind readnone
  %y.ext = fpext float %y to double
  %cmp = fcmp oeq double %floor, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test3(
; CHECK-NEXT: %floor = call float @llvm.floor.f32(float %x)
; CHECK-NEXT: fcmp oeq float %floor, %y
}


define i32 @test3_intrin(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %floor = call double @llvm.floor.f64(double %x.ext) nounwind readnone
  %y.ext = fpext float %y to double
  %cmp = fcmp oeq double %floor, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test3_intrin(
; CHECK-NEXT: %floor = call float @llvm.floor.f32(float %x)
; CHECK-NEXT: fcmp oeq float %floor, %y
}

define i32 @test4(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %nearbyint = call double @nearbyint(double %x.ext) nounwind
  %y.ext = fpext float %y to double
  %cmp = fcmp oeq double %nearbyint, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test4(
; CHECK-NEXT: %nearbyint = call float @llvm.nearbyint.f32(float %x)
; CHECK-NEXT: fcmp oeq float %nearbyint, %y
}

define i32 @shrink_nearbyint_intrin(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %nearbyint = call double @llvm.nearbyint.f64(double %x.ext) nounwind
  %y.ext = fpext float %y to double
  %cmp = fcmp oeq double %nearbyint, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @shrink_nearbyint_intrin(
; CHECK-NEXT: %nearbyint = call float @llvm.nearbyint.f32(float %x)
; CHECK-NEXT: fcmp oeq float %nearbyint, %y
}

define i32 @test5(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %rint = call double @rint(double %x.ext) nounwind
  %y.ext = fpext float %y to double
  %cmp = fcmp oeq double %rint, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test5(
; CHECK-NEXT: %rint = call float @llvm.rint.f32(float %x)
; CHECK-NEXT: fcmp oeq float %rint, %y
}

define i32 @test6(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %round = call double @round(double %x.ext) nounwind readnone
  %y.ext = fpext float %y to double
  %cmp = fcmp oeq double %round, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test6(
; CHECK-NEXT: %round = call float @llvm.round.f32(float %x)
; CHECK-NEXT: fcmp oeq float %round, %y
}

define i32 @test6_intrin(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %round = call double @llvm.round.f64(double %x.ext) nounwind readnone
  %y.ext = fpext float %y to double
  %cmp = fcmp oeq double %round, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test6_intrin(
; CHECK-NEXT: %round = call float @llvm.round.f32(float %x)
; CHECK-NEXT: fcmp oeq float %round, %y
}

define i32 @test7(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %trunc = call double @trunc(double %x.ext) nounwind
  %y.ext = fpext float %y to double
  %cmp = fcmp oeq double %trunc, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test7(
; CHECK-NEXT: %trunc = call float @llvm.trunc.f32(float %x)
; CHECK-NEXT: fcmp oeq float %trunc, %y
}

define i32 @test7_intrin(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %trunc = call double @llvm.trunc.f64(double %x.ext) nounwind
  %y.ext = fpext float %y to double
  %cmp = fcmp oeq double %trunc, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test7_intrin(
; CHECK-NEXT: %trunc = call float @llvm.trunc.f32(float %x)
; CHECK-NEXT: fcmp oeq float %trunc, %y
}

define i32 @test8(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %y.ext = fpext float %y to double
  %ceil = call double @ceil(double %x.ext) nounwind readnone
  %cmp = fcmp oeq double %y.ext, %ceil
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test8(
; CHECK-NEXT: %ceil = call float @llvm.ceil.f32(float %x)
; CHECK-NEXT: fcmp oeq float %ceil, %y
}

define i32 @test8_intrin(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %y.ext = fpext float %y to double
  %ceil = call double @llvm.ceil.f64(double %x.ext) nounwind readnone
  %cmp = fcmp oeq double %y.ext, %ceil
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test8_intrin(
; CHECK-NEXT: %ceil = call float @llvm.ceil.f32(float %x)
; CHECK-NEXT: fcmp oeq float %ceil, %y
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

define i32 @test9_intrin(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %y.ext = fpext float %y to double
  %fabs = call double @llvm.fabs.f64(double %x.ext) nounwind readnone
  %cmp = fcmp oeq double %y.ext, %fabs
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test9_intrin(
; CHECK-NEXT: %fabs = call float @llvm.fabs.f32(float %x)
; CHECK-NEXT: fcmp oeq float %fabs, %y
}

define i32 @test10(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %y.ext = fpext float %y to double
  %floor = call double @floor(double %x.ext) nounwind readnone
  %cmp = fcmp oeq double %floor, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test10(
; CHECK-NEXT: %floor = call float @llvm.floor.f32(float %x)
; CHECK-NEXT: fcmp oeq float %floor, %y
}

define i32 @test10_intrin(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %y.ext = fpext float %y to double
  %floor = call double @llvm.floor.f64(double %x.ext) nounwind readnone
  %cmp = fcmp oeq double %floor, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test10_intrin(
; CHECK-NEXT: %floor = call float @llvm.floor.f32(float %x)
; CHECK-NEXT: fcmp oeq float %floor, %y
}

define i32 @test11(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %y.ext = fpext float %y to double
  %nearbyint = call double @nearbyint(double %x.ext) nounwind
  %cmp = fcmp oeq double %nearbyint, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test11(
; CHECK-NEXT: %nearbyint = call float @llvm.nearbyint.f32(float %x)
; CHECK-NEXT: fcmp oeq float %nearbyint, %y
}


define i32 @test11_intrin(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %y.ext = fpext float %y to double
  %nearbyint = call double @llvm.nearbyint.f64(double %x.ext) nounwind
  %cmp = fcmp oeq double %nearbyint, %y.ext
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test11_intrin(
; CHECK-NEXT: %nearbyint = call float @llvm.nearbyint.f32(float %x)
; CHECK-NEXT: fcmp oeq float %nearbyint, %y
}

define i32 @test12(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %y.ext = fpext float %y to double
  %rint = call double @rint(double %x.ext) nounwind
  %cmp = fcmp oeq double %y.ext, %rint
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test12(
; CHECK-NEXT: %rint = call float @llvm.rint.f32(float %x)
; CHECK-NEXT: fcmp oeq float %rint, %y
}

define i32 @test13(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %y.ext = fpext float %y to double
  %round = call double @round(double %x.ext) nounwind readnone
  %cmp = fcmp oeq double %y.ext, %round
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test13(
; CHECK-NEXT: %round = call float @llvm.round.f32(float %x)
; CHECK-NEXT: fcmp oeq float %round, %y
}

define i32 @test13_intrin(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %y.ext = fpext float %y to double
  %round = call double @llvm.round.f64(double %x.ext) nounwind readnone
  %cmp = fcmp oeq double %y.ext, %round
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test13_intrin(
; CHECK-NEXT: %round = call float @llvm.round.f32(float %x)
; CHECK-NEXT: fcmp oeq float %round, %y
}

define i32 @test14(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %y.ext = fpext float %y to double
  %trunc = call double @trunc(double %x.ext) nounwind
  %cmp = fcmp oeq double %y.ext, %trunc
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test14(
; CHECK-NEXT: %trunc = call float @llvm.trunc.f32(float %x)
; CHECK-NEXT: fcmp oeq float %trunc, %y
}

define i32 @test14_intrin(float %x, float %y) nounwind uwtable {
  %x.ext = fpext float %x to double
  %y.ext = fpext float %y to double
  %trunc = call double @llvm.trunc.f64(double %x.ext) nounwind
  %cmp = fcmp oeq double %y.ext, %trunc
  %cmp.ext = zext i1 %cmp to i32
  ret i32 %cmp.ext
; CHECK-LABEL: @test14_intrin(
; CHECK-NEXT: %trunc = call float @llvm.trunc.f32(float %x)
; CHECK-NEXT: fcmp oeq float %trunc, %y
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

declare double @llvm.fabs.f64(double) nounwind readnone
declare double @llvm.ceil.f64(double) nounwind readnone
declare double @llvm.floor.f64(double) nounwind readnone
declare double @llvm.nearbyint.f64(double) nounwind readnone
declare double @llvm.round.f64(double) nounwind readnone
declare double @llvm.trunc.f64(double) nounwind readnone
