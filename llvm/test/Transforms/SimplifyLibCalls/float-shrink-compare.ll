; RUN: opt -S -simplify-libcalls -instcombine %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

define i32 @test1(float %x, float %y) nounwind uwtable {
  %1 = fpext float %x to double
  %2 = call double @ceil(double %1) nounwind readnone
  %3 = fpext float %y to double
  %4 = fcmp oeq double %2, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK: @test1
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
; CHECK: @test2
; CHECK-NEXT: %fabsf = call float @fabsf(float %x)
; CHECK-NEXT: fcmp oeq float %fabsf, %y
}

define i32 @test3(float %x, float %y) nounwind uwtable {
  %1 = fpext float %x to double
  %2 = call double @floor(double %1) nounwind readnone
  %3 = fpext float %y to double
  %4 = fcmp oeq double %2, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK: @test3
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
; CHECK: @test4
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
; CHECK: @test5
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
; CHECK: @test6
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
; CHECK: @test7
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
; CHECK: @test8
; CHECK-NEXT: %ceilf = call float @ceilf(float %x)
; CHECK-NEXT: fcmp oeq float %ceilf, %y
}

define i32 @test9(float %x, float %y) nounwind uwtable {
  %1 = fpext float %y to double
  %2 = fpext float %x to double
  %3 = call double @fabs(double %2) nounwind readnone
  %4 = fcmp oeq double %1, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK: @test9
; CHECK-NEXT: %fabsf = call float @fabsf(float %x)
; CHECK-NEXT: fcmp oeq float %fabsf, %y
}

define i32 @test10(float %x, float %y) nounwind uwtable {
  %1 = fpext float %y to double
  %2 = fpext float %x to double
  %3 = call double @floor(double %2) nounwind readnone
  %4 = fcmp oeq double %1, %3
  %5 = zext i1 %4 to i32
  ret i32 %5
; CHECK: @test10
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
; CHECK: @test11
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
; CHECK: @test12
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
; CHECK: @test13
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
; CHECK: @test14
; CHECK-NEXT: %truncf = call float @truncf(float %x)
; CHECK-NEXT: fcmp oeq float %truncf, %y
}

declare double @fabs(double) nounwind readnone
declare double @ceil(double) nounwind readnone
declare double @floor(double) nounwind readnone
declare double @nearbyint(double) nounwind readnone
declare double @rint(double) nounwind readnone
declare double @round(double) nounwind readnone
declare double @trunc(double) nounwind readnone
