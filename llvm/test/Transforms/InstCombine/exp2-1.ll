; Test that the exp2 library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s -check-prefix=CHECK -check-prefix=INTRINSIC -check-prefix=LDEXP -check-prefix=LDEXPF
; RUN: opt < %s -instcombine -S -mtriple=i386-pc-win32 | FileCheck %s -check-prefix=INTRINSIC -check-prefix=LDEXP -check-prefix=NOLDEXPF
; RUN: opt < %s -instcombine -S -mtriple=amdgcn-unknown-unknown | FileCheck %s -check-prefix=INTRINSIC -check-prefix=NOLDEXP -check-prefix=NOLDEXPF

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare double @exp2(double)
declare float @exp2f(float)

; Check exp2(sitofp(x)) -> ldexp(1.0, sext(x)).

define double @test_simplify1(i32 %x) {
; CHECK-LABEL: @test_simplify1(
  %conv = sitofp i32 %x to double
  %ret = call double @exp2(double %conv)
; CHECK: call double @ldexp
  ret double %ret
}

define double @test_simplify2(i16 signext %x) {
; CHECK-LABEL: @test_simplify2(
  %conv = sitofp i16 %x to double
  %ret = call double @exp2(double %conv)
; CHECK: call double @ldexp
  ret double %ret
}

define double @test_simplify3(i8 signext %x) {
; CHECK-LABEL: @test_simplify3(
  %conv = sitofp i8 %x to double
  %ret = call double @exp2(double %conv)
; CHECK: call double @ldexp
  ret double %ret
}

define float @test_simplify4(i32 %x) {
; CHECK-LABEL: @test_simplify4(
  %conv = sitofp i32 %x to float
  %ret = call float @exp2f(float %conv)
; CHECK: call float @ldexpf
  ret float %ret
}

; Check exp2(uitofp(x)) -> ldexp(1.0, zext(x)).

define double @test_no_simplify1(i32 %x) {
; CHECK-LABEL: @test_no_simplify1(
  %conv = uitofp i32 %x to double
  %ret = call double @exp2(double %conv)
; CHECK: call double @exp2
  ret double %ret
}

define double @test_simplify6(i16 zeroext %x) {
; CHECK-LABEL: @test_simplify6(
  %conv = uitofp i16 %x to double
  %ret = call double @exp2(double %conv)
; CHECK: call double @ldexp
  ret double %ret
}

define double @test_simplify7(i8 zeroext %x) {
; CHECK-LABEL: @test_simplify7(
  %conv = uitofp i8 %x to double
  %ret = call double @exp2(double %conv)
; CHECK: call double @ldexp
  ret double %ret
}

define float @test_simplify8(i8 zeroext %x) {
; CHECK-LABEL: @test_simplify8(
  %conv = uitofp i8 %x to float
  %ret = call float @exp2f(float %conv)
; CHECK: call float @ldexpf
  ret float %ret
}

declare double @llvm.exp2.f64(double)
declare float @llvm.exp2.f32(float)

define double @test_simplify9(i8 zeroext %x) {
; INTRINSIC-LABEL: @test_simplify9(
  %conv = uitofp i8 %x to double
  %ret = call double @llvm.exp2.f64(double %conv)
; LDEXP: call double @ldexp
; NOLDEXP-NOT: call double @ldexp
  ret double %ret
}

define float @test_simplify10(i8 zeroext %x) {
; INTRINSIC-LABEL: @test_simplify10(
  %conv = uitofp i8 %x to float
  %ret = call float @llvm.exp2.f32(float %conv)
; LDEXPF: call float @ldexpf
; NOLDEXPF-NOT: call float @ldexpf
  ret float %ret
}
