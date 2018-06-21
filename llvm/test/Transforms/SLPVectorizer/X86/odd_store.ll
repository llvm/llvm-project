; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

;int foo(char * restrict A, float * restrict B, float T) {
;  A[0] = (T * B[10] + 4.0);
;  A[1] = (T * B[11] + 5.0);
;  A[2] = (T * B[12] + 6.0);
;}

;CHECK-LABEL: @foo(
;CHECK-NOT: load <3 x float>
;CHECK-NOT: fmul <3 x float>
;CHECK-NOT: fpext <3 x float>
;CHECK-NOT: fadd <3 x double>
;CHECK-NOT: fptosi <3 x double>
;CHECK-NOT: store <3 x i8>
;CHECK: ret
define i32 @foo(i8* noalias nocapture %A, float* noalias nocapture %B, float %T) {
  %1 = getelementptr inbounds float, float* %B, i64 10
  %2 = load float, float* %1, align 4
  %3 = fmul float %2, %T
  %4 = fpext float %3 to double
  %5 = fadd double %4, 4.000000e+00
  %6 = fptosi double %5 to i8
  store i8 %6, i8* %A, align 1
  %7 = getelementptr inbounds float, float* %B, i64 11
  %8 = load float, float* %7, align 4
  %9 = fmul float %8, %T
  %10 = fpext float %9 to double
  %11 = fadd double %10, 5.000000e+00
  %12 = fptosi double %11 to i8
  %13 = getelementptr inbounds i8, i8* %A, i64 1
  store i8 %12, i8* %13, align 1
  %14 = getelementptr inbounds float, float* %B, i64 12
  %15 = load float, float* %14, align 4
  %16 = fmul float %15, %T
  %17 = fpext float %16 to double
  %18 = fadd double %17, 6.000000e+00
  %19 = fptosi double %18 to i8
  %20 = getelementptr inbounds i8, i8* %A, i64 2
  store i8 %19, i8* %20, align 1
  ret i32 undef
}

