; RUN: llc < %s -O0 -o - | FileCheck %s
target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "armv7s-apple-ios7"

%foo = type <{ %Sf }>
%Sf = type <{ float }>

declare float @llvm.ceil.f32(float) 

; Check that we are not emitting a tail call for the last call to ceil.
; This function returns three different results.
; CHECK-LABEL: func1:
; CHECK-NOT: b _ceilf
; CHECK: pop
define { float, float, float } @func1() {
entry:
  %0 = alloca %foo, align 4
  %1 = alloca %foo, align 4
  %2 = alloca %foo, align 4
  store float 0.000000e+00, ptr %0, align 4
  store float 1.000000e+00, ptr %1, align 4
  store float 5.000000e+00, ptr %2, align 4
  br i1 true, label %3, label %4

; <label>:3                                       ; preds = %entry
  store float 2.000000e+00, ptr %1, align 4
  br label %4

; <label>:4                                       ; preds = %3, %entry
  %5 = call float @llvm.ceil.f32(float 5.000000e+00)
  %6 = load float, ptr %1, align 4
  %7 = call float @llvm.ceil.f32(float %6)
  %8 = insertvalue { float, float, float } { float 0.000000e+00, float undef, float undef }, float %5, 1
  %9 = insertvalue { float, float, float } %8, float %7, 2
  ret { float, float, float } %9
}

; Check that we are not emitting a tail call for the last call to ceil.
; This function returns two different results.
; CHECK-LABEL: func2:
; CHECK-NOT: b _ceilf
; CHECK: pop
define { float, float } @func2() {
entry:
  %0 = alloca %foo, align 4
  %1 = alloca %foo, align 4
  %2 = alloca %foo, align 4
  store float 0.000000e+00, ptr %0, align 4
  store float 1.000000e+00, ptr %1, align 4
  store float 5.000000e+00, ptr %2, align 4
  br i1 true, label %3, label %4

; <label>:3                                       ; preds = %entry
  store float 2.000000e+00, ptr %1, align 4
  br label %4

; <label>:4                                       ; preds = %3, %entry
  %5 = call float @llvm.ceil.f32(float 5.000000e+00)
  %6 = load float, ptr %1, align 4
  %7 = call float @llvm.ceil.f32(float %6)
  %8 = insertvalue { float, float } { float 0.000000e+00, float undef }, float %7, 1
  ret { float, float } %8
}

