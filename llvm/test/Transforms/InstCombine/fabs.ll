; RUN: opt < %s -instcombine -S | FileCheck %s

; Make sure all library calls are eliminated when the input is known positive.

declare float @fabsf(float)
declare double @fabs(double)
declare fp128 @fabsl(fp128)
declare float @llvm.fma.f32(float, float, float)
declare float @llvm.fmuladd.f32(float, float, float)

define float @square_fabs_call_f32(float %x) {
  %mul = fmul float %x, %x
  %fabsf = tail call float @fabsf(float %mul)
  ret float %fabsf

; CHECK-LABEL: square_fabs_call_f32(
; CHECK-NEXT: %mul = fmul float %x, %x
; CHECK-NEXT: %fabsf = tail call float @fabsf(float %mul)
; CHECK-NEXT: ret float %fabsf
}

define double @square_fabs_call_f64(double %x) {
  %mul = fmul double %x, %x
  %fabs = tail call double @fabs(double %mul)
  ret double %fabs

; CHECK-LABEL: square_fabs_call_f64(
; CHECK-NEXT: %mul = fmul double %x, %x
; CHECK-NEXT: %fabs = tail call double @fabs(double %mul)
; CHECK-NEXT: ret double %fabs
}

define fp128 @square_fabs_call_f128(fp128 %x) {
  %mul = fmul fp128 %x, %x
  %fabsl = tail call fp128 @fabsl(fp128 %mul)
  ret fp128 %fabsl

; CHECK-LABEL: square_fabs_call_f128(
; CHECK-NEXT: %mul = fmul fp128 %x, %x
; CHECK-NEXT: %fabsl = tail call fp128 @fabsl(fp128 %mul)
; CHECK-NEXT: ret fp128 %fabsl
}

; Make sure all intrinsic calls are eliminated when the input is known
; positive.

declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare fp128 @llvm.fabs.f128(fp128)

; The fabs cannot be eliminated because %x may be a NaN
define float @square_fabs_intrinsic_f32(float %x) {
  %mul = fmul float %x, %x
  %fabsf = tail call float @llvm.fabs.f32(float %mul)
  ret float %fabsf

; CHECK-LABEL: square_fabs_intrinsic_f32(
; CHECK-NEXT: %mul = fmul float %x, %x
; CHECK-NEXT: %fabsf = tail call float @llvm.fabs.f32(float %mul)
; CHECK-NEXT: ret float %fabsf
}

define double @square_fabs_intrinsic_f64(double %x) {
  %mul = fmul double %x, %x
  %fabs = tail call double @llvm.fabs.f64(double %mul)
  ret double %fabs

; CHECK-LABEL: square_fabs_intrinsic_f64(
; CHECK-NEXT: %mul = fmul double %x, %x
; CHECK-NEXT: %fabs = tail call double @llvm.fabs.f64(double %mul)
; CHECK-NEXT: ret double %fabs
}

define fp128 @square_fabs_intrinsic_f128(fp128 %x) {
  %mul = fmul fp128 %x, %x
  %fabsl = tail call fp128 @llvm.fabs.f128(fp128 %mul)
  ret fp128 %fabsl

; CHECK-LABEL: square_fabs_intrinsic_f128(
; CHECK-NEXT: %mul = fmul fp128 %x, %x
; CHECK-NEXT: %fabsl = tail call fp128 @llvm.fabs.f128(fp128 %mul)
; CHECK-NEXT: ret fp128 %fabsl
}

define float @square_nnan_fabs_intrinsic_f32(float %x) {
  %mul = fmul nnan float %x, %x
  %fabsf = call float @llvm.fabs.f32(float %mul)
  ret float %fabsf

; CHECK-LABEL: square_nnan_fabs_intrinsic_f32(
; CHECK-NEXT: %mul = fmul nnan float %x, %x
; CHECK-NEXT: ret float %mul
}

; Shrinking a library call to a smaller type should not be inhibited by nor inhibit the square optimization.

define float @square_fabs_shrink_call1(float %x) {
  %ext = fpext float %x to double
  %sq = fmul double %ext, %ext
  %fabs = call double @fabs(double %sq)
  %trunc = fptrunc double %fabs to float
  ret float %trunc

; CHECK-LABEL: square_fabs_shrink_call1(
; CHECK-NEXT: %ext = fpext float %x to double
; CHECK-NEXT: %sq = fmul double %ext, %ext
; CHECK-NEXT: call double @fabs(double %sq)
; CHECK-NEXT: %trunc = fptrunc double %fabs to float
; CHECK-NEXT: ret float %trunc
}

define float @square_fabs_shrink_call2(float %x) {
  %sq = fmul float %x, %x
  %ext = fpext float %sq to double
  %fabs = call double @fabs(double %ext)
  %trunc = fptrunc double %fabs to float
  ret float %trunc

; CHECK-LABEL: square_fabs_shrink_call2(
; CHECK-NEXT: %sq = fmul float %x, %x
; CHECK-NEXT: %fabsf = call float @fabsf(float %sq)
; CHECK-NEXT: ret float %fabsf
}

; CHECK-LABEL: @fabs_select_constant_negative_positive(
; CHECK: %fabs = select i1 %cmp, float 1.000000e+00, float 2.000000e+00
; CHECK-NEXT: ret float %fabs
define float @fabs_select_constant_negative_positive(i32 %c) {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float -1.0, float 2.0
  %fabs = call float @llvm.fabs.f32(float %select)
  ret float %fabs
}

; CHECK-LABEL: @fabs_select_constant_positive_negative(
; CHECK: %fabs = select i1 %cmp, float 1.000000e+00, float 2.000000e+00
; CHECK-NEXT: ret float %fabs
define float @fabs_select_constant_positive_negative(i32 %c) {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float 1.0, float -2.0
  %fabs = call float @llvm.fabs.f32(float %select)
  ret float %fabs
}

; CHECK-LABEL: @fabs_select_constant_negative_negative(
; CHECK: %fabs = select i1 %cmp, float 1.000000e+00, float 2.000000e+00
; CHECK-NEXT: ret float %fabs
define float @fabs_select_constant_negative_negative(i32 %c) {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float -1.0, float -2.0
  %fabs = call float @llvm.fabs.f32(float %select)
  ret float %fabs
}

; CHECK-LABEL: @fabs_select_constant_neg0(
; CHECK-NEXT: ret float 0.0
define float @fabs_select_constant_neg0(i32 %c) {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float -0.0, float 0.0
  %fabs = call float @llvm.fabs.f32(float %select)
  ret float %fabs
}

; CHECK-LABEL: @fabs_select_var_constant_negative(
; CHECK: %select = select i1 %cmp, float %x, float -1.000000e+00
; CHECK: %fabs = call float @llvm.fabs.f32(float %select)
define float @fabs_select_var_constant_negative(i32 %c, float %x) {
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float %x, float -1.0
  %fabs = call float @llvm.fabs.f32(float %select)
  ret float %fabs
}

; The fabs cannot be eliminated because %x may be a NaN
define float @square_fma_fabs_intrinsic_f32(float %x) {
  %fma = call float @llvm.fma.f32(float %x, float %x, float 1.0)
  %fabsf = call float @llvm.fabs.f32(float %fma)
  ret float %fabsf

; CHECK-LABEL: @square_fma_fabs_intrinsic_f32(
; CHECK-NEXT: %fma = call float @llvm.fma.f32(float %x, float %x, float 1.000000e+00)
; CHECK-NEXT: %fabsf = call float @llvm.fabs.f32(float %fma)
; CHECK-NEXT: ret float %fabsf
}

; The fabs cannot be eliminated because %x may be a NaN
define float @square_nnan_fma_fabs_intrinsic_f32(float %x) {
  %fma = call nnan float @llvm.fma.f32(float %x, float %x, float 1.0)
  %fabsf = call float @llvm.fabs.f32(float %fma)
  ret float %fabsf

; CHECK-LABEL: @square_nnan_fma_fabs_intrinsic_f32(
; CHECK-NEXT: %fma = call nnan float @llvm.fma.f32(float %x, float %x, float 1.000000e+00)
; CHECK-NEXT: ret float %fma
}

define float @square_fmuladd_fabs_intrinsic_f32(float %x) {
  %fmuladd = call float @llvm.fmuladd.f32(float %x, float %x, float 1.0)
  %fabsf = call float @llvm.fabs.f32(float %fmuladd)
  ret float %fabsf

; CHECK-LABEL: @square_fmuladd_fabs_intrinsic_f32(
; CHECK-NEXT: %fmuladd = call float @llvm.fmuladd.f32(float %x, float %x, float 1.000000e+00)
; CHECK-NEXT: %fabsf = call float @llvm.fabs.f32(float %fmuladd)
; CHECK-NEXT: ret float %fabsf
}

define float @square_nnan_fmuladd_fabs_intrinsic_f32(float %x) {
  %fmuladd = call nnan float @llvm.fmuladd.f32(float %x, float %x, float 1.0)
  %fabsf = call float @llvm.fabs.f32(float %fmuladd)
  ret float %fabsf

; CHECK-LABEL: @square_nnan_fmuladd_fabs_intrinsic_f32(
; CHECK-NEXT: %fmuladd = call nnan float @llvm.fmuladd.f32(float %x, float %x, float 1.000000e+00)
; CHECK-NEXT: ret float %fmuladd
}
