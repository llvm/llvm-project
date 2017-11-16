; RUN: opt -mtriple=x86_64-unknown-linux-gnu < %s -instcombine -S | FileCheck %s

; Make sure libcalls are replaced with intrinsic calls.

declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare fp128 @llvm.fabs.f128(fp128)

declare float @fabsf(float)
declare double @fabs(double)
declare fp128 @fabsl(fp128)
declare float @llvm.fma.f32(float, float, float)
declare float @llvm.fmuladd.f32(float, float, float)

define float @replace_fabs_call_f32(float %x) {
; CHECK-LABEL: @replace_fabs_call_f32(
; CHECK-NEXT:    [[TMP1:%.*]] = call float @llvm.fabs.f32(float %x)
; CHECK-NEXT:    ret float [[TMP1]]
;
  %fabsf = tail call float @fabsf(float %x)
  ret float %fabsf
}

define double @replace_fabs_call_f64(double %x) {
; CHECK-LABEL: @replace_fabs_call_f64(
; CHECK-NEXT:    [[TMP1:%.*]] = call double @llvm.fabs.f64(double %x)
; CHECK-NEXT:    ret double [[TMP1]]
;
  %fabs = tail call double @fabs(double %x)
  ret double %fabs
}

define fp128 @replace_fabs_call_f128(fp128 %x) {
; CHECK-LABEL: @replace_fabs_call_f128(
; CHECK-NEXT:    [[TMP1:%.*]] = call fp128 @llvm.fabs.f128(fp128 %x)
; CHECK-NEXT:    ret fp128 [[TMP1]]
;
  %fabsl = tail call fp128 @fabsl(fp128 %x)
  ret fp128 %fabsl
}

; Make sure fast math flags are preserved when replacing the libcall.
define float @fmf_replace_fabs_call_f32(float %x) {
; CHECK-LABEL: @fmf_replace_fabs_call_f32(
; CHECK-NEXT:    [[TMP1:%.*]] = call nnan float @llvm.fabs.f32(float %x)
; CHECK-NEXT:    ret float [[TMP1]]
;
  %fabsf = tail call nnan float @fabsf(float %x)
  ret float %fabsf
}

; Make sure all intrinsic calls are eliminated when the input is known
; positive.

; The fabs cannot be eliminated because %x may be a NaN

define float @square_fabs_intrinsic_f32(float %x) {
; CHECK-LABEL: @square_fabs_intrinsic_f32(
; CHECK-NEXT:    [[MUL:%.*]] = fmul float %x, %x
; CHECK-NEXT:    [[FABSF:%.*]] = tail call float @llvm.fabs.f32(float [[MUL]])
; CHECK-NEXT:    ret float [[FABSF]]
;
  %mul = fmul float %x, %x
  %fabsf = tail call float @llvm.fabs.f32(float %mul)
  ret float %fabsf
}

define double @square_fabs_intrinsic_f64(double %x) {
; CHECK-LABEL: @square_fabs_intrinsic_f64(
; CHECK-NEXT:    [[MUL:%.*]] = fmul double %x, %x
; CHECK-NEXT:    [[FABS:%.*]] = tail call double @llvm.fabs.f64(double [[MUL]])
; CHECK-NEXT:    ret double [[FABS]]
;
  %mul = fmul double %x, %x
  %fabs = tail call double @llvm.fabs.f64(double %mul)
  ret double %fabs
}

define fp128 @square_fabs_intrinsic_f128(fp128 %x) {
; CHECK-LABEL: @square_fabs_intrinsic_f128(
; CHECK-NEXT:    [[MUL:%.*]] = fmul fp128 %x, %x
; CHECK-NEXT:    [[FABSL:%.*]] = tail call fp128 @llvm.fabs.f128(fp128 [[MUL]])
; CHECK-NEXT:    ret fp128 [[FABSL]]
;
  %mul = fmul fp128 %x, %x
  %fabsl = tail call fp128 @llvm.fabs.f128(fp128 %mul)
  ret fp128 %fabsl
}

define float @square_nnan_fabs_intrinsic_f32(float %x) {
; CHECK-LABEL: @square_nnan_fabs_intrinsic_f32(
; CHECK-NEXT:    [[MUL:%.*]] = fmul nnan float %x, %x
; CHECK-NEXT:    ret float [[MUL]]
;
  %mul = fmul nnan float %x, %x
  %fabsf = call float @llvm.fabs.f32(float %mul)
  ret float %fabsf
}

; Shrinking a library call to a smaller type should not be inhibited by nor inhibit the square optimization.

define float @square_fabs_shrink_call1(float %x) {
; CHECK-LABEL: @square_fabs_shrink_call1(
; CHECK-NEXT:    [[TMP1:%.*]] = fmul float %x, %x
; CHECK-NEXT:    [[TRUNC:%.*]] = call float @llvm.fabs.f32(float [[TMP1]])
; CHECK-NEXT:    ret float [[TRUNC]]
;
  %ext = fpext float %x to double
  %sq = fmul double %ext, %ext
  %fabs = call double @fabs(double %sq)
  %trunc = fptrunc double %fabs to float
  ret float %trunc
}

define float @square_fabs_shrink_call2(float %x) {
; CHECK-LABEL: @square_fabs_shrink_call2(
; CHECK-NEXT:    [[SQ:%.*]] = fmul float %x, %x
; CHECK-NEXT:    [[TRUNC:%.*]] = call float @llvm.fabs.f32(float [[SQ]])
; CHECK-NEXT:    ret float [[TRUNC]]
;
  %sq = fmul float %x, %x
  %ext = fpext float %sq to double
  %fabs = call double @fabs(double %ext)
  %trunc = fptrunc double %fabs to float
  ret float %trunc
}

define float @fabs_select_constant_negative_positive(i32 %c) {
; CHECK-LABEL: @fabs_select_constant_negative_positive(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 %c, 0
; CHECK-NEXT:    [[FABS:%.*]] = select i1 [[CMP]], float 1.000000e+00, float 2.000000e+00
; CHECK-NEXT:    ret float [[FABS]]
;
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float -1.0, float 2.0
  %fabs = call float @llvm.fabs.f32(float %select)
  ret float %fabs
}

define float @fabs_select_constant_positive_negative(i32 %c) {
; CHECK-LABEL: @fabs_select_constant_positive_negative(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 %c, 0
; CHECK-NEXT:    [[FABS:%.*]] = select i1 [[CMP]], float 1.000000e+00, float 2.000000e+00
; CHECK-NEXT:    ret float [[FABS]]
;
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float 1.0, float -2.0
  %fabs = call float @llvm.fabs.f32(float %select)
  ret float %fabs
}

define float @fabs_select_constant_negative_negative(i32 %c) {
; CHECK-LABEL: @fabs_select_constant_negative_negative(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 %c, 0
; CHECK-NEXT:    [[FABS:%.*]] = select i1 [[CMP]], float 1.000000e+00, float 2.000000e+00
; CHECK-NEXT:    ret float [[FABS]]
;
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float -1.0, float -2.0
  %fabs = call float @llvm.fabs.f32(float %select)
  ret float %fabs
}

define float @fabs_select_constant_neg0(i32 %c) {
; CHECK-LABEL: @fabs_select_constant_neg0(
; CHECK-NEXT:    ret float 0.000000e+00
;
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float -0.0, float 0.0
  %fabs = call float @llvm.fabs.f32(float %select)
  ret float %fabs
}

define float @fabs_select_var_constant_negative(i32 %c, float %x) {
; CHECK-LABEL: @fabs_select_var_constant_negative(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 %c, 0
; CHECK-NEXT:    [[SELECT:%.*]] = select i1 [[CMP]], float %x, float -1.000000e+00
; CHECK-NEXT:    [[FABS:%.*]] = call float @llvm.fabs.f32(float [[SELECT]])
; CHECK-NEXT:    ret float [[FABS]]
;
  %cmp = icmp eq i32 %c, 0
  %select = select i1 %cmp, float %x, float -1.0
  %fabs = call float @llvm.fabs.f32(float %select)
  ret float %fabs
}

; The fabs cannot be eliminated because %x may be a NaN

define float @square_fma_fabs_intrinsic_f32(float %x) {
; CHECK-LABEL: @square_fma_fabs_intrinsic_f32(
; CHECK-NEXT:    [[FMA:%.*]] = call float @llvm.fma.f32(float %x, float %x, float 1.000000e+00)
; CHECK-NEXT:    [[FABSF:%.*]] = call float @llvm.fabs.f32(float [[FMA]])
; CHECK-NEXT:    ret float [[FABSF]]
;
  %fma = call float @llvm.fma.f32(float %x, float %x, float 1.0)
  %fabsf = call float @llvm.fabs.f32(float %fma)
  ret float %fabsf
}

; The fabs cannot be eliminated because %x may be a NaN

define float @square_nnan_fma_fabs_intrinsic_f32(float %x) {
; CHECK-LABEL: @square_nnan_fma_fabs_intrinsic_f32(
; CHECK-NEXT:    [[FMA:%.*]] = call nnan float @llvm.fma.f32(float %x, float %x, float 1.000000e+00)
; CHECK-NEXT:    ret float [[FMA]]
;
  %fma = call nnan float @llvm.fma.f32(float %x, float %x, float 1.0)
  %fabsf = call float @llvm.fabs.f32(float %fma)
  ret float %fabsf
}

define float @square_fmuladd_fabs_intrinsic_f32(float %x) {
; CHECK-LABEL: @square_fmuladd_fabs_intrinsic_f32(
; CHECK-NEXT:    [[FMULADD:%.*]] = call float @llvm.fmuladd.f32(float %x, float %x, float 1.000000e+00)
; CHECK-NEXT:    [[FABSF:%.*]] = call float @llvm.fabs.f32(float [[FMULADD]])
; CHECK-NEXT:    ret float [[FABSF]]
;
  %fmuladd = call float @llvm.fmuladd.f32(float %x, float %x, float 1.0)
  %fabsf = call float @llvm.fabs.f32(float %fmuladd)
  ret float %fabsf
}

define float @square_nnan_fmuladd_fabs_intrinsic_f32(float %x) {
; CHECK-LABEL: @square_nnan_fmuladd_fabs_intrinsic_f32(
; CHECK-NEXT:    [[FMULADD:%.*]] = call nnan float @llvm.fmuladd.f32(float %x, float %x, float 1.000000e+00)
; CHECK-NEXT:    ret float [[FMULADD]]
;
  %fmuladd = call nnan float @llvm.fmuladd.f32(float %x, float %x, float 1.0)
  %fabsf = call float @llvm.fabs.f32(float %fmuladd)
  ret float %fabsf
}

; Don't introduce a second fpext

define double @multi_use_fabs_fpext(float %x) {
; CHECK-LABEL: @multi_use_fabs_fpext(
; CHECK-NEXT:    [[FPEXT:%.*]] = fpext float %x to double
; CHECK-NEXT:    [[FABS:%.*]] = call double @llvm.fabs.f64(double [[FPEXT]])
; CHECK-NEXT:    store volatile double [[FPEXT]], double* undef, align 8
; CHECK-NEXT:    ret double [[FABS]]
;
  %fpext = fpext float %x to double
  %fabs = call double @llvm.fabs.f64(double %fpext)
  store volatile double %fpext, double* undef
  ret double %fabs
}
