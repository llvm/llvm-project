; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Check for and against shrinkage when using the
; unsafe-fp-math function attribute on a math lib
; function. This optimization may be overridden by
; the -enable-double-float-shrink option.
; PR17850: http://llvm.org/bugs/show_bug.cgi?id=17850

define float @acos_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @acos(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: acos_test1
; CHECK: call fast float @acosf(float %f)
}

define double @acos_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @acos(double %conv)
   ret double %call
; CHECK-LABEL: acos_test2
; CHECK: call fast double @acos(double %conv)
}

define float @acosh_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @acosh(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: acosh_test1
; CHECK: call fast float @acoshf(float %f)
}

define double @acosh_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @acosh(double %conv)
   ret double %call
; CHECK-LABEL: acosh_test2
; CHECK: call fast double @acosh(double %conv)
}

define float @asin_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @asin(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: asin_test1
; CHECK: call fast float @asinf(float %f)
}

define double @asin_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @asin(double %conv)
   ret double %call
; CHECK-LABEL: asin_test2
; CHECK: call fast double @asin(double %conv)
}

define float @asinh_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @asinh(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: asinh_test1
; CHECK: call fast float @asinhf(float %f)
}

define double @asinh_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @asinh(double %conv)
   ret double %call
; CHECK-LABEL: asinh_test2
; CHECK: call fast double @asinh(double %conv)
}

define float @atan_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @atan(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: atan_test1
; CHECK: call fast float @atanf(float %f)
}

define double @atan_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @atan(double %conv)
   ret double %call
; CHECK-LABEL: atan_test2
; CHECK: call fast double @atan(double %conv)
}

define float @atanh_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @atanh(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: atanh_test1
; CHECK: call fast float @atanhf(float %f)
}

define double @atanh_test2(float %f)   {
    %conv = fpext float %f to double
    %call = call fast double @atanh(double %conv)
    ret double %call
; CHECK-LABEL: atanh_test2
; CHECK: call fast double @atanh(double %conv)
}

define float @cbrt_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @cbrt(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: cbrt_test1
; CHECK: call fast float @cbrtf(float %f)
}

define double @cbrt_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call fast  double @cbrt(double %conv)
   ret double %call
; CHECK-LABEL: cbrt_test2
; CHECK: call fast double @cbrt(double %conv)
}

define float @exp_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @exp(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: exp_test1
; CHECK: call fast float @expf(float %f)
}

define double @exp_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @exp(double %conv)
   ret double %call
; CHECK-LABEL: exp_test2
; CHECK: call fast double @exp(double %conv)
}

define float @expm1_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @expm1(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: expm1_test1
; CHECK: call fast float @expm1f(float %f)
}

define double @expm1_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @expm1(double %conv)
   ret double %call
; CHECK-LABEL: expm1_test2
; CHECK: call fast double @expm1(double %conv)
}

; exp10f() doesn't exist for this triple, so it doesn't shrink.

define float @exp10_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @exp10(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: exp10_test1
; CHECK: call fast double @exp10(double %conv)
}

define double @exp10_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @exp10(double %conv)
   ret double %call
; CHECK-LABEL: exp10_test2
; CHECK: call fast double @exp10(double %conv)
}

define float @log_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @log(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: log_test1
; CHECK: call fast float @logf(float %f)
}

define double @log_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @log(double %conv)
   ret double %call
; CHECK-LABEL: log_test2
; CHECK: call fast double @log(double %conv)
}

define float @log10_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @log10(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: log10_test1
; CHECK: call fast float @log10f(float %f)
}

define double @log10_test2(float %f) {
   %conv = fpext float %f to double
   %call = call fast double @log10(double %conv)
   ret double %call
; CHECK-LABEL: log10_test2
; CHECK: call fast double @log10(double %conv)
}

define float @log1p_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @log1p(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: log1p_test1
; CHECK: call fast float @log1pf(float %f)
}

define double @log1p_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @log1p(double %conv)
   ret double %call
; CHECK-LABEL: log1p_test2
; CHECK: call fast double @log1p(double %conv)
}

define float @log2_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @log2(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: log2_test1
; CHECK: call fast float @log2f(float %f)
}

define double @log2_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @log2(double %conv)
   ret double %call
; CHECK-LABEL: log2_test2
; CHECK: call fast double @log2(double %conv)
}

define float @logb_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @logb(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: logb_test1
; CHECK: call fast float @logbf(float %f)
}

define double @logb_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @logb(double %conv)
   ret double %call
; CHECK-LABEL: logb_test2
; CHECK: call fast double @logb(double %conv)
}

define float @sin_test1(float %f)   {
   %conv = fpext float %f to double
   %call = call fast double @sin(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: sin_test1
; CHECK: call fast float @sinf(float %f)
}

define double @sin_test2(float %f) {
   %conv = fpext float %f to double
   %call = call fast double @sin(double %conv)
   ret double %call
; CHECK-LABEL: sin_test2
; CHECK: call fast double @sin(double %conv)
}

define float @sqrt_test1(float %f) {
   %conv = fpext float %f to double
   %call = call double @sqrt(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: sqrt_test1
; CHECK: call float @sqrtf(float %f)
}

define double @sqrt_test2(float %f) {
   %conv = fpext float %f to double
   %call = call double @sqrt(double %conv)
   ret double %call
; CHECK-LABEL: sqrt_test2
; CHECK: call double @sqrt(double %conv)
}

define float @sqrt_int_test1(float %f) {
   %conv = fpext float %f to double
   %call = call double @llvm.sqrt.f64(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: sqrt_int_test1
; CHECK: call float @llvm.sqrt.f32(float %f)
}

define double @sqrt_int_test2(float %f) {
   %conv = fpext float %f to double
   %call = call double @llvm.sqrt.f64(double %conv)
   ret double %call
; CHECK-LABEL: sqrt_int_test2
; CHECK: call double @llvm.sqrt.f64(double %conv)
}

define float @tan_test1(float %f) {
   %conv = fpext float %f to double
   %call = call fast double @tan(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: tan_test1
; CHECK: call fast float @tanf(float %f)
}

define double @tan_test2(float %f) {
   %conv = fpext float %f to double
   %call = call fast double @tan(double %conv)
   ret double %call
; CHECK-LABEL: tan_test2
; CHECK: call fast double @tan(double %conv)
}
define float @tanh_test1(float %f) {
   %conv = fpext float %f to double
   %call = call fast double @tanh(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: tanh_test1
; CHECK: call fast float @tanhf(float %f)
}

define double @tanh_test2(float %f) {
   %conv = fpext float %f to double
   %call = call fast double @tanh(double %conv)
   ret double %call
; CHECK-LABEL: tanh_test2
; CHECK: call fast double @tanh(double %conv)
}

; 'arcp' on an fmax() is meaningless. This test just proves that
; flags are propagated for shrunken *binary* double FP calls.
define float @max1(float %a, float %b) {
  %c = fpext float %a to double
  %d = fpext float %b to double
  %e = call arcp double @fmax(double %c, double %d)
  %f = fptrunc double %e to float
  ret float %f

; CHECK-LABEL: max1(
; CHECK-NEXT:  call arcp float @fmaxf(float %a, float %b)
; CHECK-NEXT:  ret
}

declare double @fmax(double, double)

declare double @tanh(double)
declare double @tan(double)

; sqrt is a special case: the shrinking optimization 
; is valid even without unsafe-fp-math.
declare double @sqrt(double) 
declare double @llvm.sqrt.f64(double) 

declare double @sin(double)
declare double @log2(double)
declare double @log1p(double)
declare double @log10(double)
declare double @log(double)
declare double @logb(double)
declare double @exp10(double)
declare double @expm1(double)
declare double @exp(double)
declare double @cbrt(double)
declare double @atanh(double)
declare double @atan(double)
declare double @acos(double)
declare double @acosh(double)
declare double @asin(double)
declare double @asinh(double)

