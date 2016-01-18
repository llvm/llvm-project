; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Check for and against shrinkage when using the
; unsafe-fp-math function attribute on a math lib
; function. This optimization may be overridden by
; the -enable-double-float-shrink option.
; PR17850: http://llvm.org/bugs/show_bug.cgi?id=17850

define float @acos_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @acos(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: acos_test
; CHECK: call float @acosf(float %f)
}

define double @acos_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call double @acos(double %conv)
   ret double %call
; CHECK-LABEL: acos_test2
; CHECK: call double @acos(double %conv)
}

define float @acosh_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @acosh(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: acosh_test
; CHECK: call float @acoshf(float %f)
}

define double @acosh_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call double @acosh(double %conv)
   ret double %call
; CHECK-LABEL: acosh_test2
; CHECK: call double @acosh(double %conv)
}

define float @asin_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @asin(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: asin_test
; CHECK: call float @asinf(float %f)
}

define double @asin_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call double @asin(double %conv)
   ret double %call
; CHECK-LABEL: asin_test2
; CHECK: call double @asin(double %conv)
}

define float @asinh_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @asinh(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: asinh_test
; CHECK: call float @asinhf(float %f)
}

define double @asinh_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call double @asinh(double %conv)
   ret double %call
; CHECK-LABEL: asinh_test2
; CHECK: call double @asinh(double %conv)
}

define float @atan_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @atan(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: atan_test
; CHECK: call float @atanf(float %f)
}

define double @atan_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call double @atan(double %conv)
   ret double %call
; CHECK-LABEL: atan_test2
; CHECK: call double @atan(double %conv)
}
define float @atanh_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @atanh(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: atanh_test
; CHECK: call float @atanhf(float %f)
}

define double @atanh_test2(float %f)   {
    %conv = fpext float %f to double
    %call = call double @atanh(double %conv)
    ret double %call
; CHECK-LABEL: atanh_test2
; CHECK: call double @atanh(double %conv)
}
define float @cbrt_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @cbrt(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: cbrt_test
; CHECK: call float @cbrtf(float %f)
}

define double @cbrt_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call double @cbrt(double %conv)
   ret double %call
; CHECK-LABEL: cbrt_test2
; CHECK: call double @cbrt(double %conv)
}
define float @exp_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @exp(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: exp_test
; CHECK: call float @expf(float %f)
}

define double @exp_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call double @exp(double %conv)
   ret double %call
; CHECK-LABEL: exp_test2
; CHECK: call double @exp(double %conv)
}
define float @expm1_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @expm1(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: expm1_test
; CHECK: call float @expm1f(float %f)
}

define double @expm1_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call double @expm1(double %conv)
   ret double %call
; CHECK-LABEL: expm1_test2
; CHECK: call double @expm1(double %conv)
}
define float @exp10_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @exp10(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: exp10_test
; CHECK: call double @exp10(double %conv)
}

define double @exp10_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call double @exp10(double %conv)
   ret double %call
; CHECK-LABEL: exp10_test2
; CHECK: call double @exp10(double %conv)
}
define float @log_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @log(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: log_test
; CHECK: call float @logf(float %f)
}

define double @log_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call double @log(double %conv)
   ret double %call
; CHECK-LABEL: log_test2
; CHECK: call double @log(double %conv)
}
define float @log10_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @log10(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: log10_test
; CHECK: call float @log10f(float %f)
}

define double @log10_test2(float %f) {
   %conv = fpext float %f to double
   %call = call double @log10(double %conv)
   ret double %call
; CHECK-LABEL: log10_test2
; CHECK: call double @log10(double %conv)
}
define float @log1p_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @log1p(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: log1p_test
; CHECK: call float @log1pf(float %f)
}

define double @log1p_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call double @log1p(double %conv)
   ret double %call
; CHECK-LABEL: log1p_test2
; CHECK: call double @log1p(double %conv)
}
define float @log2_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @log2(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: log2_test
; CHECK: call float @log2f(float %f)
}

define double @log2_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call double @log2(double %conv)
   ret double %call
; CHECK-LABEL: log2_test2
; CHECK: call double @log2(double %conv)
}
define float @logb_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @logb(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: logb_test
; CHECK: call float @logbf(float %f)
}

define double @logb_test2(float %f)   {
   %conv = fpext float %f to double
   %call = call double @logb(double %conv)
   ret double %call
; CHECK-LABEL: logb_test2
; CHECK: call double @logb(double %conv)
}
define float @sin_test(float %f)   {
   %conv = fpext float %f to double
   %call = call double @sin(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: sin_test
; CHECK: call float @sinf(float %f)
}

define double @sin_test2(float %f) {
   %conv = fpext float %f to double
   %call = call double @sin(double %conv)
   ret double %call
; CHECK-LABEL: sin_test2
; CHECK: call double @sin(double %conv)
}

define float @sqrt_test(float %f) {
   %conv = fpext float %f to double
   %call = call double @sqrt(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: sqrt_test
; CHECK: call float @sqrtf(float %f)
}

define double @sqrt_test2(float %f) {
   %conv = fpext float %f to double
   %call = call double @sqrt(double %conv)
   ret double %call
; CHECK-LABEL: sqrt_test2
; CHECK: call double @sqrt(double %conv)
}

define float @sqrt_int_test(float %f) {
   %conv = fpext float %f to double
   %call = call double @llvm.sqrt.f64(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: sqrt_int_test
; CHECK: call float @llvm.sqrt.f32(float %f)
}

define double @sqrt_int_test2(float %f) {
   %conv = fpext float %f to double
   %call = call double @llvm.sqrt.f64(double %conv)
   ret double %call
; CHECK-LABEL: sqrt_int_test2
; CHECK: call double @llvm.sqrt.f64(double %conv)
}

define float @tan_test(float %f) {
   %conv = fpext float %f to double
   %call = call double @tan(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: tan_test
; CHECK: call float @tanf(float %f)
}

define double @tan_test2(float %f) {
   %conv = fpext float %f to double
   %call = call double @tan(double %conv)
   ret double %call
; CHECK-LABEL: tan_test2
; CHECK: call double @tan(double %conv)
}
define float @tanh_test(float %f) {
   %conv = fpext float %f to double
   %call = call double @tanh(double %conv)
   %conv1 = fptrunc double %call to float
   ret float %conv1
; CHECK-LABEL: tanh_test
; CHECK: call float @tanhf(float %f)
}

define double @tanh_test2(float %f) {
   %conv = fpext float %f to double
   %call = call double @tanh(double %conv)
   ret double %call
; CHECK-LABEL: tanh_test2
; CHECK: call double @tanh(double %conv)
}

declare double @tanh(double) #1
declare double @tan(double) #1

; sqrt is a special case: the shrinking optimization 
; is valid even without unsafe-fp-math.
declare double @sqrt(double) 
declare double @llvm.sqrt.f64(double) 

declare double @sin(double) #1
declare double @log2(double) #1
declare double @log1p(double) #1
declare double @log10(double) #1
declare double @log(double) #1
declare double @logb(double) #1
declare double @exp10(double) #1
declare double @expm1(double) #1
declare double @exp(double) #1
declare double @cbrt(double) #1
declare double @atanh(double) #1
declare double @atan(double) #1
declare double @acos(double) #1
declare double @acosh(double) #1
declare double @asin(double) #1
declare double @asinh(double) #1

attributes #1 = { "unsafe-fp-math"="true" }

