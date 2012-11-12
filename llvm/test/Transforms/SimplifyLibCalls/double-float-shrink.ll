; RUN: opt  < %s -simplify-libcalls -enable-double-float-shrink -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define float @acos_test(float %f) nounwind readnone {
; CHECK: acos_test
    %conv = fpext float %f to double
    %call = call double @acos(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @acosf(float %f)
}

define double @acos_test2(float %f) nounwind readnone {
; CHECK: acos_test2
    %conv = fpext float %f to double
    %call = call double @acos(double %conv)
    ret double %call
; CHECK: call double @acos(double %conv)
}

define float @acosh_test(float %f) nounwind readnone {
; CHECK: acosh_test
    %conv = fpext float %f to double
    %call = call double @acosh(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @acoshf(float %f)
}

define double @acosh_test2(float %f) nounwind readnone {
; CHECK: acosh_test2
    %conv = fpext float %f to double
    %call = call double @acosh(double %conv)
    ret double %call
; CHECK: call double @acosh(double %conv)
}

define float @asin_test(float %f) nounwind readnone {
; CHECK: asin_test
    %conv = fpext float %f to double
    %call = call double @asin(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @asinf(float %f)
}

define double @asin_test2(float %f) nounwind readnone {
; CHECK: asin_test2
    %conv = fpext float %f to double
    %call = call double @asin(double %conv)
    ret double %call
; CHECK: call double @asin(double %conv)
}

define float @asinh_test(float %f) nounwind readnone {
; CHECK: asinh_test
    %conv = fpext float %f to double
    %call = call double @asinh(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @asinhf(float %f)
}

define double @asinh_test2(float %f) nounwind readnone {
; CHECK: asinh_test2
    %conv = fpext float %f to double
    %call = call double @asinh(double %conv)
    ret double %call
; CHECK: call double @asinh(double %conv)
}

define float @atan_test(float %f) nounwind readnone {
; CHECK: atan_test
    %conv = fpext float %f to double
    %call = call double @atan(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @atanf(float %f)
}

define double @atan_test2(float %f) nounwind readnone {
; CHECK: atan_test2
    %conv = fpext float %f to double
    %call = call double @atan(double %conv)
    ret double %call
; CHECK: call double @atan(double %conv)
}
define float @atanh_test(float %f) nounwind readnone {
; CHECK: atanh_test
    %conv = fpext float %f to double
    %call = call double @atanh(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @atanhf(float %f)
}

define double @atanh_test2(float %f) nounwind readnone {
; CHECK: atanh_test2
    %conv = fpext float %f to double
    %call = call double @atanh(double %conv)
    ret double %call
; CHECK: call double @atanh(double %conv)
}
define float @cbrt_test(float %f) nounwind readnone {
; CHECK: cbrt_test
    %conv = fpext float %f to double
    %call = call double @cbrt(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @cbrtf(float %f)
}

define double @cbrt_test2(float %f) nounwind readnone {
; CHECK: cbrt_test2
    %conv = fpext float %f to double
    %call = call double @cbrt(double %conv)
    ret double %call
; CHECK: call double @cbrt(double %conv)
}
define float @exp_test(float %f) nounwind readnone {
; CHECK: exp_test
    %conv = fpext float %f to double
    %call = call double @exp(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @expf(float %f)
}

define double @exp_test2(float %f) nounwind readnone {
; CHECK: exp_test2
    %conv = fpext float %f to double
    %call = call double @exp(double %conv)
    ret double %call
; CHECK: call double @exp(double %conv)
}
define float @expm1_test(float %f) nounwind readnone {
; CHECK: expm1_test
    %conv = fpext float %f to double
    %call = call double @expm1(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @expm1f(float %f)
}

define double @expm1_test2(float %f) nounwind readnone {
; CHECK: expm1_test2
    %conv = fpext float %f to double
    %call = call double @expm1(double %conv)
    ret double %call
; CHECK: call double @expm1(double %conv)
}
define float @exp10_test(float %f) nounwind readnone {
; CHECK: exp10_test
    %conv = fpext float %f to double
    %call = call double @exp10(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @exp10f(float %f)
}

define double @exp10_test2(float %f) nounwind readnone {
; CHECK: exp10_test2
    %conv = fpext float %f to double
    %call = call double @exp10(double %conv)
    ret double %call
; CHECK: call double @exp10(double %conv)
}
define float @log_test(float %f) nounwind readnone {
; CHECK: log_test
    %conv = fpext float %f to double
    %call = call double @log(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @logf(float %f)
}

define double @log_test2(float %f) nounwind readnone {
; CHECK: log_test2
    %conv = fpext float %f to double
    %call = call double @log(double %conv)
    ret double %call
; CHECK: call double @log(double %conv)
}
define float @log10_test(float %f) nounwind readnone {
; CHECK: log10_test
    %conv = fpext float %f to double
    %call = call double @log10(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @log10f(float %f)
}

define double @log10_test2(float %f) nounwind readnone {
; CHECK: log10_test2
    %conv = fpext float %f to double
    %call = call double @log10(double %conv)
    ret double %call
; CHECK: call double @log10(double %conv)
}
define float @log1p_test(float %f) nounwind readnone {
; CHECK: log1p_test
    %conv = fpext float %f to double
    %call = call double @log1p(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @log1pf(float %f)
}

define double @log1p_test2(float %f) nounwind readnone {
; CHECK: log1p_test2
    %conv = fpext float %f to double
    %call = call double @log1p(double %conv)
    ret double %call
; CHECK: call double @log1p(double %conv)
}
define float @log2_test(float %f) nounwind readnone {
; CHECK: log2_test
    %conv = fpext float %f to double
    %call = call double @log2(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @log2f(float %f)
}

define double @log2_test2(float %f) nounwind readnone {
; CHECK: log2_test2
    %conv = fpext float %f to double
    %call = call double @log2(double %conv)
    ret double %call
; CHECK: call double @log2(double %conv)
}
define float @logb_test(float %f) nounwind readnone {
; CHECK: logb_test
    %conv = fpext float %f to double
    %call = call double @logb(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @logbf(float %f)
}

define double @logb_test2(float %f) nounwind readnone {
; CHECK: logb_test2
    %conv = fpext float %f to double
    %call = call double @logb(double %conv)
    ret double %call
; CHECK: call double @logb(double %conv)
}
define float @sin_test(float %f) nounwind readnone {
; CHECK: sin_test
    %conv = fpext float %f to double
    %call = call double @sin(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @sinf(float %f)
}

define double @sin_test2(float %f) nounwind readnone {
; CHECK: sin_test2
    %conv = fpext float %f to double
    %call = call double @sin(double %conv)
    ret double %call
; CHECK: call double @sin(double %conv)
}
define float @sqrt_test(float %f) nounwind readnone {
; CHECK: sqrt_test
    %conv = fpext float %f to double
    %call = call double @sqrt(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @sqrtf(float %f)
}

define double @sqrt_test2(float %f) nounwind readnone {
; CHECK: sqrt_test2
    %conv = fpext float %f to double
    %call = call double @sqrt(double %conv)
    ret double %call
; CHECK: call double @sqrt(double %conv)
}
define float @tan_test(float %f) nounwind readnone {
; CHECK: tan_test
    %conv = fpext float %f to double
    %call = call double @tan(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @tanf(float %f)
}

define double @tan_test2(float %f) nounwind readnone {
; CHECK: tan_test2
    %conv = fpext float %f to double
    %call = call double @tan(double %conv)
    ret double %call
; CHECK: call double @tan(double %conv)
}
define float @tanh_test(float %f) nounwind readnone {
; CHECK: tanh_test
    %conv = fpext float %f to double
    %call = call double @tanh(double %conv)
    %conv1 = fptrunc double %call to float
    ret float %conv1
; CHECK: call float @tanhf(float %f)
}

define double @tanh_test2(float %f) nounwind readnone {
; CHECK: tanh_test2
    %conv = fpext float %f to double
    %call = call double @tanh(double %conv)
    ret double %call
; CHECK: call double @tanh(double %conv)
}

declare double @tanh(double) nounwind readnone
declare double @tan(double) nounwind readnone
declare double @sqrt(double) nounwind readnone
declare double @sin(double) nounwind readnone
declare double @log2(double) nounwind readnone
declare double @log1p(double) nounwind readnone
declare double @log10(double) nounwind readnone
declare double @log(double) nounwind readnone
declare double @logb(double) nounwind readnone
declare double @exp10(double) nounwind readnone
declare double @expm1(double) nounwind readnone
declare double @exp(double) nounwind readnone
declare double @cbrt(double) nounwind readnone
declare double @atanh(double) nounwind readnone
declare double @atan(double) nounwind readnone
declare double @acos(double) nounwind readnone
declare double @acosh(double) nounwind readnone
declare double @asin(double) nounwind readnone
declare double @asinh(double) nounwind readnone
