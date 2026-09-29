; RUN: llc -enable-ppc-gen-scalar-mass -O3 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -enable-ppc-gen-scalar-mass -O3 -mtriple=powerpc-ibm-aix-xcoff < %s | FileCheck %s

declare float @acosf (float);
declare float @acoshf (float);
declare float @asinf (float);
declare float @asinhf (float);
declare float @atan2f (float, float);
declare float @atanf (float);
declare float @atanhf (float);
declare float @cbrtf (float);
declare float @copysignf (float, float);
declare float @cosf (float);
declare float @coshf (float);
declare float @erfcf (float);
declare float @erff (float);
declare float @expf (float);
declare float @expm1f (float);
declare float @hypotf (float, float);
declare float @lgammaf (float);
declare float @log10f (float);
declare float @log1pf (float);
declare float @logf (float);
declare float @powf (float, float);
declare float @rintf (float);
declare float @sinf (float);
declare float @sinhf (float);
declare float @tanf (float);
declare float @tanhf (float);
declare double @acos (double);
declare double @acosh (double);
declare double @anint (double);
declare double @asin (double);
declare double @asinh (double);
declare double @atan (double);
declare double @atan2 (double);
declare double @atanh (double);
declare double @cbrt (double);
declare double @copysign (double, double);
declare double @cos (double);
declare double @cosh (double);
declare double @cosisin (double);
declare double @dnint (double);
declare double @erf (double);
declare double @erfc (double);
declare double @exp (double);
declare double @expm1 (double);
declare double @hypot (double, double);
declare double @lgamma (double);
declare double @log (double);
declare double @log10 (double);
declare double @log1p (double);
declare double @pow (double, double);
declare double @rsqrt (double);
declare double @sin (double);
declare double @sincos (double);
declare double @sinh (double);
declare double @sqrt (double);
declare double @tan (double);
declare double @tanh (double);
declare float @__acosf_finite (float);
declare float @__acoshf_finite (float);
declare float @__asinf_finite (float);
declare float @__atan2f_finite (float, float);
declare float @__atanhf_finite (float);
declare float @__coshf_finite (float);
declare float @__expf_finite (float);
declare float @__logf_finite (float);
declare float @__log10f_finite (float);
declare float @__powf_finite (float, float);
declare float @__sinhf_finite (float);
declare double @__acos_finite (double);
declare double @__acosh_finite (double);
declare double @__asin_finite (double);
declare double @__atan2_finite (double, double);
declare double @__atanh_finite (double);
declare double @__cosh_finite (double);
declare double @__exp_finite (double);
declare double @__log_finite (double);
declare double @__log10_finite (double);
declare double @__pow_finite (double, double);
declare double @__sinh_finite (double);

; Without nnan ninf afn nsz flags on the call instruction
define float @acosf_f32_nofast(float %a) {
; CHECK-LABEL: acosf_f32_nofast
; CHECK-NOT: __xl_acosf_finite
; CHECK: blr
entry:
  %call = tail call float @acosf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @acoshf_f32_nofast(float %a) {
; CHECK-LABEL: acoshf_f32_nofast
; CHECK-NOT: __xl_acoshf_finite
; CHECK: blr
entry:
  %call = tail call float @acoshf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @asinf_f32_nofast(float %a) {
; CHECK-LABEL: asinf_f32_nofast
; CHECK-NOT: __xl_asinf_finite
; CHECK: blr
entry:
  %call = tail call float @asinf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @asinhf_f32_nofast(float %a) {
; CHECK-LABEL: asinhf_f32_nofast
; CHECK-NOT: __xl_asinhf_finite
; CHECK: blr
entry:
  %call = tail call float @asinhf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @atan2f_f32_nofast(float %a, float %b) {
; CHECK-LABEL: atan2f_f32_nofast
; CHECK-NOT: __xl_atan2f_finite
; CHECK: blr
entry:
  %call = tail call float @atan2f(float %a, float %b)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @atanf_f32_nofast(float %a) {
; CHECK-LABEL: atanf_f32_nofast
; CHECK-NOT: __xl_atanf_finite
; CHECK: blr
entry:
  %call = tail call float @atanf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @atanhf_f32_nofast(float %a) {
; CHECK-LABEL: atanhf_f32_nofast
; CHECK-NOT: __xl_atanhf_finite
; CHECK: blr
entry:
  %call = tail call float @atanhf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @cbrtf_f32_nofast(float %a) {
; CHECK-LABEL: cbrtf_f32_nofast
; CHECK-NOT: __xl_cbrtf_finite
; CHECK: blr
entry:
  %call = tail call float @cbrtf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @copysignf_f32_nofast(float %a, float %b) {
; CHECK-LABEL: copysignf_f32_nofast
; CHECK-NOT: __xl_copysignf_finite
; CHECK: blr
entry:
  %call = tail call float @copysignf(float %a, float %b)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @cosf_f32_nofast(float %a) {
; CHECK-LABEL: cosf_f32_nofast
; CHECK-NOT: __xl_cosf_finite
; CHECK: blr
entry:
  %call = tail call float @cosf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @coshf_f32_nofast(float %a) {
; CHECK-LABEL: coshf_f32_nofast
; CHECK-NOT: __xl_coshf_finite
; CHECK: blr
entry:
  %call = tail call float @coshf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @erfcf_f32_nofast(float %a) {
; CHECK-LABEL: erfcf_f32_nofast
; CHECK-NOT: __xl_erfcf_finite
; CHECK: blr
entry:
  %call = tail call float @erfcf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @erff_f32_nofast(float %a) {
; CHECK-LABEL: erff_f32_nofast
; CHECK-NOT: __xl_erff_finite
; CHECK: blr
entry:
  %call = tail call float @erff(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @expf_f32_nofast(float %a) {
; CHECK-LABEL: expf_f32_nofast
; CHECK-NOT: __xl_expf_finite
; CHECK: blr
entry:
  %call = tail call float @expf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @expm1f_f32_nofast(float %a) {
; CHECK-LABEL: expm1f_f32_nofast
; CHECK-NOT: __xl_expm1f_finite
; CHECK: blr
entry:
  %call = tail call float @expm1f(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @hypotf_f32_nofast(float %a, float %b) {
; CHECK-LABEL: hypotf_f32_nofast
; CHECK-NOT: __xl_hypotf_finite
; CHECK: blr
entry:
  %call = tail call float @hypotf(float %a, float %b)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @lgammaf_f32_nofast(float %a) {
; CHECK-LABEL: lgammaf_f32_nofast
; CHECK-NOT: __xl_lgammaf_finite
; CHECK: blr
entry:
  %call = tail call float @lgammaf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @log10f_f32_nofast(float %a) {
; CHECK-LABEL: log10f_f32_nofast
; CHECK-NOT: __xl_log10f_finite
; CHECK: blr
entry:
  %call = tail call float @log10f(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @log1pf_f32_nofast(float %a) {
; CHECK-LABEL: log1pf_f32_nofast
; CHECK-NOT: __xl_log1pf_finite
; CHECK: blr
entry:
  %call = tail call float @log1pf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @logf_f32_nofast(float %a) {
; CHECK-LABEL: logf_f32_nofast
; CHECK-NOT: __xl_logf_finite
; CHECK: blr
entry:
  %call = tail call float @logf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @powf_f32_nofast(float %a, float %b) {
; CHECK-LABEL: powf_f32_nofast
; CHECK-NOT: __xl_powf_finite
; CHECK: blr
entry:
  %call = tail call float @powf(float %a, float %b)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @rintf_f32_nofast(float %a) {
; CHECK-LABEL: rintf_f32_nofast
; CHECK-NOT: __xl_rintf_finite
; CHECK: blr
entry:
  %call = tail call float @rintf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @sinf_f32_nofast(float %a) {
; CHECK-LABEL: sinf_f32_nofast
; CHECK-NOT: __xl_sinf_finite
; CHECK: blr
entry:
  %call = tail call float @sinf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @sinhf_f32_nofast(float %a) {
; CHECK-LABEL: sinhf_f32_nofast
; CHECK-NOT: __xl_sinhf_finite
; CHECK: blr
entry:
  %call = tail call float @sinhf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @tanf_f32_nofast(float %a) {
; CHECK-LABEL: tanf_f32_nofast
; CHECK-NOT: __xl_tanf_finite
; CHECK: blr
entry:
  %call = tail call float @tanf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @tanhf_f32_nofast(float %a) {
; CHECK-LABEL: tanhf_f32_nofast
; CHECK-NOT: __xl_tanhf_finite
; CHECK: blr
entry:
  %call = tail call float @tanhf(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @acos_f64_nofast(double %a) {
; CHECK-LABEL: acos_f64_nofast
; CHECK-NOT: __xl_acos_finite
; CHECK: blr
entry:
  %call = tail call double @acos(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @acosh_f64_nofast(double %a) {
; CHECK-LABEL: acosh_f64_nofast
; CHECK-NOT: __xl_acosh_finite
; CHECK: blr
entry:
  %call = tail call double @acosh(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @anint_f64_nofast(double %a) {
; CHECK-LABEL: anint_f64_nofast
; CHECK-NOT: __xl_anint_finite
; CHECK: blr
entry:
  %call = tail call double @anint(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @asin_f64_nofast(double %a) {
; CHECK-LABEL: asin_f64_nofast
; CHECK-NOT: __xl_asin_finite
; CHECK: blr
entry:
  %call = tail call double @asin(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @asinh_f64_nofast(double %a) {
; CHECK-LABEL: asinh_f64_nofast
; CHECK-NOT: __xl_asinh_finite
; CHECK: blr
entry:
  %call = tail call double @asinh(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @atan_f64_nofast(double %a) {
; CHECK-LABEL: atan_f64_nofast
; CHECK-NOT: __xl_atan_finite
; CHECK: blr
entry:
  %call = tail call double @atan(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @atan2_f64_nofast(double %a) {
; CHECK-LABEL: atan2_f64_nofast
; CHECK-NOT: __xl_atan2_finite
; CHECK: blr
entry:
  %call = tail call double @atan2(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @atanh_f64_nofast(double %a) {
; CHECK-LABEL: atanh_f64_nofast
; CHECK-NOT: __xl_atanh_finite
; CHECK: blr
entry:
  %call = tail call double @atanh(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @cbrt_f64_nofast(double %a) {
; CHECK-LABEL: cbrt_f64_nofast
; CHECK-NOT: __xl_cbrt_finite
; CHECK: blr
entry:
  %call = tail call double @cbrt(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @copysign_f64_nofast(double %a, double %b) {
; CHECK-LABEL: copysign_f64_nofast
; CHECK-NOT: __xl_copysign_finite
; CHECK: blr
entry:
  %call = tail call double @copysign(double %a, double %b)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @cos_f64_nofast(double %a) {
; CHECK-LABEL: cos_f64_nofast
; CHECK-NOT: __xl_cos_finite
; CHECK: blr
entry:
  %call = tail call double @cos(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @cosh_f64_nofast(double %a) {
; CHECK-LABEL: cosh_f64_nofast
; CHECK-NOT: __xl_cosh_finite
; CHECK: blr
entry:
  %call = tail call double @cosh(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @cosisin_f64_nofast(double %a) {
; CHECK-LABEL: cosisin_f64_nofast
; CHECK-NOT: __xl_cosisin_finite
; CHECK: blr
entry:
  %call = tail call double @cosisin(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @dnint_f64_nofast(double %a) {
; CHECK-LABEL: dnint_f64_nofast
; CHECK-NOT: __xl_dnint_finite
; CHECK: blr
entry:
  %call = tail call double @dnint(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @erf_f64_nofast(double %a) {
; CHECK-LABEL: erf_f64_nofast
; CHECK-NOT: __xl_erf_finite
; CHECK: blr
entry:
  %call = tail call double @erf(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @erfc_f64_nofast(double %a) {
; CHECK-LABEL: erfc_f64_nofast
; CHECK-NOT: __xl_erfc_finite
; CHECK: blr
entry:
  %call = tail call double @erfc(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @exp_f64_nofast(double %a) {
; CHECK-LABEL: exp_f64_nofast
; CHECK-NOT: __xl_exp_finite
; CHECK: blr
entry:
  %call = tail call double @exp(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @expm1_f64_nofast(double %a) {
; CHECK-LABEL: expm1_f64_nofast
; CHECK-NOT: __xl_expm1_finite
; CHECK: blr
entry:
  %call = tail call double @expm1(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @hypot_f64_nofast(double %a, double %b) {
; CHECK-LABEL: hypot_f64_nofast
; CHECK-NOT: __xl_hypot_finite
; CHECK: blr
entry:
  %call = tail call double @hypot(double %a, double %b)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @lgamma_f64_nofast(double %a) {
; CHECK-LABEL: lgamma_f64_nofast
; CHECK-NOT: __xl_lgamma_finite
; CHECK: blr
entry:
  %call = tail call double @lgamma(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @log_f64_nofast(double %a) {
; CHECK-LABEL: log_f64_nofast
; CHECK-NOT: __xl_log_finite
; CHECK: blr
entry:
  %call = tail call double @log(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @log10_f64_nofast(double %a) {
; CHECK-LABEL: log10_f64_nofast
; CHECK-NOT: __xl_log10_finite
; CHECK: blr
entry:
  %call = tail call double @log10(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @log1p_f64_nofast(double %a) {
; CHECK-LABEL: log1p_f64_nofast
; CHECK-NOT: __xl_log1p_finite
; CHECK: blr
entry:
  %call = tail call double @log1p(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @pow_f64_nofast(double %a, double %b) {
; CHECK-LABEL: pow_f64_nofast
; CHECK-NOT: __xl_pow_finite
; CHECK: blr
entry:
  %call = tail call double @pow(double %a, double %b)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @rsqrt_f64_nofast(double %a) {
; CHECK-LABEL: rsqrt_f64_nofast
; CHECK-NOT: __xl_rsqrt_finite
; CHECK: blr
entry:
  %call = tail call double @rsqrt(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @sin_f64_nofast(double %a) {
; CHECK-LABEL: sin_f64_nofast
; CHECK-NOT: __xl_sin_finite
; CHECK: blr
entry:
  %call = tail call double @sin(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @sincos_f64_nofast(double %a) {
; CHECK-LABEL: sincos_f64_nofast
; CHECK-NOT: __xl_sincos_finite
; CHECK: blr
entry:
  %call = tail call double @sincos(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @sinh_f64_nofast(double %a) {
; CHECK-LABEL: sinh_f64_nofast
; CHECK-NOT: __xl_sinh_finite
; CHECK: blr
entry:
  %call = tail call double @sinh(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @sqrt_f64_nofast(double %a) {
; CHECK-LABEL: sqrt_f64_nofast
; CHECK-NOT: __xl_sqrt_finite
; CHECK: blr
entry:
  %call = tail call double @sqrt(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @tan_f64_nofast(double %a) {
; CHECK-LABEL: tan_f64_nofast
; CHECK-NOT: __xl_tan_finite
; CHECK: blr
entry:
  %call = tail call double @tan(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @tanh_f64_nofast(double %a) {
; CHECK-LABEL: tanh_f64_nofast
; CHECK-NOT: __xl_tanh_finite
; CHECK: blr
entry:
  %call = tail call double @tanh(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @__acosf_finite_f32_nofast(float %a) {
; CHECK-LABEL: __acosf_finite_f32_nofast
; CHECK-NOT: __xl_acosf_finite
; CHECK: blr
entry:
  %call = tail call float @__acosf_finite(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @__acoshf_finite_f32_nofast(float %a) {
; CHECK-LABEL: __acoshf_finite_f32_nofast
; CHECK-NOT: __xl_acoshf_finite
; CHECK: blr
entry:
  %call = tail call float @__acoshf_finite(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @__asinf_finite_f32_nofast(float %a) {
; CHECK-LABEL: __asinf_finite_f32_nofast
; CHECK-NOT: __xl_asinf_finite
; CHECK: blr
entry:
  %call = tail call float @__asinf_finite(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @__atan2f_finite_f32_nofast(float %a, float %b) {
; CHECK-LABEL: __atan2f_finite_f32_nofast
; CHECK-NOT: __xl_atan2f_finite
; CHECK: blr
entry:
  %call = tail call float @__atan2f_finite(float %a, float %b)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @__atanhf_finite_f32_nofast(float %a) {
; CHECK-LABEL: __atanhf_finite_f32_nofast
; CHECK-NOT: __xl_atanhf_finite
; CHECK: blr
entry:
  %call = tail call float @__atanhf_finite(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @__coshf_finite_f32_nofast(float %a) {
; CHECK-LABEL: __coshf_finite_f32_nofast
; CHECK-NOT: __xl_coshf_finite
; CHECK: blr
entry:
  %call = tail call float @__coshf_finite(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @__expf_finite_f32_nofast(float %a) {
; CHECK-LABEL: __expf_finite_f32_nofast
; CHECK-NOT: __xl_expf_finite
; CHECK: blr
entry:
  %call = tail call float @__expf_finite(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @__logf_finite_f32_nofast(float %a) {
; CHECK-LABEL: __logf_finite_f32_nofast
; CHECK-NOT: __xl_logf_finite
; CHECK: blr
entry:
  %call = tail call float @__logf_finite(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @__log10f_finite_f32_nofast(float %a) {
; CHECK-LABEL: __log10f_finite_f32_nofast
; CHECK-NOT: __xl_log10f_finite
; CHECK: blr
entry:
  %call = tail call float @__log10f_finite(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @__powf_finite_f32_nofast(float %a, float %b) {
; CHECK-LABEL: __powf_finite_f32_nofast
; CHECK-NOT: __xl_powf_finite
; CHECK: blr
entry:
  %call = tail call float @__powf_finite(float %a, float %b)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define float @__sinhf_finite_f32_nofast(float %a) {
; CHECK-LABEL: __sinhf_finite_f32_nofast
; CHECK-NOT: __xl_sinhf_finite
; CHECK: blr
entry:
  %call = tail call float @__sinhf_finite(float %a)
  ret float %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @__acos_finite_f64_nofast(double %a) {
; CHECK-LABEL: __acos_finite_f64_nofast
; CHECK-NOT: __xl_acos_finite
; CHECK: blr
entry:
  %call = tail call double @__acos_finite(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @__acosh_finite_f64_nofast(double %a) {
; CHECK-LABEL: __acosh_finite_f64_nofast
; CHECK-NOT: __xl_acosh_finite
; CHECK: blr
entry:
  %call = tail call double @__acosh_finite(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @__asin_finite_f64_nofast(double %a) {
; CHECK-LABEL: __asin_finite_f64_nofast
; CHECK-NOT: __xl_asin_finite
; CHECK: blr
entry:
  %call = tail call double @__asin_finite(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @__atan2_finite_f64_nofast(double %a, double %b) {
; CHECK-LABEL: __atan2_finite_f64_nofast
; CHECK-NOT: __xl_atan2_finite
; CHECK: blr
entry:
  %call = tail call double @__atan2_finite(double %a, double %b)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @__atanh_finite_f64_nofast(double %a) {
; CHECK-LABEL: __atanh_finite_f64_nofast
; CHECK-NOT: __xl_atanh_finite
; CHECK: blr
entry:
  %call = tail call double @__atanh_finite(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @__cosh_finite_f64_nofast(double %a) {
; CHECK-LABEL: __cosh_finite_f64_nofast
; CHECK-NOT: __xl_cosh_finite
; CHECK: blr
entry:
  %call = tail call double @__cosh_finite(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @__exp_finite_f64_nofast(double %a) {
; CHECK-LABEL: __exp_finite_f64_nofast
; CHECK-NOT: __xl_exp_finite
; CHECK: blr
entry:
  %call = tail call double @__exp_finite(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @__log_finite_f64_nofast(double %a) {
; CHECK-LABEL: __log_finite_f64_nofast
; CHECK-NOT: __xl_log_finite
; CHECK: blr
entry:
  %call = tail call double @__log_finite(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @__log10_finite_f64_nofast(double %a) {
; CHECK-LABEL: __log10_finite_f64_nofast
; CHECK-NOT: __xl_log10_finite
; CHECK: blr
entry:
  %call = tail call double @__log10_finite(double %a)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @__pow_finite_f64_nofast(double %a, double %b) {
; CHECK-LABEL: __pow_finite_f64_nofast
; CHECK-NOT: __xl_pow_finite
; CHECK: blr
entry:
  %call = tail call double @__pow_finite(double %a, double %b)
  ret double %call
}

; Without nnan ninf afn nsz flags on the call instruction
define double @__sinh_finite_f64_nofast(double %a) {
; CHECK-LABEL: __sinh_finite_f64_nofast
; CHECK-NOT: __xl_sinh_finite
; CHECK: blr
entry:
  %call = tail call double @__sinh_finite(double %a)
  ret double %call
}

