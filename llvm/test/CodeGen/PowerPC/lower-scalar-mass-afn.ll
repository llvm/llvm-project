; RUN: llc -O3 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -O3 -mtriple=powerpc-ibm-aix-xcoff < %s | FileCheck %s

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
declare double @atan2 (double, double);
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

define float @acosf_f32(float %a) #0 {
; CHECK-LABEL: acosf_f32
; CHECK: __xl_acosf
; CHECK: blr
entry:
  %call = tail call afn float @acosf(float %a)
  ret float %call
}

define float @acoshf_f32(float %a) #0 {
; CHECK-LABEL: acoshf_f32
; CHECK: __xl_acoshf
; CHECK: blr
entry:
  %call = tail call afn float @acoshf(float %a)
  ret float %call
}

define float @asinf_f32(float %a) #0 {
; CHECK-LABEL: asinf_f32
; CHECK: __xl_asinf
; CHECK: blr
entry:
  %call = tail call afn float @asinf(float %a)
  ret float %call
}

define float @asinhf_f32(float %a) #0 {
; CHECK-LABEL: asinhf_f32
; CHECK: __xl_asinhf
; CHECK: blr
entry:
  %call = tail call afn float @asinhf(float %a)
  ret float %call
}

define float @atan2f_f32(float %a, float %b) #0 {
; CHECK-LABEL: atan2f_f32
; CHECK: __xl_atan2f
; CHECK: blr
entry:
  %call = tail call afn float @atan2f(float %a, float %b)
  ret float %call
}

define float @atanf_f32(float %a) #0 {
; CHECK-LABEL: atanf_f32
; CHECK: __xl_atanf
; CHECK: blr
entry:
  %call = tail call afn float @atanf(float %a)
  ret float %call
}

define float @atanhf_f32(float %a) #0 {
; CHECK-LABEL: atanhf_f32
; CHECK: __xl_atanhf
; CHECK: blr
entry:
  %call = tail call afn float @atanhf(float %a)
  ret float %call
}

define float @cbrtf_f32(float %a) #0 {
; CHECK-LABEL: cbrtf_f32
; CHECK: __xl_cbrtf
; CHECK: blr
entry:
  %call = tail call afn float @cbrtf(float %a)
  ret float %call
}

define float @copysignf_f32(float %a, float %b) #0 {
; CHECK-LABEL: copysignf_f32
; CHECK: copysignf
; CHECK: blr
entry:
  %call = tail call afn float @copysignf(float %a, float %b)
  ret float %call
}

define float @cosf_f32(float %a) #0 {
; CHECK-LABEL: cosf_f32
; CHECK: __xl_cosf
; CHECK: blr
entry:
  %call = tail call afn float @cosf(float %a)
  ret float %call
}

define float @coshf_f32(float %a) #0 {
; CHECK-LABEL: coshf_f32
; CHECK: __xl_coshf
; CHECK: blr
entry:
  %call = tail call afn float @coshf(float %a)
  ret float %call
}

define float @erfcf_f32(float %a) #0 {
; CHECK-LABEL: erfcf_f32
; CHECK: __xl_erfcf
; CHECK: blr
entry:
  %call = tail call afn float @erfcf(float %a)
  ret float %call
}

define float @erff_f32(float %a) #0 {
; CHECK-LABEL: erff_f32
; CHECK: __xl_erff
; CHECK: blr
entry:
  %call = tail call afn float @erff(float %a)
  ret float %call
}

define float @expf_f32(float %a) #0 {
; CHECK-LABEL: expf_f32
; CHECK: __xl_expf
; CHECK: blr
entry:
  %call = tail call afn float @expf(float %a)
  ret float %call
}

define float @expm1f_f32(float %a) #0 {
; CHECK-LABEL: expm1f_f32
; CHECK: __xl_expm1f
; CHECK: blr
entry:
  %call = tail call afn float @expm1f(float %a)
  ret float %call
}

define float @hypotf_f32(float %a, float %b) #0 {
; CHECK-LABEL: hypotf_f32
; CHECK: __xl_hypotf
; CHECK: blr
entry:
  %call = tail call afn float @hypotf(float %a, float %b)
  ret float %call
}

define float @lgammaf_f32(float %a) #0 {
; CHECK-LABEL: lgammaf_f32
; CHECK: __xl_lgammaf
; CHECK: blr
entry:
  %call = tail call afn float @lgammaf(float %a)
  ret float %call
}

define float @log10f_f32(float %a) #0 {
; CHECK-LABEL: log10f_f32
; CHECK: __xl_log10f
; CHECK: blr
entry:
  %call = tail call afn float @log10f(float %a)
  ret float %call
}

define float @log1pf_f32(float %a) #0 {
; CHECK-LABEL: log1pf_f32
; CHECK: __xl_log1pf
; CHECK: blr
entry:
  %call = tail call afn float @log1pf(float %a)
  ret float %call
}

define float @logf_f32(float %a) #0 {
; CHECK-LABEL: logf_f32
; CHECK: __xl_logf
; CHECK: blr
entry:
  %call = tail call afn float @logf(float %a)
  ret float %call
}

define float @powf_f32(float %a, float %b) #0 {
; CHECK-LABEL: powf_f32
; CHECK: __xl_powf
; CHECK: blr
entry:
  %call = tail call afn float @powf(float %a, float %b)
  ret float %call
}

define float @rintf_f32(float %a) #0 {
; CHECK-LABEL: rintf_f32
; CHECK-NOT: __xl_rintf
; CHECK: blr
entry:
  %call = tail call afn float @rintf(float %a)
  ret float %call
}

define float @sinf_f32(float %a) #0 {
; CHECK-LABEL: sinf_f32
; CHECK: __xl_sinf
; CHECK: blr
entry:
  %call = tail call afn float @sinf(float %a)
  ret float %call
}

define float @sinhf_f32(float %a) #0 {
; CHECK-LABEL: sinhf_f32
; CHECK: __xl_sinhf
; CHECK: blr
entry:
  %call = tail call afn float @sinhf(float %a)
  ret float %call
}

define float @tanf_f32(float %a) #0 {
; CHECK-LABEL: tanf_f32
; CHECK: __xl_tanf
; CHECK: blr
entry:
  %call = tail call afn float @tanf(float %a)
  ret float %call
}

define float @tanhf_f32(float %a) #0 {
; CHECK-LABEL: tanhf_f32
; CHECK: __xl_tanhf
; CHECK: blr
entry:
  %call = tail call afn float @tanhf(float %a)
  ret float %call
}

define double @acos_f64(double %a) #0 {
; CHECK-LABEL: acos_f64
; CHECK: __xl_acos
; CHECK: blr
entry:
  %call = tail call afn double @acos(double %a)
  ret double %call
}

define double @acosh_f64(double %a) #0 {
; CHECK-LABEL: acosh_f64
; CHECK: __xl_acosh
; CHECK: blr
entry:
  %call = tail call afn double @acosh(double %a)
  ret double %call
}

define double @anint_f64(double %a) #0 {
; CHECK-LABEL: anint_f64
; CHECK-NOT: __xl_anint
; CHECK: blr
entry:
  %call = tail call afn double @anint(double %a)
  ret double %call
}

define double @asin_f64(double %a) #0 {
; CHECK-LABEL: asin_f64
; CHECK: __xl_asin
; CHECK: blr
entry:
  %call = tail call afn double @asin(double %a)
  ret double %call
}

define double @asinh_f64(double %a) #0 {
; CHECK-LABEL: asinh_f64
; CHECK: __xl_asinh
; CHECK: blr
entry:
  %call = tail call afn double @asinh(double %a)
  ret double %call
}

define double @atan_f64(double %a) #0 {
; CHECK-LABEL: atan_f64
; CHECK: __xl_atan
; CHECK: blr
entry:
  %call = tail call afn double @atan(double %a)
  ret double %call
}

define double @atan2_f64(double %a, double %b) #0 {
; CHECK-LABEL: atan2_f64
; CHECK: __xl_atan2
; CHECK: blr
entry:
  %call = tail call afn double @atan2(double %a, double %b)
  ret double %call
}

define double @atanh_f64(double %a) #0 {
; CHECK-LABEL: atanh_f64
; CHECK: __xl_atanh
; CHECK: blr
entry:
  %call = tail call afn double @atanh(double %a)
  ret double %call
}

define double @cbrt_f64(double %a) #0 {
; CHECK-LABEL: cbrt_f64
; CHECK: __xl_cbrt
; CHECK: blr
entry:
  %call = tail call afn double @cbrt(double %a)
  ret double %call
}

define double @copysign_f64(double %a, double %b) #0 {
; CHECK-LABEL: copysign_f64
; CHECK: copysign
; CHECK: blr
entry:
  %call = tail call afn double @copysign(double %a, double %b)
  ret double %call
}

define double @cos_f64(double %a) #0 {
; CHECK-LABEL: cos_f64
; CHECK: __xl_cos
; CHECK: blr
entry:
  %call = tail call afn double @cos(double %a)
  ret double %call
}

define double @cosh_f64(double %a) #0 {
; CHECK-LABEL: cosh_f64
; CHECK: __xl_cosh
; CHECK: blr
entry:
  %call = tail call afn double @cosh(double %a)
  ret double %call
}

define double @cosisin_f64(double %a) #0 {
; CHECK-LABEL: cosisin_f64
; CHECK-NOT: __xl_cosisin
; CHECK: blr
entry:
  %call = tail call afn double @cosisin(double %a)
  ret double %call
}

define double @dnint_f64(double %a) #0 {
; CHECK-LABEL: dnint_f64
; CHECK-NOT: __xl_dnint
; CHECK: blr
entry:
  %call = tail call afn double @dnint(double %a)
  ret double %call
}

define double @erf_f64(double %a) #0 {
; CHECK-LABEL: erf_f64
; CHECK: __xl_erf
; CHECK: blr
entry:
  %call = tail call afn double @erf(double %a)
  ret double %call
}

define double @erfc_f64(double %a) #0 {
; CHECK-LABEL: erfc_f64
; CHECK: __xl_erfc
; CHECK: blr
entry:
  %call = tail call afn double @erfc(double %a)
  ret double %call
}

define double @exp_f64(double %a) #0 {
; CHECK-LABEL: exp_f64
; CHECK: __xl_exp
; CHECK: blr
entry:
  %call = tail call afn double @exp(double %a)
  ret double %call
}

define double @expm1_f64(double %a) #0 {
; CHECK-LABEL: expm1_f64
; CHECK: __xl_expm1
; CHECK: blr
entry:
  %call = tail call afn double @expm1(double %a)
  ret double %call
}

define double @hypot_f64(double %a, double %b) #0 {
; CHECK-LABEL: hypot_f64
; CHECK: __xl_hypot
; CHECK: blr
entry:
  %call = tail call afn double @hypot(double %a, double %b)
  ret double %call
}

define double @lgamma_f64(double %a) #0 {
; CHECK-LABEL: lgamma_f64
; CHECK: __xl_lgamma
; CHECK: blr
entry:
  %call = tail call afn double @lgamma(double %a)
  ret double %call
}

define double @log_f64(double %a) #0 {
; CHECK-LABEL: log_f64
; CHECK: __xl_log
; CHECK: blr
entry:
  %call = tail call afn double @log(double %a)
  ret double %call
}

define double @log10_f64(double %a) #0 {
; CHECK-LABEL: log10_f64
; CHECK: __xl_log10
; CHECK: blr
entry:
  %call = tail call afn double @log10(double %a)
  ret double %call
}

define double @log1p_f64(double %a) #0 {
; CHECK-LABEL: log1p_f64
; CHECK: __xl_log1p
; CHECK: blr
entry:
  %call = tail call afn double @log1p(double %a)
  ret double %call
}

define double @pow_f64(double %a, double %b) #0 {
; CHECK-LABEL: pow_f64
; CHECK: __xl_pow
; CHECK: blr
entry:
  %call = tail call afn double @pow(double %a, double %b)
  ret double %call
}

define double @rsqrt_f64(double %a) #0 {
; CHECK-LABEL: rsqrt_f64
; CHECK: __xl_rsqrt
; CHECK: blr
entry:
  %call = tail call afn double @rsqrt(double %a)
  ret double %call
}

define double @sin_f64(double %a) #0 {
; CHECK-LABEL: sin_f64
; CHECK: __xl_sin
; CHECK: blr
entry:
  %call = tail call afn double @sin(double %a)
  ret double %call
}

define double @sincos_f64(double %a) #0 {
; CHECK-LABEL: sincos_f64
; CHECK-NOT: __xl_sincos
; CHECK: blr
entry:
  %call = tail call afn double @sincos(double %a)
  ret double %call
}

define double @sinh_f64(double %a) #0 {
; CHECK-LABEL: sinh_f64
; CHECK: __xl_sinh
; CHECK: blr
entry:
  %call = tail call afn double @sinh(double %a)
  ret double %call
}

define double @sqrt_f64(double %a) #0 {
; CHECK-LABEL: sqrt_f64
; CHECK: __xl_sqrt
; CHECK: blr
entry:
  %call = tail call afn double @sqrt(double %a)
  ret double %call
}

define double @tan_f64(double %a) #0 {
; CHECK-LABEL: tan_f64
; CHECK: __xl_tan
; CHECK: blr
entry:
  %call = tail call afn double @tan(double %a)
  ret double %call
}

define double @tanh_f64(double %a) #0 {
; CHECK-LABEL: tanh_f64
; CHECK: __xl_tanh
; CHECK: blr
entry:
  %call = tail call afn double @tanh(double %a)
  ret double %call
}

define float @__acosf_finite_f32(float %a) #0 {
; CHECK-LABEL: __acosf_finite_f32
; CHECK: __xl_acosf
; CHECK: blr
entry:
  %call = tail call afn float @__acosf_finite(float %a)
  ret float %call
}

define float @__acoshf_finite_f32(float %a) #0 {
; CHECK-LABEL: __acoshf_finite_f32
; CHECK: __xl_acoshf
; CHECK: blr
entry:
  %call = tail call afn float @__acoshf_finite(float %a)
  ret float %call
}

define float @__asinf_finite_f32(float %a) #0 {
; CHECK-LABEL: __asinf_finite_f32
; CHECK: __xl_asinf
; CHECK: blr
entry:
  %call = tail call afn float @__asinf_finite(float %a)
  ret float %call
}

define float @__atan2f_finite_f32(float %a, float %b) #0 {
; CHECK-LABEL: __atan2f_finite_f32
; CHECK: __xl_atan2f
; CHECK: blr
entry:
  %call = tail call afn float @__atan2f_finite(float %a, float %b)
  ret float %call
}

define float @__atanhf_finite_f32(float %a) #0 {
; CHECK-LABEL: __atanhf_finite_f32
; CHECK: __xl_atanhf
; CHECK: blr
entry:
  %call = tail call afn float @__atanhf_finite(float %a)
  ret float %call
}

define float @__coshf_finite_f32(float %a) #0 {
; CHECK-LABEL: __coshf_finite_f32
; CHECK: __xl_coshf
; CHECK: blr
entry:
  %call = tail call afn float @__coshf_finite(float %a)
  ret float %call
}
define float @__expf_finite_f32(float %a) #0 {
; CHECK-LABEL: __expf_finite_f32
; CHECK: __xl_expf
; CHECK: blr
entry:
  %call = tail call afn float @__expf_finite(float %a)
  ret float %call
}
define float @__logf_finite_f32(float %a) #0 {
; CHECK-LABEL: __logf_finite_f32
; CHECK: __xl_logf
; CHECK: blr
entry:
  %call = tail call afn float @__logf_finite(float %a)
  ret float %call
}
define float @__log10f_finite_f32(float %a) #0 {
; CHECK-LABEL: __log10f_finite_f32
; CHECK: __xl_log10f
; CHECK: blr
entry:
  %call = tail call afn float @__log10f_finite(float %a)
  ret float %call
}
define float @__powf_finite_f32(float %a, float %b) #0 {
; CHECK-LABEL: __powf_finite_f32
; CHECK: __xl_powf
; CHECK: blr
entry:
  %call = tail call afn float @__powf_finite(float %a, float %b)
  ret float %call
}
define float @__sinhf_finite_f32(float %a) #0 {
; CHECK-LABEL: __sinhf_finite_f32
; CHECK: __xl_sinhf
; CHECK: blr
entry:
  %call = tail call afn float @__sinhf_finite(float %a)
  ret float %call
}

define double @__acos_finite_f64(double %a) #0 {
; CHECK-LABEL: __acos_finite_f64
; CHECK: __xl_acos
; CHECK: blr
entry:
  %call = tail call afn double @__acos_finite(double %a)
  ret double %call
}

define double @__acosh_finite_f64(double %a) #0 {
; CHECK-LABEL: __acosh_finite_f64
; CHECK: __xl_acosh
; CHECK: blr
entry:
  %call = tail call afn double @__acosh_finite(double %a)
  ret double %call
}

define double @__asin_finite_f64(double %a) #0 {
; CHECK-LABEL: __asin_finite_f64
; CHECK: __xl_asin
; CHECK: blr
entry:
  %call = tail call afn double @__asin_finite(double %a)
  ret double %call
}

define double @__atan2_finite_f64(double %a, double %b) #0 {
; CHECK-LABEL: __atan2_finite_f64
; CHECK: __xl_atan2
; CHECK: blr
entry:
  %call = tail call afn double @__atan2_finite(double %a, double %b)
  ret double %call
}

define double @__atanh_finite_f64(double %a) #0 {
; CHECK-LABEL: __atanh_finite_f64
; CHECK: __xl_atanh
; CHECK: blr
entry:
  %call = tail call afn double @__atanh_finite(double %a)
  ret double %call
}

define double @__cosh_finite_f64(double %a) #0 {
; CHECK-LABEL: __cosh_finite_f64
; CHECK: __xl_cosh
; CHECK: blr
entry:
  %call = tail call afn double @__cosh_finite(double %a)
  ret double %call
}

define double @__exp_finite_f64(double %a) #0 {
; CHECK-LABEL: __exp_finite_f64
; CHECK: __xl_exp
; CHECK: blr
entry:
  %call = tail call afn double @__exp_finite(double %a)
  ret double %call
}

define double @__log_finite_f64(double %a) #0 {
; CHECK-LABEL: __log_finite_f64
; CHECK: __xl_log
; CHECK: blr
entry:
  %call = tail call afn double @__log_finite(double %a)
  ret double %call
}

define double @__log10_finite_f64(double %a) #0 {
; CHECK-LABEL: __log10_finite_f64
; CHECK: __xl_log10
; CHECK: blr
entry:
  %call = tail call afn double @__log10_finite(double %a)
  ret double %call
}

define double @__pow_finite_f64(double %a, double %b) #0 {
; CHECK-LABEL: __pow_finite_f64
; CHECK: __xl_pow
; CHECK: blr
entry:
  %call = tail call afn double @__pow_finite(double %a, double %b)
  ret double %call
}

define double @__sinh_finite_f64(double %a) #0 {
; CHECK-LABEL: __sinh_finite_f64
; CHECK: __xl_sinh
; CHECK: blr
entry:
  %call = tail call afn double @__sinh_finite(double %a)
  ret double %call
}

attributes #0 = { "approx-func-fp-math"="true" }
