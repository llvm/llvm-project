; RUN: opt < %s -constprop -S | FileCheck %s

; Test to verify constant folding can occur when math
; routines are mapped to the __<func>_finite versions
; of functions due to __FINITE_MATH_ONLY__ being
; enabled on headers. All calls should constant
; fold away in this test.

declare double @__acos_finite(double) #0
declare float @__acosf_finite(float) #0
declare double @__asin_finite(double) #0
declare float @__asinf_finite(float) #0
declare double @__atan2_finite(double, double) #0
declare float @__atan2f_finite(float, float) #0
declare double @__cosh_finite(double) #0
declare float @__coshf_finite(float) #0
declare double @__exp2_finite(double) #0
declare float @__exp2f_finite(float) #0
declare double @__exp_finite(double) #0
declare float @__expf_finite(float) #0
declare double @__log10_finite(double) #0
declare float @__log10f_finite(float) #0
declare double @__log_finite(double) #0
declare float @__logf_finite(float) #0
declare double @__pow_finite(double, double) #0
declare float @__powf_finite(float, float) #0
declare double @__sinh_finite(double) #0
declare float @__sinhf_finite(float) #0

attributes #0 = { nounwind readnone }

define void @T() {
; CHECK-LABEL: @T(

; CHECK-NOT: call
; CHECK: ret

  %slot = alloca double
  %slotf = alloca float
  
  %ACOS = call fast double @__acos_finite(double 1.000000e+00)
  store double %ACOS, double* %slot
  %ASIN = call fast double @__asin_finite(double 1.000000e+00)
  store double %ASIN, double* %slot
  %ATAN2 = call fast double @__atan2_finite(double 3.000000e+00, double 4.000000e+00)
  store double %ATAN2, double* %slot  
  %COSH = call fast double @__cosh_finite(double 3.000000e+00)
  store double %COSH, double* %slot
  %EXP = call fast double @__exp_finite(double 3.000000e+00)
  store double %EXP, double* %slot
  %EXP2 = call fast double @__exp2_finite(double 3.000000e+00)
  store double %EXP2, double* %slot
  %LOG = call fast double @__log_finite(double 3.000000e+00)
  store double %LOG, double* %slot
  %LOG10 = call fast double @__log10_finite(double 3.000000e+00)
  store double %LOG10, double* %slot  
  %POW = call fast double @__pow_finite(double 1.000000e+00, double 4.000000e+00)
  store double %POW, double* %slot
  %SINH = call fast double @__sinh_finite(double 3.000000e+00)
  store double %SINH, double* %slot  
  
  %ACOSF = call fast float @__acosf_finite(float 1.000000e+00)
  store float %ACOSF, float* %slotf
  %ASINF = call fast float @__asinf_finite(float 1.000000e+00)
  store float %ASINF, float* %slotf
  %ATAN2F = call fast float @__atan2f_finite(float 3.000000e+00, float 4.000000e+00)
  store float %ATAN2F, float* %slotf  
  %COSHF = call fast float @__coshf_finite(float 3.000000e+00)
  store float %COSHF, float* %slotf
  %EXPF = call fast float @__expf_finite(float 3.000000e+00)
  store float %EXPF, float* %slotf
  %EXP2F = call fast float @__exp2f_finite(float 3.000000e+00)
  store float %EXP2F, float* %slotf
  %LOGF = call fast float @__logf_finite(float 3.000000e+00)
  store float %LOGF, float* %slotf
  %LOG10F = call fast float @__log10f_finite(float 3.000000e+00)
  store float %LOG10F, float* %slotf  
  %POWF = call fast float @__powf_finite(float 3.000000e+00, float 4.000000e+00)
  store float %POWF, float* %slotf
  %SINHF = call fast float @__sinhf_finite(float 3.000000e+00)
  store float %SINHF, float* %slotf
  ret void
}
