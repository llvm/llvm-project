; Exercises the scalar->AOCL fast-call name mapping under fast-math at -O3 with
; -fast-library=AMDLIBM on X86: float variants, *_finite aliases and
; inverse-trig functions are rewritten, while math calls with no AOCL mapping
; (e.g. cbrt) are left untouched.

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O3 -fast-library=AMDLIBM < %s \
; RUN:   | FileCheck %s --check-prefix=AMD
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O3 < %s \
; RUN:   | FileCheck %s --check-prefix=STD

declare float @tanf(float)
declare float @powf(float, float)
declare double @acos(double)
declare float @acosf(float)
declare double @atan(double)
declare double @cos(double)
declare float @sinf(float)
declare double @erf(double)
declare double @__exp_finite(double)
declare double @pow(double, double)
declare double @cbrt(double)

; Single-precision variant: tanf -> amd_fasttanf
define float @call_tanf(float %x) #0 {
  %r = call float @tanf(float %x)
  %a = fadd float %r, %x
  ret float %a
}
; AMD-LABEL: call_tanf:
; AMD: callq{{.*}}amd_fasttanf
; STD-LABEL: call_tanf:
; STD: callq{{.*}}tanf

; Single-precision, two-argument variant: powf -> amd_fastpowf
define float @call_powf(float %x, float %y) #0 {
  %r = call float @powf(float %x, float %y)
  %a = fadd float %r, %x
  ret float %a
}
; AMD-LABEL: call_powf:
; AMD: callq{{.*}}amd_fastpowf

; Inverse-trigonometric functions.
define double @call_acos(double %x) #0 {
  %r = call double @acos(double %x)
  %a = fadd double %r, %x
  ret double %a
}
; AMD-LABEL: call_acos:
; AMD: callq{{.*}}amd_fastacos

define double @call_atan(double %x) #0 {
  %r = call double @atan(double %x)
  %a = fadd double %r, %x
  ret double %a
}
; AMD-LABEL: call_atan:
; AMD: callq{{.*}}amd_fastatan

; Single-precision inverse-trig: acosf -> amd_fastacosf
define float @call_acosf(float %x) #0 {
  %r = call float @acosf(float %x)
  %a = fadd float %r, %x
  ret float %a
}
; AMD-LABEL: call_acosf:
; AMD: callq{{.*}}amd_fastacosf

; cos(double) -> amd_fastcos
define double @call_cos(double %x) #0 {
  %r = call double @cos(double %x)
  %a = fadd double %r, %x
  ret double %a
}
; AMD-LABEL: call_cos:
; AMD: callq{{.*}}amd_fastcos

; sinf -> amd_fastsinf
define float @call_sinf(float %x) #0 {
  %r = call float @sinf(float %x)
  %a = fadd float %r, %x
  ret float %a
}
; AMD-LABEL: call_sinf:
; AMD: callq{{.*}}amd_fastsinf

; erf(double) -> amd_fasterf
define double @call_erf(double %x) #0 {
  %r = call double @erf(double %x)
  %a = fadd double %r, %x
  ret double %a
}
; AMD-LABEL: call_erf:
; AMD: callq{{.*}}amd_fasterf

; pow(double) -> amd_fastpow
define double @call_pow(double %x, double %y) #0 {
  %r = call double @pow(double %x, double %y)
  %a = fadd double %r, %x
  ret double %a
}
; AMD-LABEL: call_pow:
; AMD: callq{{.*}}amd_fastpow

; A *_finite alias maps to the same fast entry as the base function.
define double @call_exp_finite(double %x) #0 {
  %r = call double @__exp_finite(double %x)
  %a = fadd double %r, %x
  ret double %a
}
; AMD-LABEL: call_exp_finite:
; AMD: callq{{.*}}amd_fastexp

; cbrt has no AOCL mapping and must not be rewritten.
define double @call_cbrt_unmapped(double %x) #0 {
  %r = call double @cbrt(double %x)
  %a = fadd double %r, %x
  ret double %a
}
; AMD-LABEL: call_cbrt_unmapped:
; AMD-NOT: amd_fast
; AMD: callq{{.*}}cbrt

attributes #0 = { "approx-func-fp-math"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" }
