; RUN: opt < %s -dce -S | FileCheck %s
; RUN: opt < %s -passes=dce -S | FileCheck %s

declare double @acos(double) nounwind
declare double @asin(double) nounwind
declare double @atan(double) nounwind
declare double @atan2(double, double) nounwind
declare double @ceil(double) nounwind
declare double @cos(double) nounwind
declare double @cosh(double) nounwind
declare double @exp(double) nounwind
declare double @exp2(double) nounwind
declare double @fabs(double) nounwind
declare double @floor(double) nounwind
declare double @fmod(double, double) nounwind
declare double @log(double) nounwind
declare double @log10(double) nounwind
declare double @pow(double, double) nounwind
declare double @sin(double) nounwind
declare double @sinh(double) nounwind
declare double @sqrt(double) nounwind
declare double @tan(double) nounwind
declare double @tanh(double) nounwind

declare float @acosf(float) nounwind
declare float @asinf(float) nounwind
declare float @atanf(float) nounwind
declare float @atan2f(float, float) nounwind
declare float @ceilf(float) nounwind
declare float @cosf(float) nounwind
declare float @coshf(float) nounwind
declare float @expf(float) nounwind
declare float @exp2f(float) nounwind
declare float @fabsf(float) nounwind
declare float @floorf(float) nounwind
declare float @fmodf(float, float) nounwind
declare float @logf(float) nounwind
declare float @log10f(float) nounwind
declare float @powf(float, float) nounwind
declare float @sinf(float) nounwind
declare float @sinhf(float) nounwind
declare float @sqrtf(float) nounwind
declare float @tanf(float) nounwind
declare float @tanhf(float) nounwind

define void @T() {
entry:
; CHECK-LABEL: @T(
; CHECK-NEXT: entry:

; log(0) produces a pole error
; CHECK-NEXT: %log1 = call double @log(double 0.000000e+00)
  %log1 = call double @log(double 0.000000e+00)

; log(-1) produces a domain error
; CHECK-NEXT: %log2 = call double @log(double -1.000000e+00)
  %log2 = call double @log(double -1.000000e+00)

; log(1) is 0
  %log3 = call double @log(double 1.000000e+00)

; exp(100) is roughly 2.6e+43
  %exp1 = call double @exp(double 1.000000e+02)

; exp(1000) is a range error
; CHECK-NEXT: %exp2 = call double @exp(double 1.000000e+03)
  %exp2 = call double @exp(double 1.000000e+03)

; cos(0) is 1
  %cos1 = call double @cos(double 0.000000e+00)

; cos(inf) is a domain error
; CHECK-NEXT: %cos2 = call double @cos(double 0x7FF0000000000000)
  %cos2 = call double @cos(double 0x7FF0000000000000)

; cos(0) nobuiltin may have side effects 
; CHECK-NEXT: %cos3 = call double @cos(double 0.000000e+00)
  %cos3 = call double @cos(double 0.000000e+00) nobuiltin

; pow(0, 1) is 0
  %pow1 = call double @pow(double 0x7FF0000000000000, double 1.000000e+00)

; pow(0, -1) is a pole error
; FIXME: It fails on mingw host. Suppress checking.
; %pow2 = call double @pow(double 0.000000e+00, double -1.000000e+00)

; fmod(inf, nan) is nan
  %fmod1 = call double @fmod(double 0x7FF0000000000000, double 0x7FF0000000000001)

; fmod(inf, 1) is a domain error
; CHECK-NEXT: %fmod2 = call double @fmod(double 0x7FF0000000000000, double 1.000000e+00)
  %fmod2 = call double @fmod(double 0x7FF0000000000000, double 1.000000e+00)

; CHECK-NEXT: ret void
  ret void
}

define void @Tstrict() strictfp {
entry:
; CHECK-LABEL: @Tstrict(
; CHECK-NEXT: entry:

; cos(1) strictfp sets FP status flags
; CHECK-NEXT: %cos4 = call double @cos(double 1.000000e+00)
  %cos4 = call double @cos(double 1.000000e+00) strictfp

; CHECK-NEXT: ret void
  ret void
}
