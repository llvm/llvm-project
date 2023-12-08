; RUN: opt -passes=early-cse -S < %s | FileCheck %s

; We allow either sign to provide flexibility for mathlib
; implementations. The POSIX standard is not strict here.

define float @callatan0() {
; CHECK-LABEL: @callatan0(
; CHECK-NEXT:    ret float {{-?}}0.000000e+00
;
  %call = call float @atanf(float -0.0)
  ret float %call
}

; TODO: constant should be folded
define float @callatanInf() {
; CHECK-LABEL: @callatanInf(
; CHECK-NEXT:    [[CALL:%.*]] = call float @atanf(float 0x7FF0000000000000)
; CHECK-NEXT:    ret float [[CALL]]
;
  %call = call float @atanf(float 0x7FF0000000000000)
  ret float %call
}

; TODO: constant should be folded
define float @callatanNaN() {
; CHECK-LABEL: @callatanNaN(
; CHECK-NEXT:    [[CALL:%.*]] = call float @atanf(float 0x7FF8000000000000)
; CHECK-NEXT:    ret float [[CALL]]
;
  %call = call float @atanf(float 0x7FF8000000000000)
  ret float %call
}

; POSIX: May fail with Range Error. We choose not to fail.
define float @callatanDenorm() {
; CHECK-LABEL: @callatanDenorm(
; CHECK-NEXT:    ret float 0x37A16C2000000000
;
  %call = call float @atanf(float 0x37A16C2000000000)
  ret float %call
}

; TODO: long double calls currently not folded
define x86_fp80 @atanl_x86(x86_fp80 %x) {
; CHECK-LABEL: @atanl_x86(
; CHECK-NEXT:    [[CALL:%.*]] = call x86_fp80 @atanl(x86_fp80 noundef 0xK3FFF8CCCCCCCCCCCCCCD)
; CHECK-NEXT:    ret x86_fp80 [[CALL]]
;
  %call = call x86_fp80 @atanl(x86_fp80 noundef 0xK3FFF8CCCCCCCCCCCCCCD)
  ret x86_fp80 %call
}

; This is not folded because it is known to set errno on some systems.

define float @callatan2_00() {
; CHECK-LABEL: @callatan2_00(
; CHECK-NEXT:    [[CALL:%.*]] = call float @atan2f(float 0.000000e+00, float 0.000000e+00)
; CHECK-NEXT:    ret float [[CALL]]
;
  %call = call float @atan2f(float 0.0, float 0.0)
  ret float %call
}

; This is not folded because it is known to set errno on some systems.

define float @callatan2_n00() {
; CHECK-LABEL: @callatan2_n00(
; CHECK-NEXT:    [[CALL:%.*]] = call float @atan2f(float -0.000000e+00, float 0.000000e+00)
; CHECK-NEXT:    ret float [[CALL]]
;
  %call = call float @atan2f(float -0.0, float 0.0)
  ret float %call
}

; This is not folded because it is known to set errno on some systems.

define float @callatan2_0n0() {
; CHECK-LABEL: @callatan2_0n0(
; CHECK-NEXT:    [[CALL:%.*]] = call float @atan2f(float 0.000000e+00, float -0.000000e+00)
; CHECK-NEXT:    ret float [[CALL]]
;
  %call = call float @atan2f(float 0.0, float -0.0)
  ret float %call
}

; This is not folded because it is known to set errno on some systems.

define float @callatan2_n0n0() {
; CHECK-LABEL: @callatan2_n0n0(
; CHECK-NEXT:    [[CALL:%.*]] = call float @atan2f(float -0.000000e+00, float -0.000000e+00)
; CHECK-NEXT:    ret float [[CALL]]
;
  %call = call float @atan2f(float -0.0, float -0.0)
  ret float %call
}

define float @callatan2_x0() {
; CHECK-LABEL: @callatan2_x0(
; CHECK-NEXT:    ret float 0x3FF921FB60000000
;
  %call = call float @atan2f(float 1.0, float -0.000000e+00)
  ret float %call
}

define float @callatan2_0x() {
; CHECK-LABEL: @callatan2_0x(
; CHECK-NEXT:    ret float -0.000000e+00
;
  %call = call float @atan2f(float -0.0, float 1.0)
  ret float %call
}

define float @callatan2_xx() {
; CHECK-LABEL: @callatan2_xx(
; CHECK-NEXT:    ret float 0xBFE921FB60000000
;
  %call = call float @atan2f(float -1.0, float 1.0)
  ret float %call
}

define float @callatan2_denorm() {
; CHECK-LABEL: @callatan2_denorm(
; CHECK-NEXT:    ret float 0x37A16C2000000000
;
  %call = call float @atan2f(float 0x39B4484C00000000, float 1.0e+10)
  ret float %call
}

define float @callatan2_flush_to_zero() {
; CHECK-LABEL: @callatan2_flush_to_zero(
; CHECK-NEXT:    ret float 0.000000e+00
;
  %call = call float @atan2f(float 0x39B4484C00000000, float 0x4415AF1D80000000)
  ret float %call
}

define float @callatan2_NaN() {
; CHECK-LABEL: @callatan2_NaN(
; CHECK-NEXT:    ret float 0x7FF8000000000000
;
  %call = call float @atan2f(float 0x7FF8000000000000, float 0x7FF8000000000000)
  ret float %call
}

define float @callatan2_Inf() {
; CHECK-LABEL: @callatan2_Inf(
; CHECK-NEXT:    ret float 0x3FE921FB60000000
;
  %call = call float @atan2f(float 0x7FF0000000000000, float 0x7FF0000000000000)
  ret float %call
}

declare dso_local float @atanf(float) #0
declare dso_local x86_fp80 @atanl(x86_fp80) #0

declare dso_local float @atan2f(float, float) #0

attributes #0 = { nofree nounwind willreturn }
