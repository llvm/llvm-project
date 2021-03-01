; RUN: llc -verify-machineinstrs -mcpu=pwr9  < %s -mtriple=powerpc64le-unknown-linux-gnu | FileCheck -check-prefixes=CHECK-PWR9,CHECK-ALL %s 
; RUN: llc -verify-machineinstrs -mcpu=pwr8  < %s -mtriple=powerpc64le-unknown-linux-gnu | FileCheck -check-prefixes=CHECK-PWR8,CHECK-ALL %s 
; RUN: llc -verify-machineinstrs -mcpu=pwr8  < %s -mtriple=powerpc64le-unknown-linux-gnu | FileCheck --check-prefix=CHECK-ALL %s 

declare <2 x double> @__cbrtd2_massv(<2 x double>)
declare <4 x float> @__cbrtf4_massv(<4 x float>)

declare <2 x double> @__powd2_massv(<2 x double>, <2 x double>)
declare <4 x float> @__powf4_massv(<4 x float>, <4 x float>)

declare <2 x double> @__expd2_massv(<2 x double>)
declare <4 x float> @__expf4_massv(<4 x float>)

declare <2 x double> @__exp2d2_massv(<2 x double>)
declare <4 x float> @__exp2f4_massv(<4 x float>)

declare <2 x double> @__expm1d2_massv(<2 x double>)
declare <4 x float> @__expm1f4_massv(<4 x float>)

declare <2 x double> @__logd2_massv(<2 x double>)
declare <4 x float> @__logf4_massv(<4 x float>)

declare <2 x double> @__log1pd2_massv(<2 x double>)
declare <4 x float> @__log1pf4_massv(<4 x float>)

declare <2 x double> @__log10d2_massv(<2 x double>)
declare <4 x float> @__log10f4_massv(<4 x float>)

declare <2 x double> @__log2d2_massv(<2 x double>)
declare <4 x float> @__log2f4_massv(<4 x float>)

declare <2 x double> @__sind2_massv(<2 x double>)
declare <4 x float> @__sinf4_massv(<4 x float>)

declare <2 x double> @__cosd2_massv(<2 x double>)
declare <4 x float> @__cosf4_massv(<4 x float>)

declare <2 x double> @__tand2_massv(<2 x double>)
declare <4 x float> @__tanf4_massv(<4 x float>)

declare <2 x double> @__asind2_massv(<2 x double>)
declare <4 x float> @__asinf4_massv(<4 x float>)

declare <2 x double> @__acosd2_massv(<2 x double>)
declare <4 x float> @__acosf4_massv(<4 x float>)

declare <2 x double> @__atand2_massv(<2 x double>)
declare <4 x float> @__atanf4_massv(<4 x float>)

declare <2 x double> @__atan2d2_massv(<2 x double>)
declare <4 x float> @__atan2f4_massv(<4 x float>)

declare <2 x double> @__sinhd2_massv(<2 x double>)
declare <4 x float> @__sinhf4_massv(<4 x float>)

declare <2 x double> @__coshd2_massv(<2 x double>)
declare <4 x float> @__coshf4_massv(<4 x float>)

declare <2 x double> @__tanhd2_massv(<2 x double>)
declare <4 x float> @__tanhf4_massv(<4 x float>)

declare <2 x double> @__asinhd2_massv(<2 x double>)
declare <4 x float> @__asinhf4_massv(<4 x float>)

declare <2 x double> @__acoshd2_massv(<2 x double>)
declare <4 x float> @__acoshf4_massv(<4 x float>)

declare <2 x double> @__atanhd2_massv(<2 x double>)
declare <4 x float> @__atanhf4_massv(<4 x float>)

; following tests check generation of subtarget-specific calls
; cbrt
define <2 x double>  @cbrt_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @cbrt_f64_massv
; CHECK-PWR9: bl __cbrtd2_P9
; CHECK-PWR8: bl __cbrtd2_P8
; CHECK-NOT: bl __cbrtd2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__cbrtd2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @cbrt_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @cbrt_f32_massv
; CHECK-PWR9: bl __cbrtf4_P9
; CHECK-PWR8: bl __cbrtf4_P8
; CHECK-NOT: bl __cbrtf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__cbrtf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; pow
define <2 x double>  @pow_f64_massv(<2 x double> %opnd1, <2 x double> %opnd2) {
; CHECK-ALL-LABEL: @pow_f64_massv
; CHECK-PWR9: bl __powd2_P9
; CHECK-PWR8: bl __powd2_P8
; CHECK-NOT: bl __powd2_massv
; CHECK-ALL: blr
;
 %1 = call <2 x double> @__powd2_massv(<2 x double> %opnd1, <2 x double> %opnd2)
  ret <2 x double> %1 
}

define <4 x float>  @pow_f32_massv(<4 x float> %opnd1, <4 x float> %opnd2) {
; CHECK-ALL-LABEL: @pow_f32_massv
; CHECK-PWR9: bl __powf4_P9
; CHECK-PWR8: bl __powf4_P8
; CHECK-NOT: bl __powf4_massv
; CHECK-ALL: blr
;
 %1 = call <4 x float> @__powf4_massv(<4 x float> %opnd1, <4 x float> %opnd2)
  ret <4 x float> %1 
}

; exp
define <2 x double>  @exp_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @exp_f64_massv
; CHECK-PWR9: bl __expd2_P9
; CHECK-PWR8: bl __expd2_P8
; CHECK-NOT: bl __expd2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__expd2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @exp_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @exp_f32_massv
; CHECK-PWR9: bl __expf4_P9
; CHECK-PWR8: bl __expf4_P8
; CHECK-NOT: bl __expf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__expf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; exp2
define <2 x double>  @exp2_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @exp2_f64_massv
; CHECK-PWR9: bl __exp2d2_P9
; CHECK-PWR8: bl __exp2d2_P8
; CHECK-NOT: bl __exp2d2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__exp2d2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @exp2_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @exp2_f32_massv
; CHECK-PWR9: bl __exp2f4_P9
; CHECK-PWR8: bl __exp2f4_P8
; CHECK-NOT: bl __exp2f4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__exp2f4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; expm1
define <2 x double>  @expm1_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @expm1_f64_massv
; CHECK-PWR9: bl __expm1d2_P9
; CHECK-PWR8: bl __expm1d2_P8
; CHECK-NOT: bl __expm1d2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__expm1d2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @expm1_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @expm1_f32_massv
; CHECK-PWR9: bl __expm1f4_P9
; CHECK-PWR8: bl __expm1f4_P8
; CHECK-NOT: bl __expm1f4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__expm1f4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; log
define <2 x double>  @log_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @log_f64_massv
; CHECK-PWR9: bl __logd2_P9
; CHECK-PWR8: bl __logd2_P8
; CHECK-NOT: bl __logd2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__logd2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @log_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @log_f32_massv
; CHECK-PWR9: bl __logf4_P9
; CHECK-PWR8: bl __logf4_P8
; CHECK-NOT: bl __logf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__logf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; log1p
define <2 x double>  @log1p_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @log1p_f64_massv
; CHECK-PWR9: bl __log1pd2_P9
; CHECK-PWR8: bl __log1pd2_P8
; CHECK-NOT: bl __log1pd2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__log1pd2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @log1p_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @log1p_f32_massv
; CHECK-PWR9: bl __log1pf4_P9
; CHECK-PWR8: bl __log1pf4_P8
; CHECK-NOT: bl __log1pf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__log1pf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; log10
define <2 x double>  @log10_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @log10_f64_massv
; CHECK-PWR9: bl __log10d2_P9
; CHECK-PWR8: bl __log10d2_P8
; CHECK-NOT: bl __log10d2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__log10d2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @log10_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @log10_f32_massv
; CHECK-PWR9: bl __log10f4_P9
; CHECK-PWR8: bl __log10f4_P8
; CHECK-NOT: bl __log10f4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__log10f4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; log2
define <2 x double>  @log2_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @log2_f64_massv
; CHECK-PWR9: bl __log2d2_P9
; CHECK-PWR8: bl __log2d2_P8
; CHECK-NOT: bl __log2d2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__log2d2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @log2_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @log2_f32_massv
; CHECK-PWR9: bl __log2f4_P9
; CHECK-PWR8: bl __log2f4_P8
; CHECK-NOT: bl __log2f4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__log2f4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; sin
define <2 x double>  @sin_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @sin_f64_massv
; CHECK-PWR9: bl __sind2_P9
; CHECK-PWR8: bl __sind2_P8
; CHECK-NOT: bl __sind2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__sind2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @sin_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @sin_f32_massv
; CHECK-PWR9: bl __sinf4_P9
; CHECK-PWR8: bl __sinf4_P8
; CHECK-NOT: bl __sinf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__sinf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; cos
define <2 x double>  @cos_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @cos_f64_massv
; CHECK-PWR9: bl __cosd2_P9
; CHECK-PWR8: bl __cosd2_P8
; CHECK-NOT: bl __cosd2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__cosd2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @cos_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @cos_f32_massv
; CHECK-PWR9: bl __cosf4_P9
; CHECK-PWR8: bl __cosf4_P8
; CHECK-NOT: bl __cosf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__cosf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; tan
define <2 x double>  @tan_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @tan_f64_massv
; CHECK-PWR9: bl __tand2_P9
; CHECK-PWR8: bl __tand2_P8
; CHECK-NOT: bl __tand2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__tand2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @tan_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @tan_f32_massv
; CHECK-PWR9: bl __tanf4_P9
; CHECK-PWR8: bl __tanf4_P8
; CHECK-NOT: bl __tanf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__tanf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; asin
define <2 x double>  @asin_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @asin_f64_massv
; CHECK-PWR9: bl __asind2_P9
; CHECK-PWR8: bl __asind2_P8
; CHECK-NOT: bl __asind2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__asind2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @asin_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @asin_f32_massv
; CHECK-PWR9: bl __asinf4_P9
; CHECK-PWR8: bl __asinf4_P8
; CHECK-NOT: bl __asinf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__asinf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; acos
define <2 x double>  @acos_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @acos_f64_massv
; CHECK-PWR9: bl __acosd2_P9
; CHECK-PWR8: bl __acosd2_P8
; CHECK-NOT: bl __acosd2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__acosd2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @acos_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @acos_f32_massv
; CHECK-PWR9: bl __acosf4_P9
; CHECK-PWR8: bl __acosf4_P8
; CHECK-NOT: bl __acosf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__acosf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; atan
define <2 x double>  @atan_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @atan_f64_massv
; CHECK-PWR9: bl __atand2_P9
; CHECK-PWR8: bl __atand2_P8
; CHECK-NOT: bl __atand2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__atand2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @atan_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @atan_f32_massv
; CHECK-PWR9: bl __atanf4_P9
; CHECK-PWR8: bl __atanf4_P8
; CHECK-NOT: bl __atanf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__atanf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; atan2
define <2 x double>  @atan2_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @atan2_f64_massv
; CHECK-PWR9: bl __atan2d2_P9
; CHECK-PWR8: bl __atan2d2_P8
; CHECK-NOT: bl __atan2d2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__atan2d2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @atan2_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @atan2_f32_massv
; CHECK-PWR9: bl __atan2f4_P9
; CHECK-PWR8: bl __atan2f4_P8
; CHECK-NOT: bl __atan2f4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__atan2f4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; sinh
define <2 x double>  @sinh_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @sinh_f64_massv
; CHECK-PWR9: bl __sinhd2_P9
; CHECK-PWR8: bl __sinhd2_P8
; CHECK-NOT: bl __sinhd2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__sinhd2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @sinh_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @sinh_f32_massv
; CHECK-PWR9: bl __sinhf4_P9
; CHECK-PWR8: bl __sinhf4_P8
; CHECK-NOT: bl __sinhf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__sinhf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; cosh
define <2 x double>  @cosh_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @cosh_f64_massv
; CHECK-PWR9: bl __coshd2_P9
; CHECK-PWR8: bl __coshd2_P8
; CHECK-NOT: bl __coshd2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__coshd2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @cosh_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @cosh_f32_massv
; CHECK-PWR9: bl __coshf4_P9
; CHECK-PWR8: bl __coshf4_P8
; CHECK-NOT: bl __coshf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__coshf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; tanh
define <2 x double>  @tanh_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @tanh_f64_massv
; CHECK-PWR9: bl __tanhd2_P9
; CHECK-PWR8: bl __tanhd2_P8
; CHECK-NOT: bl __tanhd2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__tanhd2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @tanh_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @tanh_f32_massv
; CHECK-PWR9: bl __tanhf4_P9
; CHECK-PWR8: bl __tanhf4_P8
; CHECK-NOT: bl __tanhf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__tanhf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; asinh
define <2 x double>  @asinh_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @asinh_f64_massv
; CHECK-PWR9: bl __asinhd2_P9
; CHECK-PWR8: bl __asinhd2_P8
; CHECK-NOT: bl __asinhd2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__asinhd2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @asinh_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @asinh_f32_massv
; CHECK-PWR9: bl __asinhf4_P9
; CHECK-PWR8: bl __asinhf4_P8
; CHECK-NOT: bl __asinhf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__asinhf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; acosh
define <2 x double>  @acosh_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @acosh_f64_massv
; CHECK-PWR9: bl __acoshd2_P9
; CHECK-PWR8: bl __acoshd2_P8
; CHECK-NOT: bl __acoshd2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__acoshd2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @acosh_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @acosh_f32_massv
; CHECK-PWR9: bl __acoshf4_P9
; CHECK-PWR8: bl __acoshf4_P8
; CHECK-NOT: bl __acoshf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__acoshf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

; atanh
define <2 x double>  @atanh_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @atanh_f64_massv
; CHECK-PWR9: bl __atanhd2_P9
; CHECK-PWR8: bl __atanhd2_P8
; CHECK-NOT: bl __atanhd2_massv
; CHECK-ALL: blr
;
  %1 = call <2 x double> @__atanhd2_massv(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @atanh_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @atanh_f32_massv
; CHECK-PWR9: bl __atanhf4_P9
; CHECK-PWR8: bl __atanhf4_P8
; CHECK-NOT: bl __atanhf4_massv
; CHECK-ALL: blr
;
  %1 = call <4 x float> @__atanhf4_massv(<4 x float> %opnd)
  ret <4 x float> %1 
}

