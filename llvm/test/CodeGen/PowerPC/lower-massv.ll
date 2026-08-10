; RUN: llc -verify-machineinstrs -mcpu=pwr10  < %s -mtriple=powerpc64le-unknown-linux-gnu | FileCheck -check-prefixes=CHECK-PWR9,CHECK-ALL %s 
; RUN: llc -verify-machineinstrs -mcpu=pwr9   < %s -mtriple=powerpc64le-unknown-linux-gnu | FileCheck -check-prefixes=CHECK-PWR9,CHECK-ALL %s 
; RUN: llc -verify-machineinstrs -mcpu=pwr8   < %s -mtriple=powerpc64le-unknown-linux-gnu | FileCheck -check-prefixes=CHECK-PWR8,CHECK-ALL %s 
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu | FileCheck -check-prefixes=CHECK-PWR8,CHECK-ALL %s 
; RUN: llc -verify-machineinstrs -mcpu=pwr10  < %s -mtriple=powerpc-ibm-aix-xcoff | FileCheck -check-prefixes=CHECK-PWR10,CHECK-ALL %s 
; RUN: llc -verify-machineinstrs -mcpu=pwr9   < %s -mtriple=powerpc-ibm-aix-xcoff | FileCheck -check-prefixes=CHECK-PWR9,CHECK-ALL %s 
; RUN: llc -verify-machineinstrs -mcpu=pwr8   < %s -mtriple=powerpc-ibm-aix-xcoff | FileCheck -check-prefixes=CHECK-PWR8,CHECK-ALL %s 
; RUN: llc -verify-machineinstrs -mcpu=pwr7   < %s -mtriple=powerpc-ibm-aix-xcoff | FileCheck -check-prefixes=CHECK-PWR7,CHECK-ALL %s 
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-ibm-aix-xcoff | FileCheck -check-prefixes=CHECK-PWR7,CHECK-ALL %s 

declare <2 x double> @__cbrtd2(<2 x double>)
declare <4 x float> @__cbrtf4(<4 x float>)

declare <2 x double> @__powd2(<2 x double>, <2 x double>)
declare <4 x float> @__powf4(<4 x float>, <4 x float>)

declare <2 x double> @__expd2(<2 x double>)
declare <4 x float> @__expf4(<4 x float>)

declare <2 x double> @__exp2d2(<2 x double>)
declare <4 x float> @__exp2f4(<4 x float>)

declare <2 x double> @__expm1d2(<2 x double>)
declare <4 x float> @__expm1f4(<4 x float>)

declare <2 x double> @__logd2(<2 x double>)
declare <4 x float> @__logf4(<4 x float>)

declare <2 x double> @__log1pd2(<2 x double>)
declare <4 x float> @__log1pf4(<4 x float>)

declare <2 x double> @__log10d2(<2 x double>)
declare <4 x float> @__log10f4(<4 x float>)

declare <2 x double> @__log2d2(<2 x double>)
declare <4 x float> @__log2f4(<4 x float>)

declare <2 x double> @__sind2(<2 x double>)
declare <4 x float> @__sinf4(<4 x float>)

declare <2 x double> @__cosd2(<2 x double>)
declare <4 x float> @__cosf4(<4 x float>)

declare <2 x double> @__tand2(<2 x double>)
declare <4 x float> @__tanf4(<4 x float>)

declare <2 x double> @__asind2(<2 x double>)
declare <4 x float> @__asinf4(<4 x float>)

declare <2 x double> @__acosd2(<2 x double>)
declare <4 x float> @__acosf4(<4 x float>)

declare <2 x double> @__atand2(<2 x double>)
declare <4 x float> @__atanf4(<4 x float>)

declare <2 x double> @__atan2d2(<2 x double>)
declare <4 x float> @__atan2f4(<4 x float>)

declare <2 x double> @__sinhd2(<2 x double>)
declare <4 x float> @__sinhf4(<4 x float>)

declare <2 x double> @__coshd2(<2 x double>)
declare <4 x float> @__coshf4(<4 x float>)

declare <2 x double> @__tanhd2(<2 x double>)
declare <4 x float> @__tanhf4(<4 x float>)

declare <2 x double> @__asinhd2(<2 x double>)
declare <4 x float> @__asinhf4(<4 x float>)

declare <2 x double> @__acoshd2(<2 x double>)
declare <4 x float> @__acoshf4(<4 x float>)

declare <2 x double> @__atanhd2(<2 x double>)
declare <4 x float> @__atanhf4(<4 x float>)

; following tests check generation of subtarget-specific calls
; cbrt
define <2 x double>  @cbrt_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @cbrt_f64_massv
; CHECK-PWR10: __cbrtd2_P10
; CHECK-PWR9:  __cbrtd2_P9
; CHECK-PWR8:  __cbrtd2_P8
; CHECK-PWR7:  __cbrtd2_P7
; CHECK-NOT:   __cbrtd2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__cbrtd2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @cbrt_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @cbrt_f32_massv
; CHECK-PWR10: __cbrtf4_P10
; CHECK-PWR9:  __cbrtf4_P9
; CHECK-PWR8:  __cbrtf4_P8
; CHECK-PWR7:  __cbrtf4_P7
; CHECK-NOT:   __cbrtf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__cbrtf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; pow
define <2 x double>  @pow_f64_massv(<2 x double> %opnd1, <2 x double> %opnd2) {
; CHECK-ALL-LABEL: @pow_f64_massv
; CHECK-PWR10: __powd2_P10
; CHECK-PWR9:  __powd2_P9
; CHECK-PWR8:  __powd2_P8
; CHECK-PWR7:  __powd2_P7
; CHECK-NOT:   __powd2_massv
; CHECK-ALL:   blr
;
 %1 = call <2 x double> @__powd2(<2 x double> %opnd1, <2 x double> %opnd2)
  ret <2 x double> %1 
}

define <4 x float>  @pow_f32_massv(<4 x float> %opnd1, <4 x float> %opnd2) {
; CHECK-ALL-LABEL: @pow_f32_massv
; CHECK-PWR10: __powf4_P10
; CHECK-PWR9:  __powf4_P9
; CHECK-PWR8:  __powf4_P8
; CHECK-PWR7:  __powf4_P7
; CHECK-NOT:   __powf4_massv
; CHECK-ALL:   blr
;
 %1 = call <4 x float> @__powf4(<4 x float> %opnd1, <4 x float> %opnd2)
  ret <4 x float> %1 
}

; exp
define <2 x double>  @exp_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @exp_f64_massv
; CHECK-PWR10: __expd2_P10
; CHECK-PWR9:  __expd2_P9
; CHECK-PWR8:  __expd2_P8
; CHECK-PWR7:  __expd2_P7
; CHECK-NOT:   __expd2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__expd2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @exp_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @exp_f32_massv
; CHECK-PWR10: __expf4_P10
; CHECK-PWR9:  __expf4_P9
; CHECK-PWR8:  __expf4_P8
; CHECK-PWR7:  __expf4_P7
; CHECK-NOT:   __expf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__expf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; exp2
define <2 x double>  @exp2_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @exp2_f64_massv
; CHECK-PWR10: __exp2d2_P10
; CHECK-PWR9:  __exp2d2_P9
; CHECK-PWR8:  __exp2d2_P8
; CHECK-PWR7:  __exp2d2_P7
; CHECK-NOT:   __exp2d2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__exp2d2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @exp2_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @exp2_f32_massv
; CHECK-PWR10: __exp2f4_P10
; CHECK-PWR9:  __exp2f4_P9
; CHECK-PWR8:  __exp2f4_P8
; CHECK-PWR7:  __exp2f4_P7
; CHECK-NOT:   __exp2f4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__exp2f4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; expm1
define <2 x double>  @expm1_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @expm1_f64_massv
; CHECK-PWR10: __expm1d2_P10
; CHECK-PWR9:  __expm1d2_P9
; CHECK-PWR8:  __expm1d2_P8
; CHECK-PWR7:  __expm1d2_P7
; CHECK-NOT:   __expm1d2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__expm1d2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @expm1_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @expm1_f32_massv
; CHECK-PWR10: __expm1f4_P10
; CHECK-PWR9:  __expm1f4_P9
; CHECK-PWR8:  __expm1f4_P8
; CHECK-PWR7:  __expm1f4_P7
; CHECK-NOT:   __expm1f4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__expm1f4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; log
define <2 x double>  @log_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @log_f64_massv
; CHECK-PWR10: __logd2_P10
; CHECK-PWR9:  __logd2_P9
; CHECK-PWR8:  __logd2_P8
; CHECK-PWR7:  __logd2_P7
; CHECK-NOT:   __logd2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__logd2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @log_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @log_f32_massv
; CHECK-PWR10: __logf4_P10
; CHECK-PWR9:  __logf4_P9
; CHECK-PWR8:  __logf4_P8
; CHECK-PWR7:  __logf4_P7
; CHECK-NOT:   __logf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__logf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; log1p
define <2 x double>  @log1p_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @log1p_f64_massv
; CHECK-PWR10: __log1pd2_P10
; CHECK-PWR9:  __log1pd2_P9
; CHECK-PWR8:  __log1pd2_P8
; CHECK-PWR7:  __log1pd2_P7
; CHECK-NOT:   __log1pd2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__log1pd2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @log1p_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @log1p_f32_massv
; CHECK-PWR10: __log1pf4_P10
; CHECK-PWR9:  __log1pf4_P9
; CHECK-PWR8:  __log1pf4_P8
; CHECK-PWR7:  __log1pf4_P7
; CHECK-NOT:   __log1pf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__log1pf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; log10
define <2 x double>  @log10_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @log10_f64_massv
; CHECK-PWR10: __log10d2_P10
; CHECK-PWR9:  __log10d2_P9
; CHECK-PWR8:  __log10d2_P8
; CHECK-PWR7:  __log10d2_P7
; CHECK-NOT:   __log10d2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__log10d2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @log10_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @log10_f32_massv
; CHECK-PWR10: __log10f4_P10
; CHECK-PWR9:  __log10f4_P9
; CHECK-PWR8:  __log10f4_P8
; CHECK-PWR7:  __log10f4_P7
; CHECK-NOT:   __log10f4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__log10f4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; log2
define <2 x double>  @log2_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @log2_f64_massv
; CHECK-PWR10: __log2d2_P10
; CHECK-PWR9:  __log2d2_P9
; CHECK-PWR8:  __log2d2_P8
; CHECK-PWR7:  __log2d2_P7
; CHECK-NOT:   __log2d2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__log2d2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @log2_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @log2_f32_massv
; CHECK-PWR10: __log2f4_P10
; CHECK-PWR9:  __log2f4_P9
; CHECK-PWR8:  __log2f4_P8
; CHECK-PWR7:  __log2f4_P7
; CHECK-NOT:   __log2f4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__log2f4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; sin
define <2 x double>  @sin_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @sin_f64_massv
; CHECK-PWR10: __sind2_P10
; CHECK-PWR9:  __sind2_P9
; CHECK-PWR8:  __sind2_P8
; CHECK-PWR7:  __sind2_P7
; CHECK-NOT:   __sind2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__sind2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @sin_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @sin_f32_massv
; CHECK-PWR10: __sinf4_P10
; CHECK-PWR9:  __sinf4_P9
; CHECK-PWR8:  __sinf4_P8
; CHECK-PWR7:  __sinf4_P7
; CHECK-NOT:   __sinf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__sinf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; cos
define <2 x double>  @cos_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @cos_f64_massv
; CHECK-PWR10: __cosd2_P10
; CHECK-PWR9:  __cosd2_P9
; CHECK-PWR8:  __cosd2_P8
; CHECK-PWR7:  __cosd2_P7
; CHECK-NOT:   __cosd2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__cosd2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @cos_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @cos_f32_massv
; CHECK-PWR10: __cosf4_P10
; CHECK-PWR9:  __cosf4_P9
; CHECK-PWR8:  __cosf4_P8
; CHECK-PWR7:  __cosf4_P7
; CHECK-NOT:   __cosf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__cosf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; tan
define <2 x double>  @tan_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @tan_f64_massv
; CHECK-PWR10: __tand2_P10
; CHECK-PWR9:  __tand2_P9
; CHECK-PWR8:  __tand2_P8
; CHECK-PWR7:  __tand2_P7
; CHECK-NOT:   __tand2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__tand2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @tan_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @tan_f32_massv
; CHECK-PWR10: __tanf4_P10
; CHECK-PWR9:  __tanf4_P9
; CHECK-PWR8:  __tanf4_P8
; CHECK-PWR7:  __tanf4_P7
; CHECK-NOT:   __tanf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__tanf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; asin
define <2 x double>  @asin_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @asin_f64_massv
; CHECK-PWR10: __asind2_P10
; CHECK-PWR9:  __asind2_P9
; CHECK-PWR8:  __asind2_P8
; CHECK-PWR7:  __asind2_P7
; CHECK-NOT:   __asind2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__asind2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @asin_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @asin_f32_massv
; CHECK-PWR10: __asinf4_P10
; CHECK-PWR9:  __asinf4_P9
; CHECK-PWR8:  __asinf4_P8
; CHECK-PWR7:  __asinf4_P7
; CHECK-NOT:   __asinf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__asinf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; acos
define <2 x double>  @acos_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @acos_f64_massv
; CHECK-PWR10: __acosd2_P10
; CHECK-PWR9:  __acosd2_P9
; CHECK-PWR8:  __acosd2_P8
; CHECK-PWR7:  __acosd2_P7
; CHECK-NOT:   __acosd2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__acosd2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @acos_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @acos_f32_massv
; CHECK-PWR10: __acosf4_P10
; CHECK-PWR9:  __acosf4_P9
; CHECK-PWR8:  __acosf4_P8
; CHECK-PWR7:  __acosf4_P7
; CHECK-NOT:   __acosf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__acosf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; atan
define <2 x double>  @atan_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @atan_f64_massv
; CHECK-PWR10: __atand2_P10
; CHECK-PWR9:  __atand2_P9
; CHECK-PWR8:  __atand2_P8
; CHECK-PWR7:  __atand2_P7
; CHECK-NOT:   __atand2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__atand2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @atan_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @atan_f32_massv
; CHECK-PWR10: __atanf4_P10
; CHECK-PWR9:  __atanf4_P9
; CHECK-PWR8:  __atanf4_P8
; CHECK-PWR7:  __atanf4_P7
; CHECK-NOT:   __atanf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__atanf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; atan2
define <2 x double>  @atan2_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @atan2_f64_massv
; CHECK-PWR10: __atan2d2_P10
; CHECK-PWR9:  __atan2d2_P9
; CHECK-PWR8:  __atan2d2_P8
; CHECK-PWR7:  __atan2d2_P7
; CHECK-NOT:   __atan2d2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__atan2d2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @atan2_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @atan2_f32_massv
; CHECK-PWR10: __atan2f4_P10
; CHECK-PWR9:  __atan2f4_P9
; CHECK-PWR8:  __atan2f4_P8
; CHECK-PWR7:  __atan2f4_P7
; CHECK-NOT:   __atan2f4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__atan2f4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; sinh
define <2 x double>  @sinh_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @sinh_f64_massv
; CHECK-PWR10: __sinhd2_P10
; CHECK-PWR9:  __sinhd2_P9
; CHECK-PWR8:  __sinhd2_P8
; CHECK-PWR7:  __sinhd2_P7
; CHECK-NOT:   __sinhd2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__sinhd2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @sinh_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @sinh_f32_massv
; CHECK-PWR10: __sinhf4_P10
; CHECK-PWR9:  __sinhf4_P9
; CHECK-PWR8:  __sinhf4_P8
; CHECK-PWR7:  __sinhf4_P7
; CHECK-NOT:   __sinhf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__sinhf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; cosh
define <2 x double>  @cosh_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @cosh_f64_massv
; CHECK-PWR10: __coshd2_P10
; CHECK-PWR9:  __coshd2_P9
; CHECK-PWR8:  __coshd2_P8
; CHECK-PWR7:  __coshd2_P7
; CHECK-NOT:   __coshd2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__coshd2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @cosh_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @cosh_f32_massv
; CHECK-PWR10: __coshf4_P10
; CHECK-PWR9:  __coshf4_P9
; CHECK-PWR8:  __coshf4_P8
; CHECK-PWR7:  __coshf4_P7
; CHECK-NOT:   __coshf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__coshf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; tanh
define <2 x double>  @tanh_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @tanh_f64_massv
; CHECK-PWR10: __tanhd2_P10
; CHECK-PWR9:  __tanhd2_P9
; CHECK-PWR8:  __tanhd2_P8
; CHECK-PWR7:  __tanhd2_P7
; CHECK-NOT:   __tanhd2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__tanhd2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @tanh_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @tanh_f32_massv
; CHECK-PWR10: __tanhf4_P10
; CHECK-PWR9:  __tanhf4_P9
; CHECK-PWR8:  __tanhf4_P8
; CHECK-PWR7:  __tanhf4_P7
; CHECK-NOT:   __tanhf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__tanhf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; asinh
define <2 x double>  @asinh_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @asinh_f64_massv
; CHECK-PWR10: __asinhd2_P10
; CHECK-PWR9:  __asinhd2_P9
; CHECK-PWR8:  __asinhd2_P8
; CHECK-PWR7:  __asinhd2_P7
; CHECK-NOT:   __asinhd2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__asinhd2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @asinh_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @asinh_f32_massv
; CHECK-PWR10: __asinhf4_P10
; CHECK-PWR9:  __asinhf4_P9
; CHECK-PWR8:  __asinhf4_P8
; CHECK-PWR7:  __asinhf4_P7
; CHECK-NOT:   __asinhf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__asinhf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; acosh
define <2 x double>  @acosh_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @acosh_f64_massv
; CHECK-PWR10: __acoshd2_P10
; CHECK-PWR9:  __acoshd2_P9
; CHECK-PWR8:  __acoshd2_P8
; CHECK-PWR7:  __acoshd2_P7
; CHECK-NOT:   __acoshd2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__acoshd2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @acosh_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @acosh_f32_massv
; CHECK-PWR10: __acoshf4_P10
; CHECK-PWR9:  __acoshf4_P9
; CHECK-PWR8:  __acoshf4_P8
; CHECK-PWR7:  __acoshf4_P7
; CHECK-NOT:   __acoshf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__acoshf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

; atanh
define <2 x double>  @atanh_f64_massv(<2 x double> %opnd) {
; CHECK-ALL-LABEL: @atanh_f64_massv
; CHECK-PWR10: __atanhd2_P10
; CHECK-PWR9:  __atanhd2_P9
; CHECK-PWR8:  __atanhd2_P8
; CHECK-PWR7:  __atanhd2_P7
; CHECK-NOT:   __atanhd2_massv
; CHECK-ALL:   blr
;
  %1 = call <2 x double> @__atanhd2(<2 x double> %opnd)
  ret <2 x double> %1 
}

define <4 x float>  @atanh_f32_massv(<4 x float> %opnd) {
; CHECK-ALL-LABEL: @atanh_f32_massv
; CHECK-PWR10: __atanhf4_P10
; CHECK-PWR9:  __atanhf4_P9
; CHECK-PWR8:  __atanhf4_P8
; CHECK-PWR7:  __atanhf4_P7
; CHECK-NOT:   __atanhf4_massv
; CHECK-ALL:   blr
;
  %1 = call <4 x float> @__atanhf4(<4 x float> %opnd)
  ret <4 x float> %1 
}

