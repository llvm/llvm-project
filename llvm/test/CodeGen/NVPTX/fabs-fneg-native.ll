; RUN: llc < %s -mcpu=sm_80 -mattr=+ptx70 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,SM80
; RUN: llc < %s -mcpu=sm_100 -mattr=+ptx88 -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,SM100
; RUN: llc < %s -mcpu=sm_53 -mattr=+ptx60 -verify-machineinstrs | FileCheck %s --check-prefix=SM53
; RUN: %if ptxas-sm_100 && ptxas-isa-8.8 %{ llc < %s -mcpu=sm_100 -mattr=+ptx88 | %ptxas-verify -arch=sm_100 %}

target triple = "nvptx64-nvidia-cuda"

; These consumers cannot observe the sign or payload of a NaN.
define float @fabs_max(float %x, float %y) {
; CHECK-LABEL: fabs_max(
; CHECK: abs.f32
; CHECK: max.f32
  %a = call float @llvm.fabs.f32(float %x)
  %r = call float @llvm.maxnum.f32(float %a, float %y)
  ret float %r
}

define double @fneg_max(double %x, double %y) {
; CHECK-LABEL: fneg_max(
; CHECK: neg.f64
; CHECK: max.f64
  %a = fneg double %x
  %r = call double @llvm.maxnum.f64(double %a, double %y)
  ret double %r
}

define i1 @fabs_compare(float %x) {
; CHECK-LABEL: fabs_compare(
; CHECK: abs.f32
; CHECK: setp.ne.f32
  %a = call float @llvm.fabs.f32(float %x)
  %r = fcmp one float %a, 0x7FF0000000000000
  ret i1 %r
}

define i1 @fabs_infinite(double %x) {
; CHECK-LABEL: fabs_infinite(
; CHECK: abs.f64
; CHECK: setp.eq.f64
  %a = call double @llvm.fabs.f64(double %x)
  %r = fcmp oeq double %a, 0x7FF0000000000000
  ret i1 %r
}

; Non-NaN inputs and nnan flags also make a native sign operation exact.
define float @fabs_nnan(float %x) {
; CHECK-LABEL: fabs_nnan(
; CHECK: abs.f32
  %a = call nnan float @llvm.fabs.f32(float %x)
  ret float %a
}

define double @fneg_nnan(double %x) {
; CHECK-LABEL: fneg_nnan(
; CHECK: neg.f64
  %a = fneg nnan double %x
  ret double %a
}

define double @fneg_integer(i32 %x) {
; CHECK-LABEL: fneg_integer(
; CHECK: cvt.rn.f64.u32
; CHECK: neg.f64
  %f = uitofp i32 %x to double
  %r = fneg double %f
  ret double %r
}

; Keep scalar and packed half/bfloat recovery when those instructions exist.
define half @fabs_half(half %x, half %y) {
; CHECK-LABEL: fabs_half(
; CHECK: abs.f16
  %a = call half @llvm.fabs.f16(half %x)
  %r = call half @llvm.maxnum.f16(half %a, half %y)
  ret half %r
}

define half @fneg_half(half %x, half %y) {
; CHECK-LABEL: fneg_half(
; CHECK: neg.f16
  %a = fneg half %x
  %r = call half @llvm.maxnum.f16(half %a, half %y)
  ret half %r
}

define <2 x half> @fabs_half2(<2 x half> %x, <2 x half> %y) {
; CHECK-LABEL: fabs_half2(
; CHECK: abs.f16x2
  %a = call <2 x half> @llvm.fabs.v2f16(<2 x half> %x)
  %r = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %a, <2 x half> %y)
  ret <2 x half> %r
}

define <2 x bfloat> @fneg_bfloat2(<2 x bfloat> %x, <2 x bfloat> %y) {
; CHECK-LABEL: fneg_bfloat2(
; CHECK: neg.bf16x2
  %a = fneg <2 x bfloat> %x
  %r = call <2 x bfloat> @llvm.maxnum.v2bf16(<2 x bfloat> %a, <2 x bfloat> %y)
  ret <2 x bfloat> %r
}

; nnan permits native instructions, but does not permit flushing subnormals.
define float @fabs_nnan_ftz(float %x) #0 {
; CHECK-LABEL: fabs_nnan_ftz(
; CHECK: abs.f32
; CHECK-NOT: .ftz
; CHECK: ret;
  %r = call nnan float @llvm.fabs.f32(float %x)
  ret float %r
}

define half @fneg_half_nnan_ftz(half %x) #0 {
; CHECK-LABEL: fneg_half_nnan_ftz(
; CHECK: neg.f16
; CHECK-NOT: .ftz
; CHECK: ret;
; SM53-LABEL: fneg_half_nnan_ftz(
; SM53: neg.f16
; SM53-NOT: .ftz
; SM53: ret;
  %r = fneg nnan half %x
  ret half %r
}

; Type legalization inserts BUILD_VECTOR between scalar abs and vector mul.
define <2 x float> @fabs_vector_mul(<2 x float> %x, <2 x float> %y) {
; CHECK-LABEL: fabs_vector_mul(
; CHECK-COUNT-2: abs.f32
; SM100: mul.rn.f32x2
  %a = call <2 x float> @llvm.fabs.v2f32(<2 x float> %x)
  %r = fmul <2 x float> %a, %y
  ret <2 x float> %r
}

define <2 x float> @fabs_vector_mixed(<2 x float> %x, <2 x float> %y, ptr %out) {
; CHECK-LABEL: fabs_vector_mixed(
; CHECK-NOT: abs.f32
; CHECK-COUNT-2: copysign.f32
; CHECK-NOT: abs.f32
; CHECK: ret;
  %a = call <2 x float> @llvm.fabs.v2f32(<2 x float> %x)
  store <2 x float> %a, ptr %out
  %r = fmul <2 x float> %a, %y
  ret <2 x float> %r
}

; A store still observes the exact bits even if another use ignores NaN bits.
define float @fabs_mixed_uses(float %x, float %y, ptr %out) {
; CHECK-LABEL: fabs_mixed_uses(
; CHECK-NOT: abs.f32
; CHECK: copysign.f32
; CHECK-NOT: abs.f32
; CHECK: ret;
  %a = call float @llvm.fabs.f32(float %x)
  store float %a, ptr %out
  %r = call float @llvm.maxnum.f32(float %a, float %y)
  ret float %r
}

define double @fneg_mixed_uses(double %x, double %y, ptr %out) {
; CHECK-LABEL: fneg_mixed_uses(
; CHECK-NOT: neg.f64
; CHECK: xor.b64
; CHECK-NOT: neg.f64
; CHECK: ret;
  %a = fneg double %x
  store double %a, ptr %out
  %r = call double @llvm.maxnum.f64(double %a, double %y)
  ret double %r
}

; A canonicalizing producer alone does not permit native abs/neg: the returned
; value must preserve that producer's NaN bits, apart from the specified sign.
define float @fabs_canonical_input(float %x, float %y) {
; CHECK-LABEL: fabs_canonical_input(
; CHECK-NOT: abs.f32
; CHECK: copysign.f32
  %f = fadd float %x, %y
  %r = call float @llvm.fabs.f32(float %f)
  ret float %r
}

define double @fneg_canonical_input(double %x, double %y) {
; CHECK-LABEL: fneg_canonical_input(
; CHECK-NOT: neg.f64
; CHECK: xor.b64
  %f = fadd double %x, %y
  %r = fneg double %f
  ret double %r
}

; On Blackwell, min/max combines happen before abs legalization. Recognize the
; target's three-input nodes, and keep the .abs instruction-selection pattern.
define float @fabs_max3(float %x, float %y, float %z) {
; SM80-LABEL: fabs_max3(
; SM80-COUNT-3: abs.f32
; SM100-LABEL: fabs_max3(
; SM100-NOT: copysign
; SM100: max.abs.f32
  %a = call float @llvm.fabs.f32(float %x)
  %b = call float @llvm.fabs.f32(float %y)
  %c = call float @llvm.fabs.f32(float %z)
  %ab = call float @llvm.maxnum.f32(float %a, float %b)
  %r = call float @llvm.maxnum.f32(float %ab, float %c)
  ret float %r
}

define float @fabs_minimum3(float %x, float %y, float %z) {
; SM80-LABEL: fabs_minimum3(
; SM80-COUNT-3: abs.f32
; SM100-LABEL: fabs_minimum3(
; SM100-NOT: copysign
; SM100: min.NaN.abs.f32
  %a = call float @llvm.fabs.f32(float %x)
  %b = call float @llvm.fabs.f32(float %y)
  %c = call float @llvm.fabs.f32(float %z)
  %ab = call float @llvm.minimum.f32(float %a, float %b)
  %r = call float @llvm.minimum.f32(float %ab, float %c)
  ret float %r
}

declare <2 x float> @llvm.fabs.v2f32(<2 x float>)
declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare half @llvm.fabs.f16(half)
declare <2 x half> @llvm.fabs.v2f16(<2 x half>)
declare float @llvm.maxnum.f32(float, float)
declare double @llvm.maxnum.f64(double, double)
declare half @llvm.maxnum.f16(half, half)
declare <2 x half> @llvm.maxnum.v2f16(<2 x half>, <2 x half>)
declare <2 x bfloat> @llvm.maxnum.v2bf16(<2 x bfloat>, <2 x bfloat>)
declare float @llvm.minimum.f32(float, float)

attributes #0 = { "denormal-fp-math-f32"="preserve-sign" }
