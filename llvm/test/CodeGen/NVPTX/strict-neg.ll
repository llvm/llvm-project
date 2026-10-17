; RUN: llc < %s -mcpu=sm_80 -mattr=+ptx70 --nvptx-strict-neg -verify-machineinstrs | FileCheck %s --check-prefix=STRICT --implicit-check-not=neg. --implicit-check-not=cvt.
; RUN: llc < %s -mcpu=sm_80 -mattr=+ptx70 --nvptx-strict-neg -O0 -verify-machineinstrs | FileCheck %s --check-prefix=STRICT --implicit-check-not=neg. --implicit-check-not=cvt.
; RUN: llc < %s -mcpu=sm_100 -mattr=+ptx86 --nvptx-strict-neg -verify-machineinstrs | FileCheck %s --check-prefix=STRICT --implicit-check-not=neg. --implicit-check-not=cvt.
; RUN: llc < %s -mcpu=sm_52 -mattr=+ptx60 --nvptx-strict-neg -verify-machineinstrs | FileCheck %s --check-prefix=STRICT --implicit-check-not=neg. --implicit-check-not=cvt.
; RUN: llc < %s -mcpu=sm_80 -mattr=+ptx70 --nvptx-strict-neg --nvptx-no-f16-math -verify-machineinstrs | FileCheck %s --check-prefix=STRICT --implicit-check-not=neg. --implicit-check-not=cvt.
; RUN: llc < %s -mcpu=sm_80 -mattr=+ptx70 --nvptx-strict-neg=false | FileCheck %s --check-prefix=RELAXED
; RUN: llc < %s -mcpu=sm_80 -mattr=+ptx70 --nvptx-strict-neg=false --nvptx-strict-abs | FileCheck %s --check-prefix=RELAXED
; RUN: %if ptxas-sm_80 && ptxas-isa-7.0 %{ llc < %s -mcpu=sm_80 -mattr=+ptx70 --nvptx-strict-neg | %ptxas-verify -arch=sm_80 %}

target triple = "nvptx64-nvidia-cuda"

define float @fneg_f32(float %a) {
; STRICT-LABEL: fneg_f32(
; STRICT: ld.param{{(::func)?}}.b32 [[A:%r[0-9]+]], [fneg_f32_param_0];
; STRICT: xor.b32 [[R:%r[0-9]+]], [[A]], -2147483648;
; STRICT: st.param{{(::func)?}}.b32 [func_retval0], [[R]];
; RELAXED-LABEL: fneg_f32(
; RELAXED: neg.f32
  %r = fneg float %a
  ret float %r
}

; fneg must preserve subnormals even in a function that flushes arithmetic.
define float @fneg_f32_ftz(float %a) #0 {
; STRICT-LABEL: fneg_f32_ftz(
; STRICT: ld.param{{(::func)?}}.b32 [[A:%r[0-9]+]], [fneg_f32_ftz_param_0];
; STRICT: xor.b32 [[R:%r[0-9]+]], [[A]], -2147483648;
; STRICT: st.param{{(::func)?}}.b32 [func_retval0], [[R]];
; RELAXED-LABEL: fneg_f32_ftz(
; RELAXED: neg.f32
  %r = fneg float %a
  ret float %r
}

define double @fneg_f64(double %a) {
; STRICT-LABEL: fneg_f64(
; STRICT: ld.param{{(::func)?}}.b64 [[A:%rd[0-9]+]], [fneg_f64_param_0];
; STRICT: xor.b64 [[R:%rd[0-9]+]], [[A]], -9223372036854775808;
; STRICT: st.param{{(::func)?}}.b64 [func_retval0], [[R]];
; RELAXED-LABEL: fneg_f64(
; RELAXED: neg.f64
  %r = fneg double %a
  ret double %r
}

define half @fneg_f16(half %a) {
; STRICT-LABEL: fneg_f16(
; STRICT: ld.param{{(::func)?}}.b16 [[A:%rs[0-9]+]], [fneg_f16_param_0];
; STRICT: xor.b16 [[R:%rs[0-9]+]], [[A]], -32768;
; STRICT: st.param{{(::func)?}}.b16 [func_retval0], [[R]];
; RELAXED-LABEL: fneg_f16(
; RELAXED: neg.f16
  %r = fneg half %a
  ret half %r
}

define <2 x half> @fneg_v2f16(<2 x half> %a) {
; STRICT-LABEL: fneg_v2f16(
; STRICT: ld.param{{(::func)?}}.b32 [[A:%r[0-9]+]], [fneg_v2f16_param_0];
; STRICT: xor.b32 [[R:%r[0-9]+]], [[A]], -2147450880;
; STRICT: st.param{{(::func)?}}.b32 [func_retval0], [[R]];
; RELAXED-LABEL: fneg_v2f16(
; RELAXED: neg.f16x2
  %r = fneg <2 x half> %a
  ret <2 x half> %r
}

define bfloat @fneg_bf16(bfloat %a) {
; STRICT-LABEL: fneg_bf16(
; STRICT: ld.param{{(::func)?}}.b16 [[A:%rs[0-9]+]], [fneg_bf16_param_0];
; STRICT: xor.b16 [[R:%rs[0-9]+]], [[A]], -32768;
; STRICT: st.param{{(::func)?}}.b16 [func_retval0], [[R]];
; RELAXED-LABEL: fneg_bf16(
; RELAXED: neg.bf16
  %r = fneg bfloat %a
  ret bfloat %r
}

define <2 x bfloat> @fneg_v2bf16(<2 x bfloat> %a) {
; STRICT-LABEL: fneg_v2bf16(
; STRICT: ld.param{{(::func)?}}.b32 [[A:%r[0-9]+]], [fneg_v2bf16_param_0];
; STRICT: xor.b32 [[R:%r[0-9]+]], [[A]], -2147450880;
; STRICT: st.param{{(::func)?}}.b32 [func_retval0], [[R]];
; RELAXED-LABEL: fneg_v2bf16(
; RELAXED: neg.bf16x2
  %r = fneg <2 x bfloat> %a
  ret <2 x bfloat> %r
}

define <2 x float> @fneg_v2f32(<2 x float> %a) {
; STRICT-LABEL: fneg_v2f32(
; STRICT-DAG: ld.param{{(::func)?}}.v2.b32 {[[A:%r[0-9]+]], [[B:%r[0-9]+]]}, [fneg_v2f32_param_0];
; STRICT-DAG: xor.b32 [[RA:%r[0-9]+]], [[A]], -2147483648;
; STRICT-DAG: xor.b32 [[RB:%r[0-9]+]], [[B]], -2147483648;
; STRICT: st.param{{(::func)?}}.v2.b32 [func_retval0], {[[RA]], [[RB]]};
; RELAXED-LABEL: fneg_v2f32(
; RELAXED-COUNT-2: neg.f32
  %r = fneg <2 x float> %a
  ret <2 x float> %r
}

attributes #0 = { "denormal-fp-math-f32"="preserve-sign" }
