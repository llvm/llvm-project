; RUN: llc < %s -mcpu=sm_80 -mattr=+ptx70 --nvptx-strict-abs -verify-machineinstrs | FileCheck %s --check-prefix=STRICT --implicit-check-not=abs. --implicit-check-not='cvt{{[^;]*\.(bf16|f16|f32|f64)}}'
; RUN: llc < %s -mcpu=sm_80 -mattr=+ptx70 --nvptx-strict-abs -O0 -verify-machineinstrs | FileCheck %s --check-prefix=STRICT --implicit-check-not=abs. --implicit-check-not='cvt{{[^;]*\.(bf16|f16|f32|f64)}}'
; RUN: llc < %s -mcpu=sm_100 -mattr=+ptx86 --nvptx-strict-abs -verify-machineinstrs | FileCheck %s --check-prefix=STRICT --implicit-check-not=abs. --implicit-check-not='cvt{{[^;]*\.(bf16|f16|f32|f64)}}'
; RUN: llc < %s -mcpu=sm_52 -mattr=+ptx60 --nvptx-strict-abs -verify-machineinstrs | FileCheck %s --check-prefix=STRICT --implicit-check-not=abs. --implicit-check-not='cvt{{[^;]*\.(bf16|f16|f32|f64)}}'
; RUN: llc < %s -mcpu=sm_80 -mattr=+ptx70 --nvptx-strict-abs --nvptx-no-f16-math -verify-machineinstrs | FileCheck %s --check-prefix=STRICT --implicit-check-not=abs. --implicit-check-not='cvt{{[^;]*\.(bf16|f16|f32|f64)}}'
; RUN: llc < %s -mcpu=sm_80 -mattr=+ptx70 --nvptx-strict-abs=false | FileCheck %s --check-prefix=RELAXED
; RUN: llc < %s -mcpu=sm_80 -mattr=+ptx70 --nvptx-strict-abs=false --nvptx-strict-neg | FileCheck %s --check-prefix=RELAXED
; RUN: %if ptxas-sm_80 && ptxas-isa-7.0 %{ llc < %s -mcpu=sm_80 -mattr=+ptx70 --nvptx-strict-abs | %ptxas-verify -arch=sm_80 %}

target triple = "nvptx64-nvidia-cuda"

define float @fabs_f32(float %a) {
; STRICT-LABEL: fabs_f32(
; STRICT: ld.param{{(::func)?}}.b32 [[A:%r[0-9]+]], [fabs_f32_param_0];
; STRICT: and.b32 [[SIGN:%r[0-9]+]], [[A]], -2147483648;
; STRICT: xor.b32 [[R:%r[0-9]+]], [[A]], [[SIGN]];
; STRICT: st.param{{(::func)?}}.b32 [func_retval0], [[R]];
; RELAXED-LABEL: fabs_f32(
; RELAXED: abs.f32
  %r = call float @llvm.fabs.f32(float %a)
  ret float %r
}

; fabs must preserve subnormals even in a function that flushes arithmetic.
define float @fabs_f32_ftz(float %a) #0 {
; STRICT-LABEL: fabs_f32_ftz(
; STRICT: ld.param{{(::func)?}}.b32 [[A:%r[0-9]+]], [fabs_f32_ftz_param_0];
; STRICT: and.b32 [[SIGN:%r[0-9]+]], [[A]], -2147483648;
; STRICT: xor.b32 [[R:%r[0-9]+]], [[A]], [[SIGN]];
; STRICT: st.param{{(::func)?}}.b32 [func_retval0], [[R]];
; RELAXED-LABEL: fabs_f32_ftz(
; RELAXED: abs.f32
  %r = call float @llvm.fabs.f32(float %a)
  ret float %r
}

define double @fabs_f64(double %a) {
; STRICT-LABEL: fabs_f64(
; STRICT: ld.param{{(::func)?}}.b64 [[A:%rd[0-9]+]], [fabs_f64_param_0];
; STRICT: mov.b64 {{[{](tmp|_)}}, [[HI:%r[0-9]+]]}, [[A]];
; STRICT: and.b32 [[SIGN:%r[0-9]+]], [[HI]], -2147483648;
; STRICT: xor.b32 [[ABS:%r[0-9]+]], [[HI]], [[SIGN]];
; STRICT: cvt.u32.u64 [[LO:%r[0-9]+]], [[A]];
; STRICT: mov.b64 [[R:%rd[0-9]+]], {[[LO]], [[ABS]]};
; STRICT: st.param{{(::func)?}}.b64 [func_retval0], [[R]];
; RELAXED-LABEL: fabs_f64(
; RELAXED: abs.f64
  %r = call double @llvm.fabs.f64(double %a)
  ret double %r
}

define half @fabs_f16(half %a) {
; STRICT-LABEL: fabs_f16(
; STRICT: ld.param{{(::func)?}}.b16 [[A:%rs[0-9]+]], [fabs_f16_param_0];
; STRICT: and.b16 [[SIGN:%rs[0-9]+]], [[A]], -32768;
; STRICT: xor.b16 [[R:%rs[0-9]+]], [[A]], [[SIGN]];
; STRICT: st.param{{(::func)?}}.b16 [func_retval0], [[R]];
; RELAXED-LABEL: fabs_f16(
; RELAXED: abs.f16
  %r = call half @llvm.fabs.f16(half %a)
  ret half %r
}

define <2 x half> @fabs_v2f16(<2 x half> %a) {
; STRICT-LABEL: fabs_v2f16(
; STRICT: ld.param{{(::func)?}}.b32 [[A:%r[0-9]+]], [fabs_v2f16_param_0];
; STRICT: and.b32 [[R:%r[0-9]+]], [[A]], 2147450879;
; STRICT: st.param{{(::func)?}}.b32 [func_retval0], [[R]];
; RELAXED-LABEL: fabs_v2f16(
; RELAXED: abs.f16x2
  %r = call <2 x half> @llvm.fabs.v2f16(<2 x half> %a)
  ret <2 x half> %r
}

define bfloat @fabs_bf16(bfloat %a) {
; STRICT-LABEL: fabs_bf16(
; STRICT: ld.param{{(::func)?}}.b16 [[A:%rs[0-9]+]], [fabs_bf16_param_0];
; STRICT: and.b16 [[SIGN:%rs[0-9]+]], [[A]], -32768;
; STRICT: xor.b16 [[R:%rs[0-9]+]], [[A]], [[SIGN]];
; STRICT: st.param{{(::func)?}}.b16 [func_retval0], [[R]];
; RELAXED-LABEL: fabs_bf16(
; RELAXED: abs.bf16
  %r = call bfloat @llvm.fabs.bf16(bfloat %a)
  ret bfloat %r
}

define <2 x bfloat> @fabs_v2bf16(<2 x bfloat> %a) {
; STRICT-LABEL: fabs_v2bf16(
; STRICT: ld.param{{(::func)?}}.b32 [[A:%r[0-9]+]], [fabs_v2bf16_param_0];
; STRICT: and.b32 [[R:%r[0-9]+]], [[A]], 2147450879;
; STRICT: st.param{{(::func)?}}.b32 [func_retval0], [[R]];
; RELAXED-LABEL: fabs_v2bf16(
; RELAXED: abs.bf16x2
  %r = call <2 x bfloat> @llvm.fabs.v2bf16(<2 x bfloat> %a)
  ret <2 x bfloat> %r
}

define <2 x float> @fabs_v2f32(<2 x float> %a) {
; STRICT-LABEL: fabs_v2f32(
; STRICT-DAG: ld.param{{(::func)?}}.v2.b32 {[[A:%r[0-9]+]], [[B:%r[0-9]+]]}, [fabs_v2f32_param_0];
; STRICT-DAG: and.b32 [[SIGN_A:%r[0-9]+]], [[A]], -2147483648;
; STRICT-DAG: xor.b32 [[RA:%r[0-9]+]], [[A]], [[SIGN_A]];
; STRICT-DAG: and.b32 [[SIGN_B:%r[0-9]+]], [[B]], -2147483648;
; STRICT-DAG: xor.b32 [[RB:%r[0-9]+]], [[B]], [[SIGN_B]];
; STRICT: st.param{{(::func)?}}.v2.b32 [func_retval0], {[[RA]], [[RB]]};
; RELAXED-LABEL: fabs_v2f32(
; RELAXED-COUNT-2: abs.f32
  %r = call <2 x float> @llvm.fabs.v2f32(<2 x float> %a)
  ret <2 x float> %r
}

declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare half @llvm.fabs.f16(half)
declare <2 x half> @llvm.fabs.v2f16(<2 x half>)
declare bfloat @llvm.fabs.bf16(bfloat)
declare <2 x bfloat> @llvm.fabs.v2bf16(<2 x bfloat>)
declare <2 x float> @llvm.fabs.v2f32(<2 x float>)

attributes #0 = { "denormal-fp-math-f32"="preserve-sign" }
