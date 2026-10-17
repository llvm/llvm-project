; RUN: not llc < %s -mcpu=sm_100 -mattr=+ptx88 -march=nvptx64 2>&1 | FileCheck %s
; RUN: not llc < %s -mcpu=sm_90 -mattr=+ptx78 -march=nvptx64 2>&1 | FileCheck %s --check-prefix=NOF32X2
; RUN: not llc < %s -mcpu=sm_70 -mattr=+ptx70 -march=nvptx64 2>&1 | FileCheck %s --check-prefix=NOBF16

target triple = "nvptx64-nvidia-cuda"

; CHECK: error: {{.*}}llvm.nvvm.fma.sat with rounding mode rn and operand type v2f32 is not supported
define <2 x float> @sat_f32x2(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
  %r = call <2 x float> @llvm.nvvm.fma.sat.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, /* rnd=rn */ i32 1)
  ret <2 x float> %r
}

; NOF32X2: error: {{.*}}llvm.nvvm.fma with rounding mode rn and operand type v2f32 is not supported
define <2 x float> @unsupported_f32x2(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
  %r = call <2 x float> @llvm.nvvm.fma.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, /* rnd=rn */ i32 1)
  ret <2 x float> %r
}

; CHECK: error: {{.*}}llvm.nvvm.fma.ftz with rounding mode rn and operand type f64 is not supported
define double @ftz_f64(double %a, double %b, double %c) {
  %r = call double @llvm.nvvm.fma.ftz.f64(double %a, double %b, double %c, /* rnd=rn */ i32 1)
  ret double %r
}

; CHECK: error: {{.*}}llvm.nvvm.fma.sat with rounding mode rz and operand type f16 is not supported
define half @rz_f16(half %a, half %b, half %c) {
  %r = call half @llvm.nvvm.fma.sat.f16(half %a, half %b, half %c, /* rnd=rz */ i32 0)
  ret half %r
}

; CHECK: error: {{.*}}llvm.nvvm.fma.ftz with rounding mode rn and operand type bf16 is not supported
define bfloat @ftz_bf16(bfloat %a, bfloat %b, bfloat %c) {
  %r = call bfloat @llvm.nvvm.fma.ftz.bf16(bfloat %a, bfloat %b, bfloat %c, /* rnd=rn */ i32 1)
  ret bfloat %r
}

; NOBF16: error: {{.*}}llvm.nvvm.fma with rounding mode rn and operand type bf16 is not supported
define bfloat @unsupported_bf16(bfloat %a, bfloat %b, bfloat %c) {
  %r = call bfloat @llvm.nvvm.fma.bf16(bfloat %a, bfloat %b, bfloat %c, /* rnd=rn */ i32 1)
  ret bfloat %r
}

; CHECK: error: {{.*}}llvm.nvvm.fma.relu with rounding mode rn and operand type f32 is not supported
define float @relu_f32(float %a, float %b, float %c) {
  %r = call float @llvm.nvvm.fma.relu.f32(float %a, float %b, float %c, /* rnd=rn */ i32 1)
  ret float %r
}

; CHECK: error: {{.*}}llvm.nvvm.fma.relu with rounding mode rn and operand type f64 is not supported
define double @relu_f64(double %a, double %b, double %c) {
  %r = call double @llvm.nvvm.fma.relu.f64(double %a, double %b, double %c, /* rnd=rn */ i32 1)
  ret double %r
}

; CHECK: error: {{.*}}llvm.nvvm.fma.relu with rounding mode rz and operand type f16 is not supported
define half @relu_rz_f16(half %a, half %b, half %c) {
  %r = call half @llvm.nvvm.fma.relu.f16(half %a, half %b, half %c, /* rnd=rz */ i32 0)
  ret half %r
}

; CHECK: error: {{.*}}llvm.nvvm.fma with rounding mode rn and operand type v4f32 is not supported
define <4 x float> @v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
  %r = call <4 x float> @llvm.nvvm.fma.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c, /* rnd=rn */ i32 1)
  ret <4 x float> %r
}
