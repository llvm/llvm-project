; RUN: not llc < %s -mcpu=sm_100 -mattr=+ptx88 -march=nvptx64 2>&1 | FileCheck %s
; RUN: not llc < %s -mcpu=sm_90 -mattr=+ptx78 -march=nvptx64 2>&1 | FileCheck %s --check-prefix=NOF32X2
; RUN: not llc < %s -mcpu=sm_80 -mattr=+ptx78 -march=nvptx64 2>&1 | FileCheck %s --check-prefix=NOBF16

target triple = "nvptx64-nvidia-cuda"

; CHECK: error: {{.*}}llvm.nvvm.fadd.rn.sat with operand type v2f32 is not supported
define <2 x float> @sat_f32x2(<2 x float> %a, <2 x float> %b) {
  %r = call <2 x float> @llvm.nvvm.fadd.rn.sat.v2f32(<2 x float> %a, <2 x float> %b)
  ret <2 x float> %r
}

; NOF32X2: error: {{.*}}llvm.nvvm.fadd.rn with operand type v2f32 is not supported
define <2 x float> @unsupported_f32x2(<2 x float> %a, <2 x float> %b) {
  %r = call <2 x float> @llvm.nvvm.fadd.rn.v2f32(<2 x float> %a, <2 x float> %b)
  ret <2 x float> %r
}

; CHECK: error: {{.*}}llvm.nvvm.fadd.rn.ftz with operand type f64 is not supported
define double @ftz_f64(double %a, double %b) {
  %r = call double @llvm.nvvm.fadd.rn.ftz.f64(double %a, double %b)
  ret double %r
}

; CHECK: error: {{.*}}llvm.nvvm.fadd.rz.sat with operand type f16 is not supported
define half @rz_f16(half %a, half %b) {
  %r = call half @llvm.nvvm.fadd.rz.sat.f16(half %a, half %b)
  ret half %r
}

; CHECK: error: {{.*}}llvm.nvvm.fadd.rn.ftz with operand type bf16 is not supported
define bfloat @ftz_bf16(bfloat %a, bfloat %b) {
  %r = call bfloat @llvm.nvvm.fadd.rn.ftz.bf16(bfloat %a, bfloat %b)
  ret bfloat %r
}

; NOBF16: error: {{.*}}llvm.nvvm.fadd.rn with operand type bf16 is not supported
define bfloat @unsupported_bf16(bfloat %a, bfloat %b) {
  %r = call bfloat @llvm.nvvm.fadd.rn.bf16(bfloat %a, bfloat %b)
  ret bfloat %r
}

; CHECK: error: {{.*}}llvm.nvvm.fadd.rn with operand type v4f32 is not supported
define <4 x float> @v4f32(<4 x float> %a, <4 x float> %b) {
  %r = call <4 x float> @llvm.nvvm.fadd.rn.v4f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %r
}
