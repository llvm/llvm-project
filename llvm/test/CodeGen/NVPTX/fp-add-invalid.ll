; RUN: not llc < %s -mcpu=sm_90 -mattr=+ptx78 -march=nvptx64 2>&1 | FileCheck %s --check-prefix=NOF32X2
; RUN: not llc < %s -mcpu=sm_80 -mattr=+ptx78 -march=nvptx64 2>&1 | FileCheck %s --check-prefix=NOBF16
; RUN: not llc < %s -mcpu=sm_50 -mattr=+ptx70 -march=nvptx64 2>&1 | FileCheck %s --check-prefix=NOF16

target triple = "nvptx64-nvidia-cuda"

; NOF16: error: {{.*}}llvm.nvvm.fadd with rounding mode rn and operand type f16 is not supported
define half @unsupported_f16(half %a, half %b) {
  %r = call half @llvm.nvvm.fadd.f16(half %a, half %b, i32 1)
  ret half %r
}

; NOF32X2: error: {{.*}}llvm.nvvm.fadd with rounding mode rn and operand type v2f32 is not supported
define <2 x float> @unsupported_f32x2(<2 x float> %a, <2 x float> %b) {
  %r = call <2 x float> @llvm.nvvm.fadd.v2f32(<2 x float> %a, <2 x float> %b, i32 1)
  ret <2 x float> %r
}

; NOBF16: error: {{.*}}llvm.nvvm.fadd with rounding mode rn and operand type bf16 is not supported
define bfloat @unsupported_bf16(bfloat %a, bfloat %b) {
  %r = call bfloat @llvm.nvvm.fadd.bf16(bfloat %a, bfloat %b, i32 1)
  ret bfloat %r
}
