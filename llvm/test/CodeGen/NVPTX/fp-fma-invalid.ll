; RUN: not llc < %s -mcpu=sm_90 -mattr=+ptx78 -march=nvptx64 2>&1 | FileCheck %s --check-prefix=NOF32X2
; RUN: not llc < %s -mcpu=sm_70 -mattr=+ptx70 -march=nvptx64 2>&1 | FileCheck %s --check-prefix=NOBF16
; RUN: not llc < %s -mcpu=sm_75 -mattr=+ptx70 -march=nvptx64 2>&1 | FileCheck %s --check-prefix=NORELU
; RUN: not llc < %s -mcpu=sm_50 -mattr=+ptx70 -march=nvptx64 2>&1 | FileCheck %s --check-prefix=NOF16

target triple = "nvptx64-nvidia-cuda"

; NOF16: error: {{.*}}llvm.nvvm.fma with rounding mode rn and operand type f16 is not supported
define half @unsupported_f16(half %a, half %b, half %c) {
  %r = call half @llvm.nvvm.fma.f16(half %a, half %b, half %c, /* rnd=rn */ i32 1)
  ret half %r
}

; NORELU: error: {{.*}}llvm.nvvm.fma.relu with rounding mode rn and operand type v2f16 is not supported
define <2 x half> @unsupported_relu_v2f16(<2 x half> %a, <2 x half> %b, <2 x half> %c) {
  %r = call <2 x half> @llvm.nvvm.fma.relu.v2f16(<2 x half> %a, <2 x half> %b, <2 x half> %c, /* rnd=rn */ i32 1)
  ret <2 x half> %r
}

; NOF32X2: error: {{.*}}llvm.nvvm.fma with rounding mode rn and operand type v2f32 is not supported
define <2 x float> @unsupported_f32x2(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
  %r = call <2 x float> @llvm.nvvm.fma.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c, /* rnd=rn */ i32 1)
  ret <2 x float> %r
}

; NOBF16: error: {{.*}}llvm.nvvm.fma with rounding mode rn and operand type bf16 is not supported
define bfloat @unsupported_bf16(bfloat %a, bfloat %b, bfloat %c) {
  %r = call bfloat @llvm.nvvm.fma.bf16(bfloat %a, bfloat %b, bfloat %c, /* rnd=rn */ i32 1)
  ret bfloat %r
}
