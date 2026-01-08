; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | FileCheck %s
; RUN: %if ptxas-sm_80 && ptxas-isa-7.0 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | %ptxas-verify -arch=sm_80 %}

declare bfloat @llvm.nvvm.abs.bf16(bfloat)
declare <2 x bfloat> @llvm.nvvm.abs.bf16x2(<2 x bfloat>)

; CHECK-LABEL: abs_bf16
define bfloat @abs_bf16(bfloat %0) {
  ; CHECK-NOT: call
  ; CHECK: abs.bf16
  %res = call bfloat @llvm.nvvm.abs.bf16(bfloat %0);
  ret bfloat %res
}

; CHECK-LABEL: abs_bf16x2
define <2 x bfloat> @abs_bf16x2(<2 x bfloat> %0) {
  ; CHECK-NOT: call
  ; CHECK: abs.bf16x2
  %res = call <2 x bfloat> @llvm.nvvm.abs.bf16x2(<2 x bfloat> %0);
  ret <2 x bfloat> %res
}

; CHECK-LABEL: neg_bf16
define bfloat @neg_bf16(bfloat %0) {
  ; CHECK-NOT: call
  ; CHECK: neg.bf16
  %res = call bfloat @llvm.nvvm.neg.bf16(bfloat %0);
  ret bfloat %res
}

; CHECK-LABEL: neg_bf16x2
define <2 x bfloat> @neg_bf16x2(<2 x bfloat> %0) {
  ; CHECK-NOT: call
  ; CHECK: neg.bf16x2
  %res = call <2 x bfloat> @llvm.nvvm.neg.bf16x2(<2 x bfloat> %0);
  ret <2 x bfloat> %res
}

; CHECK-LABEL: fmin_nan_f
define float @fmin_nan_f(float %0, float %1) {
  ; CHECK-NOT: call
  ; CHECK: min.NaN.f32
  %res = call float @llvm.nvvm.fmin.nan.f(float %0, float %1);
  ret float %res
}

; CHECK-LABEL: fmin_ftz_nan_f
define float @fmin_ftz_nan_f(float %0, float %1) {
  ; CHECK-NOT: call
  ; CHECK: min.ftz.NaN.f32
  %res = call float @llvm.nvvm.fmin.ftz.nan.f(float %0, float %1);
  ret float %res
}

; CHECK-LABEL: fmin_f16
define half @fmin_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: min.f16
  %res = call half @llvm.nvvm.fmin.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmin_ftz_f16
define half @fmin_ftz_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: min.ftz.f16
  %res = call half @llvm.nvvm.fmin.ftz.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmin_nan_f16
define half @fmin_nan_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: min.NaN.f16
  %res = call half @llvm.nvvm.fmin.nan.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmin_ftz_nan_f16
define half @fmin_ftz_nan_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: min.ftz.NaN.f16
  %res = call half @llvm.nvvm.fmin.ftz.nan.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmin_f16x2
define <2 x half> @fmin_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: min.f16x2
  %res = call <2 x half> @llvm.nvvm.fmin.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmin_ftz_f16x2
define <2 x half> @fmin_ftz_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: min.ftz.f16x2
  %res = call <2 x half> @llvm.nvvm.fmin.ftz.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmin_nan_f16x2
define <2 x half> @fmin_nan_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: min.NaN.f16x2
  %res = call <2 x half> @llvm.nvvm.fmin.nan.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmin_ftz_nan_f16x2
define <2 x half> @fmin_ftz_nan_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: min.ftz.NaN.f16x2
  %res = call <2 x half> @llvm.nvvm.fmin.ftz.nan.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmin_bf16
define bfloat @fmin_bf16(bfloat %0, bfloat %1) {
  ; CHECK-NOT: call
  ; CHECK: min.bf16
  %res = call bfloat @llvm.nvvm.fmin.bf16(bfloat %0, bfloat %1)
  ret bfloat %res
}

; CHECK-LABEL: fmin_nan_bf16
define bfloat @fmin_nan_bf16(bfloat %0, bfloat %1) {
  ; CHECK-NOT: call
  ; CHECK: min.NaN.bf16
  %res = call bfloat @llvm.nvvm.fmin.nan.bf16(bfloat %0, bfloat %1)
  ret bfloat %res
}

; CHECK-LABEL: fmin_bf16x2
define <2 x bfloat> @fmin_bf16x2(<2 x bfloat> %0, <2 x bfloat> %1) {
  ; CHECK-NOT: call
  ; CHECK: min.bf16x2
  %res = call <2 x bfloat> @llvm.nvvm.fmin.bf16x2(<2 x bfloat> %0, <2 x bfloat> %1)
  ret <2 x bfloat> %res
}

; CHECK-LABEL: fmin_nan_bf16x2
define <2 x bfloat> @fmin_nan_bf16x2(<2 x bfloat> %0, <2 x bfloat> %1) {
  ; CHECK-NOT: call
  ; CHECK: min.NaN.bf16x2
  %res = call <2 x bfloat> @llvm.nvvm.fmin.nan.bf16x2(<2 x bfloat> %0, <2 x bfloat> %1)
  ret <2 x bfloat> %res
}

; CHECK-LABEL: fmax_nan_f
define float @fmax_nan_f(float %0, float %1) {
  ; CHECK-NOT: call
  ; CHECK: max.NaN.f32
  %res = call float @llvm.nvvm.fmax.nan.f(float %0, float %1);
  ret float %res
}

; CHECK-LABEL: fmax_ftz_nan_f
define float @fmax_ftz_nan_f(float %0, float %1) {
  ; CHECK-NOT: call
  ; CHECK: max.ftz.NaN.f32
  %res = call float @llvm.nvvm.fmax.ftz.nan.f(float %0, float %1);
  ret float %res
}

; CHECK-LABEL: fmax_f16
define half @fmax_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: max.f16
  %res = call half @llvm.nvvm.fmax.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmax_ftz_f16
define half @fmax_ftz_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: max.ftz.f16
  %res = call half @llvm.nvvm.fmax.ftz.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmax_nan_f16
define half @fmax_nan_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: max.NaN.f16
  %res = call half @llvm.nvvm.fmax.nan.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmax_ftz_nan_f16
define half @fmax_ftz_nan_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: max.ftz.NaN.f16
  %res = call half @llvm.nvvm.fmax.ftz.nan.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmax_f16x2
define <2 x half> @fmax_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: max.f16x2
  %res = call <2 x half> @llvm.nvvm.fmax.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmax_ftz_f16x2
define <2 x half> @fmax_ftz_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: max.ftz.f16x2
  %res = call <2 x half> @llvm.nvvm.fmax.ftz.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmax_nan_f16x2
define <2 x half> @fmax_nan_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: max.NaN.f16x2
  %res = call <2 x half> @llvm.nvvm.fmax.nan.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmax_ftz_nan_f16x2
define <2 x half> @fmax_ftz_nan_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: max.ftz.NaN.f16x2
  %res = call <2 x half> @llvm.nvvm.fmax.ftz.nan.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmax_bf16
define bfloat @fmax_bf16(bfloat %0, bfloat %1) {
  ; CHECK-NOT: call
  ; CHECK: max.bf16
  %res = call bfloat @llvm.nvvm.fmax.bf16(bfloat %0, bfloat %1)
  ret bfloat %res
}

; CHECK-LABEL: fmax_nan_bf16
define bfloat @fmax_nan_bf16(bfloat %0, bfloat %1) {
  ; CHECK-NOT: call
  ; CHECK: max.NaN.bf16
  %res = call bfloat @llvm.nvvm.fmax.nan.bf16(bfloat %0, bfloat %1)
  ret bfloat %res
}

; CHECK-LABEL: fmax_bf16x2
define <2 x bfloat> @fmax_bf16x2(<2 x bfloat> %0, <2 x bfloat> %1) {
  ; CHECK-NOT: call
  ; CHECK: max.bf16x2
  %res = call <2 x bfloat> @llvm.nvvm.fmax.bf16x2(<2 x bfloat> %0, <2 x bfloat> %1)
  ret <2 x bfloat> %res
}

; CHECK-LABEL: fmax_nan_bf16x2
define <2 x bfloat> @fmax_nan_bf16x2(<2 x bfloat> %0, <2 x bfloat> %1) {
  ; CHECK-NOT: call
  ; CHECK: max.NaN.bf16x2
  %res = call <2 x bfloat> @llvm.nvvm.fmax.nan.bf16x2(<2 x bfloat> %0, <2 x bfloat> %1)
  ret <2 x bfloat> %res
}

; CHECK-LABEL: fma_rn_relu_f16
define half @fma_rn_relu_f16(half %0, half %1, half %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.relu.f16
  %res = call half @llvm.nvvm.fma.rn.relu.f16(half %0, half %1, half %2)
  ret half %res
}

; CHECK-LABEL: fma_rn_ftz_relu_f16
define half @fma_rn_ftz_relu_f16(half %0, half %1, half %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.ftz.relu.f16
  %res = call half @llvm.nvvm.fma.rn.ftz.relu.f16(half %0, half %1, half %2)
  ret half %res
}

; CHECK-LABEL: fma_rn_relu_f16x2
define <2 x half> @fma_rn_relu_f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.relu.f16x2
  %res = call <2 x half> @llvm.nvvm.fma.rn.relu.f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2)
  ret <2 x half> %res
}

; CHECK-LABEL: fma_rn_ftz_relu_f16x2
define <2 x half> @fma_rn_ftz_relu_f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.ftz.relu.f16x2
  %res = call <2 x half> @llvm.nvvm.fma.rn.ftz.relu.f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2)
  ret <2 x half> %res
}

; CHECK-LABEL: fma_rn_bf16
define bfloat @fma_rn_bf16(bfloat %0, bfloat %1, bfloat %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.bf16
  %res = call bfloat @llvm.nvvm.fma.rn.bf16(bfloat %0, bfloat %1, bfloat %2)
  ret bfloat %res
}

; CHECK-LABEL: fma_rn_relu_bf16
define bfloat @fma_rn_relu_bf16(bfloat %0, bfloat %1, bfloat %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.relu.bf16
  %res = call bfloat @llvm.nvvm.fma.rn.relu.bf16(bfloat %0, bfloat %1, bfloat %2)
  ret bfloat %res
}

; CHECK-LABEL: fma_rn_bf16x2
define <2 x bfloat> @fma_rn_bf16x2(<2 x bfloat> %0, <2 x bfloat> %1, <2 x bfloat> %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.bf16x2
  %res = call <2 x bfloat> @llvm.nvvm.fma.rn.bf16x2(<2 x bfloat> %0, <2 x bfloat> %1, <2 x bfloat> %2)
  ret <2 x bfloat> %res
}

; CHECK-LABEL: fma_rn_relu_bf16x2
define <2 x bfloat> @fma_rn_relu_bf16x2(<2 x bfloat> %0, <2 x bfloat> %1, <2 x bfloat> %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.relu.bf16x2
  %res = call <2 x bfloat> @llvm.nvvm.fma.rn.relu.bf16x2(<2 x bfloat> %0, <2 x bfloat> %1, <2 x bfloat> %2)
  ret <2 x bfloat> %res
}
