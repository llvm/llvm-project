; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_86 -mattr=+ptx72 | FileCheck %s
; RUN: %if ptxas-sm_86 && ptxas-isa-7.2 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_86 -mattr=+ptx72 | %ptxas-verify -arch=sm_86 %}

declare half @llvm.nvvm.fmin.xorsign.abs.f16(half, half)
declare half @llvm.nvvm.fmin.ftz.xorsign.abs.f16(half, half)
declare half @llvm.nvvm.fmin.nan.xorsign.abs.f16(half, half)
declare half @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f16(half, half)
declare <2 x half> @llvm.nvvm.fmin.xorsign.abs.f16x2(<2 x half> , <2 x half>)
declare <2 x half> @llvm.nvvm.fmin.ftz.xorsign.abs.f16x2(<2 x half> , <2 x half>)
declare <2 x half> @llvm.nvvm.fmin.nan.xorsign.abs.f16x2(<2 x half> , <2 x half>)
declare <2 x half> @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f16x2(<2 x half> , <2 x half>)
declare bfloat @llvm.nvvm.fmin.xorsign.abs.bf16(bfloat, bfloat)
declare bfloat @llvm.nvvm.fmin.nan.xorsign.abs.bf16(bfloat, bfloat)
declare <2 x bfloat> @llvm.nvvm.fmin.xorsign.abs.bf16x2(<2 x bfloat>, <2 x bfloat>)
declare <2 x bfloat> @llvm.nvvm.fmin.nan.xorsign.abs.bf16x2(<2 x bfloat>, <2 x bfloat>)
declare float @llvm.nvvm.fmin.xorsign.abs.f(float, float)
declare float @llvm.nvvm.fmin.ftz.xorsign.abs.f(float, float)
declare float @llvm.nvvm.fmin.nan.xorsign.abs.f(float, float)
declare float @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f(float, float)

declare half @llvm.nvvm.fmax.xorsign.abs.f16(half, half)
declare half @llvm.nvvm.fmax.ftz.xorsign.abs.f16(half, half)
declare half @llvm.nvvm.fmax.nan.xorsign.abs.f16(half, half)
declare half @llvm.nvvm.fmax.ftz.nan.xorsign.abs.f16(half, half)
declare <2 x half> @llvm.nvvm.fmax.xorsign.abs.f16x2(<2 x half> , <2 x half>)
declare <2 x half> @llvm.nvvm.fmax.ftz.xorsign.abs.f16x2(<2 x half> , <2 x half>)
declare <2 x half> @llvm.nvvm.fmax.nan.xorsign.abs.f16x2(<2 x half> , <2 x half>)
declare <2 x half> @llvm.nvvm.fmax.ftz.nan.xorsign.abs.f16x2(<2 x half> , <2 x half>)
declare bfloat @llvm.nvvm.fmax.xorsign.abs.bf16(bfloat, bfloat)
declare bfloat @llvm.nvvm.fmax.nan.xorsign.abs.bf16(bfloat, bfloat)
declare <2 x bfloat> @llvm.nvvm.fmax.xorsign.abs.bf16x2(<2 x bfloat>, <2 x bfloat>)
declare <2 x bfloat> @llvm.nvvm.fmax.nan.xorsign.abs.bf16x2(<2 x bfloat>, <2 x bfloat>)
declare float @llvm.nvvm.fmax.xorsign.abs.f(float, float)
declare float @llvm.nvvm.fmax.ftz.xorsign.abs.f(float, float)
declare float @llvm.nvvm.fmax.nan.xorsign.abs.f(float, float)
declare float @llvm.nvvm.fmax.ftz.nan.xorsign.abs.f(float, float)

; CHECK-LABEL: fmin_xorsign_abs_f16
define half @fmin_xorsign_abs_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: min.xorsign.abs.f16
  %res = call half @llvm.nvvm.fmin.xorsign.abs.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmin_ftz_xorsign_abs_f16
define half @fmin_ftz_xorsign_abs_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: min.ftz.xorsign.abs.f16
  %res = call half @llvm.nvvm.fmin.ftz.xorsign.abs.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmin_nan_xorsign_abs_f16
define half @fmin_nan_xorsign_abs_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: min.NaN.xorsign.abs.f16
  %res = call half @llvm.nvvm.fmin.nan.xorsign.abs.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmin_ftz_nan_xorsign_abs_f16
define half @fmin_ftz_nan_xorsign_abs_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: min.ftz.NaN.xorsign.abs.f16
  %res = call half @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmin_xorsign_abs_f16x2
define <2 x half> @fmin_xorsign_abs_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: min.xorsign.abs.f16x2
  %res = call <2 x half> @llvm.nvvm.fmin.xorsign.abs.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmin_ftz_xorsign_abs_f16x2
define <2 x half> @fmin_ftz_xorsign_abs_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: min.ftz.xorsign.abs.f16x2
  %res = call <2 x half> @llvm.nvvm.fmin.ftz.xorsign.abs.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmin_nan_xorsign_abs_f16x2
define <2 x half> @fmin_nan_xorsign_abs_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: min.NaN.xorsign.abs.f16x2
  %res = call <2 x half> @llvm.nvvm.fmin.nan.xorsign.abs.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmin_ftz_nan_xorsign_abs_f16x2
define <2 x half> @fmin_ftz_nan_xorsign_abs_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: min.ftz.NaN.xorsign.abs.f16x2
  %res = call <2 x half> @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmin_xorsign_abs_bf16
define bfloat @fmin_xorsign_abs_bf16(bfloat %0, bfloat %1) {
  ; CHECK-NOT: call
  ; CHECK: min.xorsign.abs.bf16
  %res = call bfloat @llvm.nvvm.fmin.xorsign.abs.bf16(bfloat %0, bfloat %1)
  ret bfloat %res
}

; CHECK-LABEL: fmin_nan_xorsign_abs_bf16
define bfloat @fmin_nan_xorsign_abs_bf16(bfloat %0, bfloat %1) {
  ; CHECK-NOT: call
  ; CHECK: min.NaN.xorsign.abs.bf16
  %res = call bfloat @llvm.nvvm.fmin.nan.xorsign.abs.bf16(bfloat %0, bfloat %1)
  ret bfloat %res
}

; CHECK-LABEL: fmin_xorsign_abs_bf16x2
define <2 x bfloat> @fmin_xorsign_abs_bf16x2(<2 x bfloat> %0, <2 x bfloat> %1) {
  ; CHECK-NOT: call
  ; CHECK: min.xorsign.abs.bf16x2
  %res = call <2 x bfloat> @llvm.nvvm.fmin.xorsign.abs.bf16x2(<2 x bfloat> %0, <2 x bfloat> %1)
  ret <2 x bfloat> %res
}

; CHECK-LABEL: fmin_nan_xorsign_abs_bf16x2
define <2 x bfloat> @fmin_nan_xorsign_abs_bf16x2(<2 x bfloat> %0, <2 x bfloat> %1) {
  ; CHECK-NOT: call
  ; CHECK: min.NaN.xorsign.abs.bf16x2
  %res = call <2 x bfloat> @llvm.nvvm.fmin.nan.xorsign.abs.bf16x2(<2 x bfloat> %0, <2 x bfloat> %1)
  ret <2 x bfloat> %res
}

; CHECK-LABEL: fmin_xorsign_abs_f
define float @fmin_xorsign_abs_f(float %0, float %1) {
  ; CHECK-NOT: call
  ; CHECK: min.xorsign.abs.f
  %res = call float @llvm.nvvm.fmin.xorsign.abs.f(float %0, float %1)
  ret float %res
}

; CHECK-LABEL: fmin_ftz_xorsign_abs_f
define float @fmin_ftz_xorsign_abs_f(float %0, float %1) {
  ; CHECK-NOT: call
  ; CHECK: min.ftz.xorsign.abs.f
  %res = call float @llvm.nvvm.fmin.ftz.xorsign.abs.f(float %0, float %1)
  ret float %res
}

; CHECK-LABEL: fmin_nan_xorsign_abs_f
define float @fmin_nan_xorsign_abs_f(float %0, float %1) {
  ; CHECK-NOT: call
  ; CHECK: min.NaN.xorsign.abs.f
  %res = call float @llvm.nvvm.fmin.nan.xorsign.abs.f(float %0, float %1)
  ret float %res
}

; CHECK-LABEL: fmin_ftz_nan_xorsign_abs_f
define float @fmin_ftz_nan_xorsign_abs_f(float %0, float %1) {
  ; CHECK-NOT: call
  ; CHECK: min.ftz.NaN.xorsign.abs.f
  %res = call float @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f(float %0, float %1)
  ret float %res
}

; CHECK-LABEL: fmax_xorsign_abs_f16
define half @fmax_xorsign_abs_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: max.xorsign.abs.f16
  %res = call half @llvm.nvvm.fmax.xorsign.abs.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmax_ftz_xorsign_abs_f16
define half @fmax_ftz_xorsign_abs_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: max.ftz.xorsign.abs.f16
  %res = call half @llvm.nvvm.fmax.ftz.xorsign.abs.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmax_nan_xorsign_abs_f16
define half @fmax_nan_xorsign_abs_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: max.NaN.xorsign.abs.f16
  %res = call half @llvm.nvvm.fmax.nan.xorsign.abs.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmax_ftz_nan_xorsign_abs_f16
define half @fmax_ftz_nan_xorsign_abs_f16(half %0, half %1) {
  ; CHECK-NOT: call
  ; CHECK: max.ftz.NaN.xorsign.abs.f16
  %res = call half @llvm.nvvm.fmax.ftz.nan.xorsign.abs.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmax_xorsign_abs_f16x2
define <2 x half> @fmax_xorsign_abs_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: max.xorsign.abs.f16x2
  %res = call <2 x half> @llvm.nvvm.fmax.xorsign.abs.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmax_ftz_xorsign_abs_f16x2
define <2 x half> @fmax_ftz_xorsign_abs_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: max.ftz.xorsign.abs.f16x2
  %res = call <2 x half> @llvm.nvvm.fmax.ftz.xorsign.abs.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmax_nan_xorsign_abs_f16x2
define <2 x half> @fmax_nan_xorsign_abs_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: max.NaN.xorsign.abs.f16x2
  %res = call <2 x half> @llvm.nvvm.fmax.nan.xorsign.abs.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmax_ftz_nan_xorsign_abs_f16x2
define <2 x half> @fmax_ftz_nan_xorsign_abs_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: call
  ; CHECK: max.ftz.NaN.xorsign.abs.f16x2
  %res = call <2 x half> @llvm.nvvm.fmax.ftz.nan.xorsign.abs.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmax_xorsign_abs_bf16
define bfloat @fmax_xorsign_abs_bf16(bfloat %0, bfloat %1) {
  ; CHECK-NOT: call
  ; CHECK: max.xorsign.abs.bf16
  %res = call bfloat @llvm.nvvm.fmax.xorsign.abs.bf16(bfloat %0, bfloat %1)
  ret bfloat %res
}

; CHECK-LABEL: fmax_nan_xorsign_abs_bf16
define bfloat @fmax_nan_xorsign_abs_bf16(bfloat %0, bfloat %1) {
  ; CHECK-NOT: call
  ; CHECK: max.NaN.xorsign.abs.bf16
  %res = call bfloat @llvm.nvvm.fmax.nan.xorsign.abs.bf16(bfloat %0, bfloat %1)
  ret bfloat %res
}

; CHECK-LABEL: fmax_xorsign_abs_bf16x2
define <2 x bfloat> @fmax_xorsign_abs_bf16x2(<2 x bfloat> %0, <2 x bfloat> %1) {
  ; CHECK-NOT: call
  ; CHECK: max.xorsign.abs.bf16x2
  %res = call <2 x bfloat> @llvm.nvvm.fmax.xorsign.abs.bf16x2(<2 x bfloat> %0, <2 x bfloat> %1)
  ret <2 x bfloat> %res
}

; CHECK-LABEL: fmax_nan_xorsign_abs_bf16x2
define <2 x bfloat> @fmax_nan_xorsign_abs_bf16x2(<2 x bfloat> %0, <2 x bfloat> %1) {
  ; CHECK-NOT: call
  ; CHECK: max.NaN.xorsign.abs.bf16x2
  %res = call <2 x bfloat> @llvm.nvvm.fmax.nan.xorsign.abs.bf16x2(<2 x bfloat> %0, <2 x bfloat> %1)
  ret <2 x bfloat> %res
}

; CHECK-LABEL: fmax_xorsign_abs_f
define float @fmax_xorsign_abs_f(float %0, float %1) {
  ; CHECK-NOT: call
  ; CHECK: max.xorsign.abs.f
  %res = call float @llvm.nvvm.fmax.xorsign.abs.f(float %0, float %1)
  ret float %res
}

; CHECK-LABEL: fmax_ftz_xorsign_abs_f
define float @fmax_ftz_xorsign_abs_f(float %0, float %1) {
  ; CHECK-NOT: call
  ; CHECK: max.ftz.xorsign.abs.f
  %res = call float @llvm.nvvm.fmax.ftz.xorsign.abs.f(float %0, float %1)
  ret float %res
}

; CHECK-LABEL: fmax_nan_xorsign_abs_f
define float @fmax_nan_xorsign_abs_f(float %0, float %1) {
  ; CHECK-NOT: call
  ; CHECK: max.NaN.xorsign.abs.f
  %res = call float @llvm.nvvm.fmax.nan.xorsign.abs.f(float %0, float %1)
  ret float %res
}

; CHECK-LABEL: fmax_ftz_nan_xorsign_abs_f
define float @fmax_ftz_nan_xorsign_abs_f(float %0, float %1) {
  ; CHECK-NOT: call
  ; CHECK: max.ftz.NaN.xorsign.abs.f
  %res = call float @llvm.nvvm.fmax.ftz.nan.xorsign.abs.f(float %0, float %1)
  ret float %res
}
