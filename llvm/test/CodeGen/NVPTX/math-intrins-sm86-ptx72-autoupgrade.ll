; RUN: llc < %s -march=nvptx64 -mcpu=sm_86 -mattr=+ptx72 | FileCheck %s
; RUN: %if ptxas-11.2 %{ llc < %s -march=nvptx64 -mcpu=sm_86 -mattr=+ptx72 | %ptxas-verify -arch=sm_86 %}

declare half @llvm.nvvm.fmin.xorsign.abs.f16(half, half)
declare half @llvm.nvvm.fmin.ftz.xorsign.abs.f16(half, half)
declare half @llvm.nvvm.fmin.nan.xorsign.abs.f16(half, half)
declare half @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f16(half, half)
declare <2 x half> @llvm.nvvm.fmin.xorsign.abs.f16x2(<2 x half> , <2 x half>)
declare <2 x half> @llvm.nvvm.fmin.ftz.xorsign.abs.f16x2(<2 x half> , <2 x half>)
declare <2 x half> @llvm.nvvm.fmin.nan.xorsign.abs.f16x2(<2 x half> , <2 x half>)
declare <2 x half> @llvm.nvvm.fmin.ftz.nan.xorsign.abs.f16x2(<2 x half> , <2 x half>)
declare i16 @llvm.nvvm.fmin.xorsign.abs.bf16(i16, i16)
declare i16 @llvm.nvvm.fmin.nan.xorsign.abs.bf16(i16, i16)
declare i32 @llvm.nvvm.fmin.xorsign.abs.bf16x2(i32, i32)
declare i32 @llvm.nvvm.fmin.nan.xorsign.abs.bf16x2(i32, i32)
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
declare i16 @llvm.nvvm.fmax.xorsign.abs.bf16(i16, i16)
declare i16 @llvm.nvvm.fmax.nan.xorsign.abs.bf16(i16, i16)
declare i32 @llvm.nvvm.fmax.xorsign.abs.bf16x2(i32, i32)
declare i32 @llvm.nvvm.fmax.nan.xorsign.abs.bf16x2(i32, i32)
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
define i16 @fmin_xorsign_abs_bf16(i16 %0, i16 %1) {
  ; CHECK-NOT: call
  ; CHECK: min.xorsign.abs.bf16
  %res = call i16 @llvm.nvvm.fmin.xorsign.abs.bf16(i16 %0, i16 %1)
  ret i16 %res
}

; CHECK-LABEL: fmin_nan_xorsign_abs_bf16
define i16 @fmin_nan_xorsign_abs_bf16(i16 %0, i16 %1) {
  ; CHECK-NOT: call
  ; CHECK: min.NaN.xorsign.abs.bf16
  %res = call i16 @llvm.nvvm.fmin.nan.xorsign.abs.bf16(i16 %0, i16 %1)
  ret i16 %res
}

; CHECK-LABEL: fmin_xorsign_abs_bf16x2
define i32 @fmin_xorsign_abs_bf16x2(i32 %0, i32 %1) {
  ; CHECK-NOT: call
  ; CHECK: min.xorsign.abs.bf16x2
  %res = call i32 @llvm.nvvm.fmin.xorsign.abs.bf16x2(i32 %0, i32 %1)
  ret i32 %res
}

; CHECK-LABEL: fmin_nan_xorsign_abs_bf16x2
define i32 @fmin_nan_xorsign_abs_bf16x2(i32 %0, i32 %1) {
  ; CHECK-NOT: call
  ; CHECK: min.NaN.xorsign.abs.bf16x2
  %res = call i32 @llvm.nvvm.fmin.nan.xorsign.abs.bf16x2(i32 %0, i32 %1)
  ret i32 %res
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
define i16 @fmax_xorsign_abs_bf16(i16 %0, i16 %1) {
  ; CHECK-NOT: call
  ; CHECK: max.xorsign.abs.bf16
  %res = call i16 @llvm.nvvm.fmax.xorsign.abs.bf16(i16 %0, i16 %1)
  ret i16 %res
}

; CHECK-LABEL: fmax_nan_xorsign_abs_bf16
define i16 @fmax_nan_xorsign_abs_bf16(i16 %0, i16 %1) {
  ; CHECK-NOT: call
  ; CHECK: max.NaN.xorsign.abs.bf16
  %res = call i16 @llvm.nvvm.fmax.nan.xorsign.abs.bf16(i16 %0, i16 %1)
  ret i16 %res
}

; CHECK-LABEL: fmax_xorsign_abs_bf16x2
define i32 @fmax_xorsign_abs_bf16x2(i32 %0, i32 %1) {
  ; CHECK-NOT: call
  ; CHECK: max.xorsign.abs.bf16x2
  %res = call i32 @llvm.nvvm.fmax.xorsign.abs.bf16x2(i32 %0, i32 %1)
  ret i32 %res
}

; CHECK-LABEL: fmax_nan_xorsign_abs_bf16x2
define i32 @fmax_nan_xorsign_abs_bf16x2(i32 %0, i32 %1) {
  ; CHECK-NOT: call
  ; CHECK: max.NaN.xorsign.abs.bf16x2
  %res = call i32 @llvm.nvvm.fmax.nan.xorsign.abs.bf16x2(i32 %0, i32 %1)
  ret i32 %res
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
