; RUN: llc < %s -march=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | FileCheck %s

declare i16 @llvm.nvvm.abs.bf16(i16)
declare i32 @llvm.nvvm.abs.bf16x2(i32)
declare i16 @llvm.nvvm.neg.bf16(i16)
declare i32 @llvm.nvvm.neg.bf16x2(i32)

declare float @llvm.nvvm.fmin.nan.f(float, float)
declare float @llvm.nvvm.fmin.ftz.nan.f(float, float)
declare half @llvm.nvvm.fmin.f16(half, half)
declare half @llvm.nvvm.fmin.ftz.f16(half, half)
declare half @llvm.nvvm.fmin.nan.f16(half, half)
declare half @llvm.nvvm.fmin.ftz.nan.f16(half, half)
declare <2 x half> @llvm.nvvm.fmin.f16x2(<2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fmin.ftz.f16x2(<2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fmin.nan.f16x2(<2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fmin.ftz.nan.f16x2(<2 x half>, <2 x half>)
declare i16 @llvm.nvvm.fmin.bf16(i16, i16)
declare i16 @llvm.nvvm.fmin.nan.bf16(i16, i16)
declare i32 @llvm.nvvm.fmin.bf16x2(i32, i32)
declare i32 @llvm.nvvm.fmin.nan.bf16x2(i32, i32)

declare float @llvm.nvvm.fmax.nan.f(float, float)
declare float @llvm.nvvm.fmax.ftz.nan.f(float, float)
declare half @llvm.nvvm.fmax.f16(half, half)
declare half @llvm.nvvm.fmax.ftz.f16(half, half)
declare half @llvm.nvvm.fmax.nan.f16(half, half)
declare half @llvm.nvvm.fmax.ftz.nan.f16(half, half)
declare <2 x half> @llvm.nvvm.fmax.f16x2(<2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fmax.ftz.f16x2(<2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fmax.nan.f16x2(<2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fmax.ftz.nan.f16x2(<2 x half>, <2 x half>)
declare i16 @llvm.nvvm.fmax.bf16(i16, i16)
declare i16 @llvm.nvvm.fmax.nan.bf16(i16, i16)
declare i32 @llvm.nvvm.fmax.bf16x2(i32, i32)
declare i32 @llvm.nvvm.fmax.nan.bf16x2(i32, i32)

declare half @llvm.nvvm.fma.rn.relu.f16(half, half, half)
declare half @llvm.nvvm.fma.rn.ftz.relu.f16(half, half, half)
declare <2 x half> @llvm.nvvm.fma.rn.relu.f16x2(<2 x half>, <2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fma.rn.ftz.relu.f16x2(<2 x half>, <2 x half>, <2 x half>)
declare i16 @llvm.nvvm.fma.rn.bf16(i16, i16, i16)
declare i16 @llvm.nvvm.fma.rn.relu.bf16(i16, i16, i16)
declare i32 @llvm.nvvm.fma.rn.bf16x2(i32, i32, i32)
declare i32 @llvm.nvvm.fma.rn.relu.bf16x2(i32, i32, i32)

; CHECK-LABEL: abs_bf16
define i16 @abs_bf16(i16 %0) {
  ; CHECK-NOT: call
  ; CHECK: abs.bf16
  %res = call i16 @llvm.nvvm.abs.bf16(i16 %0);
  ret i16 %res
}

; CHECK-LABEL: abs_bf16x2
define i32 @abs_bf16x2(i32 %0) {
  ; CHECK-NOT: call
  ; CHECK: abs.bf16x2
  %res = call i32 @llvm.nvvm.abs.bf16x2(i32 %0);
  ret i32 %res
}

; CHECK-LABEL: neg_bf16
define i16 @neg_bf16(i16 %0) {
  ; CHECK-NOT: call
  ; CHECK: neg.bf16
  %res = call i16 @llvm.nvvm.neg.bf16(i16 %0);
  ret i16 %res
}

; CHECK-LABEL: neg_bf16x2
define i32 @neg_bf16x2(i32 %0) {
  ; CHECK-NOT: call
  ; CHECK: neg.bf16x2
  %res = call i32 @llvm.nvvm.neg.bf16x2(i32 %0);
  ret i32 %res
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
define i16 @fmin_bf16(i16 %0, i16 %1) {
  ; CHECK-NOT: call
  ; CHECK: min.bf16
  %res = call i16 @llvm.nvvm.fmin.bf16(i16 %0, i16 %1)
  ret i16 %res
}

; CHECK-LABEL: fmin_nan_bf16
define i16 @fmin_nan_bf16(i16 %0, i16 %1) {
  ; CHECK-NOT: call
  ; CHECK: min.NaN.bf16
  %res = call i16 @llvm.nvvm.fmin.nan.bf16(i16 %0, i16 %1)
  ret i16 %res
}

; CHECK-LABEL: fmin_bf16x2
define i32 @fmin_bf16x2(i32 %0, i32 %1) {
  ; CHECK-NOT: call
  ; CHECK: min.bf16x2
  %res = call i32 @llvm.nvvm.fmin.bf16x2(i32 %0, i32 %1)
  ret i32 %res
}

; CHECK-LABEL: fmin_nan_bf16x2
define i32 @fmin_nan_bf16x2(i32 %0, i32 %1) {
  ; CHECK-NOT: call
  ; CHECK: min.NaN.bf16x2
  %res = call i32 @llvm.nvvm.fmin.nan.bf16x2(i32 %0, i32 %1)
  ret i32 %res
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
define i16 @fmax_bf16(i16 %0, i16 %1) {
  ; CHECK-NOT: call
  ; CHECK: max.bf16
  %res = call i16 @llvm.nvvm.fmax.bf16(i16 %0, i16 %1)
  ret i16 %res
}

; CHECK-LABEL: fmax_nan_bf16
define i16 @fmax_nan_bf16(i16 %0, i16 %1) {
  ; CHECK-NOT: call
  ; CHECK: max.NaN.bf16
  %res = call i16 @llvm.nvvm.fmax.nan.bf16(i16 %0, i16 %1)
  ret i16 %res
}

; CHECK-LABEL: fmax_bf16x2
define i32 @fmax_bf16x2(i32 %0, i32 %1) {
  ; CHECK-NOT: call
  ; CHECK: max.bf16x2
  %res = call i32 @llvm.nvvm.fmax.bf16x2(i32 %0, i32 %1)
  ret i32 %res
}

; CHECK-LABEL: fmax_nan_bf16x2
define i32 @fmax_nan_bf16x2(i32 %0, i32 %1) {
  ; CHECK-NOT: call
  ; CHECK: max.NaN.bf16x2
  %res = call i32 @llvm.nvvm.fmax.nan.bf16x2(i32 %0, i32 %1)
  ret i32 %res
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
define i16 @fma_rn_bf16(i16 %0, i16 %1, i16 %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.bf16
  %res = call i16 @llvm.nvvm.fma.rn.bf16(i16 %0, i16 %1, i16 %2)
  ret i16 %res
}

; CHECK-LABEL: fma_rn_relu_bf16
define i16 @fma_rn_relu_bf16(i16 %0, i16 %1, i16 %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.relu.bf16
  %res = call i16 @llvm.nvvm.fma.rn.relu.bf16(i16 %0, i16 %1, i16 %2)
  ret i16 %res
}

; CHECK-LABEL: fma_rn_bf16x2
define i32 @fma_rn_bf16x2(i32 %0, i32 %1, i32 %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.bf16x2
  %res = call i32 @llvm.nvvm.fma.rn.bf16x2(i32 %0, i32 %1, i32 %2)
  ret i32 %res
}

; CHECK-LABEL: fma_rn_relu_bf16x2
define i32 @fma_rn_relu_bf16x2(i32 %0, i32 %1, i32 %2) {
  ; CHECK-NOT: call
  ; CHECK: fma.rn.relu.bf16x2
  %res = call i32 @llvm.nvvm.fma.rn.relu.bf16x2(i32 %0, i32 %1, i32 %2)
  ret i32 %res
}
