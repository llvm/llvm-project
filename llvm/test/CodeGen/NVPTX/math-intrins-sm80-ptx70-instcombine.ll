; RUN: opt < %s -passes=instcombine -S -mtriple=nvptx-nvidia-cuda -march=nvptx64 \
; RUN:    -mcpu=sm_80 -mattr=+ptx70 | \
; RUN: FileCheck %s

declare half @llvm.nvvm.fmin.f16(half, half)
declare half @llvm.nvvm.fmin.ftz.f16(half, half)
declare <2 x half> @llvm.nvvm.fmin.f16x2(<2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fmin.ftz.f16x2(<2 x half>, <2 x half>)
declare float @llvm.nvvm.fmin.nan.f(float, float)
declare float @llvm.nvvm.fmin.ftz.nan.f(float, float)
declare half @llvm.nvvm.fmin.nan.f16(half, half)
declare half @llvm.nvvm.fmin.ftz.nan.f16(half, half)
declare <2 x half> @llvm.nvvm.fmin.nan.f16x2(<2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fmin.ftz.nan.f16x2(<2 x half>, <2 x half>)

declare half @llvm.nvvm.fmax.f16(half, half)
declare half @llvm.nvvm.fmax.ftz.f16(half, half)
declare <2 x half> @llvm.nvvm.fmax.f16x2(<2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fmax.ftz.f16x2(<2 x half>, <2 x half>)
declare float @llvm.nvvm.fmax.nan.f(float, float)
declare float @llvm.nvvm.fmax.ftz.nan.f(float, float)
declare half @llvm.nvvm.fmax.nan.f16(half, half)
declare half @llvm.nvvm.fmax.ftz.nan.f16(half, half)
declare <2 x half> @llvm.nvvm.fmax.nan.f16x2(<2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fmax.ftz.nan.f16x2(<2 x half>, <2 x half>)

; f16 and f16x2 fma are available since ptx 4.2 and sm_53.
declare half @llvm.nvvm.fma.rn.f16(half, half, half)
declare half @llvm.nvvm.fma.rn.ftz.f16(half, half, half)
declare <2 x half> @llvm.nvvm.fma.rn.f16x2(<2 x half>, <2 x half>, <2 x half>)
declare <2 x half> @llvm.nvvm.fma.rn.ftz.f16x2(<2 x half>, <2 x half>, <2 x half>)

; CHECK-LABEL: fmin_f16
define half @fmin_f16(half %0, half %1) {
  ; CHECK-NOT: @llvm.nvvm.fmin.f16
  ; CHECK: @llvm.minnum.f16
  %res = call half @llvm.nvvm.fmin.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmin_ftz_f16
define half @fmin_ftz_f16(half %0, half %1) #0 {
  ; CHECK-NOT: @llvm.nvvm.fmin.ftz.f16
  ; CHECK: @llvm.minnum.f16
  %res = call half @llvm.nvvm.fmin.ftz.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmin_ftz_f16_no_attr
define half @fmin_ftz_f16_no_attr(half %0, half %1) {
  ; CHECK-NOT: @llvm.minnum.f16
  ; CHECK: @llvm.nvvm.fmin.ftz.f16
  %res = call half @llvm.nvvm.fmin.ftz.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmin_f16x2
define <2 x half> @fmin_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: @llvm.nvvm.fmin.f16x2
  ; CHECK: @llvm.minnum.v2f16
  %res = call <2 x half> @llvm.nvvm.fmin.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmin_ftz_f16x2
define <2 x half> @fmin_ftz_f16x2(<2 x half> %0, <2 x half> %1) #0 {
  ; CHECK-NOT: @llvm.nvvm.fmin.ftz.f16x2
  ; CHECK: @llvm.minnum.v2f16
  %res = call <2 x half> @llvm.nvvm.fmin.ftz.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmin_ftz_f16x2_no_attr
define <2 x half> @fmin_ftz_f16x2_no_attr(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: @llvm.minnum.v2f16
  ; CHECK: @llvm.nvvm.fmin.ftz.f16x2
  %res = call <2 x half> @llvm.nvvm.fmin.ftz.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmin_nan_f
define float @fmin_nan_f(float %0, float %1) {
  ; CHECK-NOT: @llvm.nvvm.fmin.nan.f
  ; CHECK: @llvm.minimum.f32
  %res = call float @llvm.nvvm.fmin.nan.f(float %0, float %1)
  ret float %res
}

; CHECK-LABEL: fmin_ftz_nan_f
define float @fmin_ftz_nan_f(float %0, float %1) #1 {
  ; CHECK-NOT: @llvm.nvvm.fmin.ftz.nan.f
  ; CHECK: @llvm.minimum.f32
  %res = call float @llvm.nvvm.fmin.ftz.nan.f(float %0, float %1)
  ret float %res
}

; CHECK-LABEL: fmin_ftz_nan_f_no_attr
define float @fmin_ftz_nan_f_no_attr(float %0, float %1) {
  ; CHECK: @llvm.nvvm.fmin.ftz.nan.f
  ; CHECK-NOT: @llvm.minimum.f32
  %res = call float @llvm.nvvm.fmin.ftz.nan.f(float %0, float %1)
  ret float %res
}

; CHECK-LABEL: fmin_nan_f16
define half @fmin_nan_f16(half %0, half %1) {
  ; CHECK-NOT: @llvm.nvvm.fmin.nan.f16
  ; CHECK: @llvm.minimum.f16
  %res = call half @llvm.nvvm.fmin.nan.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmin_ftz_nan_f16
define half @fmin_ftz_nan_f16(half %0, half %1) #0 {
  ; CHECK-NOT: @llvm.nvvm.fmin.ftz.nan.f16
  ; CHECK: @llvm.minimum.f16
  %res = call half @llvm.nvvm.fmin.ftz.nan.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmin_ftz_nan_f16_no_attr
define half @fmin_ftz_nan_f16_no_attr(half %0, half %1) {
  ; CHECK: @llvm.nvvm.fmin.ftz.nan.f16
  ; CHECK-NOT: @llvm.minimum.f16
  %res = call half @llvm.nvvm.fmin.ftz.nan.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmin_nan_f16x2
define <2 x half> @fmin_nan_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: @llvm.nvvm.fmin.nan.f16x2
  ; CHECK: @llvm.minimum.v2f16
  %res = call <2 x half> @llvm.nvvm.fmin.nan.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmin_ftz_nan_f16x2
define <2 x half> @fmin_ftz_nan_f16x2(<2 x half> %0, <2 x half> %1) #0 {
  ; CHECK-NOT: @llvm.nvvm.fmin.ftz.nan.f16x2
  ; CHECK: @llvm.minimum.v2f16
  %res = call <2 x half> @llvm.nvvm.fmin.ftz.nan.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmin_ftz_nan_f16x2_no_attr
define <2 x half> @fmin_ftz_nan_f16x2_no_attr(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: @llvm.minimum.v2f16
  ; CHECK: @llvm.nvvm.fmin.ftz.nan.f16x2
  %res = call <2 x half> @llvm.nvvm.fmin.ftz.nan.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmax_f16
define half @fmax_f16(half %0, half %1) {
  ; CHECK-NOT: @llvm.nvvm.fmax.f16
  ; CHECK: @llvm.maxnum.f16
  %res = call half @llvm.nvvm.fmax.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmax_ftz_f16
define half @fmax_ftz_f16(half %0, half %1) #0 {
  ; CHECK-NOT: @llvm.nvvm.fmax.ftz.f16
  ; CHECK: @llvm.maxnum.f16
  %res = call half @llvm.nvvm.fmax.ftz.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmax_ftz_f16_no_attr
define half @fmax_ftz_f16_no_attr(half %0, half %1) {
  ; CHECK-NOT: @llvm.maxnum.f16
  ; CHECK: @llvm.nvvm.fmax.ftz.f16
  %res = call half @llvm.nvvm.fmax.ftz.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmax_f16x2
define <2 x half> @fmax_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: @llvm.nvvm.fmax.f16x2
  ; CHECK: @llvm.maxnum.v2f16
  %res = call <2 x half> @llvm.nvvm.fmax.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmax_ftz_f16x2
define <2 x half> @fmax_ftz_f16x2(<2 x half> %0, <2 x half> %1) #0 {
  ; CHECK-NOT: @llvm.nvvm.fmax.ftz.f16x2
  ; CHECK: @llvm.maxnum.v2f16
  %res = call <2 x half> @llvm.nvvm.fmax.ftz.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmax_ftz_f16x2_no_attr
define <2 x half> @fmax_ftz_f16x2_no_attr(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: @llvm.maxnum.v2f16
  ; CHECK: @llvm.nvvm.fmax.ftz.f16x2
  %res = call <2 x half> @llvm.nvvm.fmax.ftz.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmax_nan_f
define float @fmax_nan_f(float %0, float %1) {
  ; CHECK-NOT: @llvm.nvvm.fmax.nan.f
  ; CHECK: @llvm.maximum.f32
  %res = call float @llvm.nvvm.fmax.nan.f(float %0, float %1)
  ret float %res
}

; CHECK-LABEL: fmax_ftz_nan_f
define float @fmax_ftz_nan_f(float %0, float %1) #1 {
  ; CHECK-NOT: @llvm.nvvm.fmax.ftz.nan.f
  ; CHECK: @llvm.maximum.f32
  %res = call float @llvm.nvvm.fmax.ftz.nan.f(float %0, float %1)
  ret float %res
}

; CHECK-LABEL: fmax_ftz_nan_f_no_attr
define float @fmax_ftz_nan_f_no_attr(float %0, float %1) {
  ; CHECK: @llvm.nvvm.fmax.ftz.nan.f
  ; CHECK-NOT: @llvm.maximum.f32
  %res = call float @llvm.nvvm.fmax.ftz.nan.f(float %0, float %1)
  ret float %res
}

; CHECK-LABEL: fmax_nan_f16
define half @fmax_nan_f16(half %0, half %1) {
  ; CHECK-NOT: @llvm.nvvm.fmax.nan.f16
  ; CHECK: @llvm.maximum.f16
  %res = call half @llvm.nvvm.fmax.nan.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmax_ftz_nan_f16
define half @fmax_ftz_nan_f16(half %0, half %1) #0 {
  ; CHECK-NOT: @llvm.nvvm.fmax.ftz.nan.f16
  ; CHECK: @llvm.maximum.f16
  %res = call half @llvm.nvvm.fmax.ftz.nan.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmax_ftz_nan_f16_no_attr
define half @fmax_ftz_nan_f16_no_attr(half %0, half %1) {
  ; CHECK: @llvm.nvvm.fmax.ftz.nan.f16
  ; CHECK-NOT: @llvm.maximum.f16
  %res = call half @llvm.nvvm.fmax.ftz.nan.f16(half %0, half %1)
  ret half %res
}

; CHECK-LABEL: fmax_nan_f16x2
define <2 x half> @fmax_nan_f16x2(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: @llvm.nvvm.fmax.nan.f16x2
  ; CHECK: @llvm.maximum.v2f16
  %res = call <2 x half> @llvm.nvvm.fmax.nan.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmax_ftz_nan_f16x2
define <2 x half> @fmax_ftz_nan_f16x2(<2 x half> %0, <2 x half> %1) #0 {
  ; CHECK-NOT: @llvm.nvvm.fmax.ftz.nan.f16x2
  ; CHECK: @llvm.maximum.v2f16
  %res = call <2 x half> @llvm.nvvm.fmax.ftz.nan.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fmax_ftz_nan_f16x2_no_attr
define <2 x half> @fmax_ftz_nan_f16x2_no_attr(<2 x half> %0, <2 x half> %1) {
  ; CHECK-NOT: @llvm.maximum.v2f16
  ; CHECK: @llvm.nvvm.fmax.ftz.nan.f16x2
  %res = call <2 x half> @llvm.nvvm.fmax.ftz.nan.f16x2(<2 x half> %0, <2 x half> %1)
  ret <2 x half> %res
}

; CHECK-LABEL: fma_rn_f16
define half @fma_rn_f16(half %0, half %1, half %2) {
  ; CHECK-NOT: @llvm.nvvm.fma.rn.f16
  ; CHECK: @llvm.fma.f16
  %res = call half @llvm.nvvm.fma.rn.f16(half %0, half %1, half %2)
  ret half %res
}

; CHECK-LABEL: fma_rn_ftz_f16_no_attr
define half @fma_rn_ftz_f16_no_attr(half %0, half %1, half %2) {
  ; CHECK-NOT: @llvm.fma.f16
  ; CHECK: @llvm.nvvm.fma.rn.ftz.f16
  %res = call half @llvm.nvvm.fma.rn.ftz.f16(half %0, half %1, half %2)
  ret half %res
}

; CHECK-LABEL: fma_rn_ftz_f16
define half @fma_rn_ftz_f16(half %0, half %1, half %2) #0 {
  ; CHECK-NOT: @llvm.nvvm.fma.rn.ftz.f16
  ; CHECK: @llvm.fma.f16
  %res = call half @llvm.nvvm.fma.rn.ftz.f16(half %0, half %1, half %2)
  ret half %res
}

; CHECK-LABEL: fma_rn_f16x2
define <2 x half> @fma_rn_f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2) {
  ; CHECK-NOT: @llvm.nvvm.fma.rn.f16x2
  ; CHECK: @llvm.fma.v2f16
  %res = call <2 x half> @llvm.nvvm.fma.rn.f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2)
  ret <2 x half> %res
}

; CHECK-LABEL: fma_rn_ftz_f16x2
define <2 x half> @fma_rn_ftz_f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2) #0 {
  ; CHECK-NOT: @llvm.nvvm.fma.rn.ftz.f16x2
  ; CHECK: @llvm.fma.v2f16
  %res = call <2 x half> @llvm.nvvm.fma.rn.ftz.f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2)
  ret <2 x half> %res
}

; CHECK-LABEL: fma_rn_ftz_f16x2_no_attr
define <2 x half> @fma_rn_ftz_f16x2_no_attr(<2 x half> %0, <2 x half> %1, <2 x half> %2) {
  ; CHECK-NOT: @llvm.fma.v2f16
  ; CHECK: @llvm.nvvm.fma.rn.ftz.f16x2
  %res = call <2 x half> @llvm.nvvm.fma.rn.ftz.f16x2(<2 x half> %0, <2 x half> %1, <2 x half> %2)
  ret <2 x half> %res
}

attributes #0 = { "denormal-fp-math"="preserve-sign" }
attributes #1 = { "denormal-fp-math-f32"="preserve-sign" }
