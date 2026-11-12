; RUN: opt -S < %s | FileCheck %s
; RUN: llvm-dis < %s.bc | FileCheck %s


define float @fadd_v2(<4 x float> %in, float %acc) {
; CHECK-LABEL: @fadd_v2
; CHECK: %res = call float @llvm.vector.reduce.fadd.v4f32(float %acc, <4 x float> %in)
  %res = call float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float %acc, <4 x float> %in)
  ret float %res
}

define float @fadd_v2_fast(<4 x float> %in, float %acc) {
; CHECK-LABEL: @fadd_v2_fast
; CHECK: %res = call fast float @llvm.vector.reduce.fadd.v4f32(float %acc, <4 x float> %in)
  %res = call fast float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float %acc, <4 x float> %in)
  ret float %res
}

define float @fmul_v2(<4 x float> %in, float %acc) {
; CHECK-LABEL: @fmul_v2
; CHECK: %res = call float @llvm.vector.reduce.fmul.v4f32(float %acc, <4 x float> %in)
  %res = call float @llvm.experimental.vector.reduce.v2.fmul.f32.v4f32(float %acc, <4 x float> %in)
  ret float %res
}

define float @fmul_v2_fast(<4 x float> %in, float %acc) {
; CHECK-LABEL: @fmul_v2_fast
; CHECK: %res = call fast  float @llvm.vector.reduce.fmul.v4f32(float %acc, <4 x float> %in)
  %res = call fast float @llvm.experimental.vector.reduce.v2.fmul.f32.v4f32(float %acc, <4 x float> %in)
  ret float %res
}

define float @fmin(<4 x float> %in) {
; CHECK-LABEL: @fmin
; CHECK: %res = call float @llvm.vector.reduce.fmin.v4f32(<4 x float> %in)
  %res = call float @llvm.experimental.vector.reduce.fmin.v4f32(<4 x float> %in)
  ret float %res
}

define float @fmax(<4 x float> %in) {
; CHECK-LABEL: @fmax
; CHECK: %res = call float @llvm.vector.reduce.fmax.v4f32(<4 x float> %in)
  %res = call float @llvm.experimental.vector.reduce.fmax.v4f32(<4 x float> %in)
  ret float %res
}

define i32 @and(<4 x i32> %in) {
; CHECK-LABEL: @and
; CHECK: %res = call i32 @llvm.vector.reduce.and.v4i32(<4 x i32> %in)
  %res = call i32 @llvm.experimental.vector.reduce.and.v4i32(<4 x i32> %in)
  ret i32 %res
}

define i32 @or(<4 x i32> %in) {
; CHECK-LABEL: @or
; CHECK: %res = call i32 @llvm.vector.reduce.or.v4i32(<4 x i32> %in)
  %res = call i32 @llvm.experimental.vector.reduce.or.v4i32(<4 x i32> %in)
  ret i32 %res
}

define i32 @xor(<4 x i32> %in) {
; CHECK-LABEL: @xor
; CHECK: %res = call i32 @llvm.vector.reduce.xor.v4i32(<4 x i32> %in)
  %res = call i32 @llvm.experimental.vector.reduce.xor.v4i32(<4 x i32> %in)
  ret i32 %res
}

define i32 @smin(<4 x i32> %in) {
; CHECK-LABEL: @smin
; CHECK: %res = call i32 @llvm.vector.reduce.smin.v4i32(<4 x i32> %in)
  %res = call i32 @llvm.experimental.vector.reduce.smin.v4i32(<4 x i32> %in)
  ret i32 %res
}

define i32 @smax(<4 x i32> %in) {
; CHECK-LABEL: @smax
; CHECK: %res = call i32 @llvm.vector.reduce.smax.v4i32(<4 x i32> %in)
  %res = call i32 @llvm.experimental.vector.reduce.smax.v4i32(<4 x i32> %in)
  ret i32 %res
}

define i32 @umin(<4 x i32> %in) {
; CHECK-LABEL: @umin
; CHECK: %res = call i32 @llvm.vector.reduce.umin.v4i32(<4 x i32> %in)
  %res = call i32 @llvm.experimental.vector.reduce.umin.v4i32(<4 x i32> %in)
  ret i32 %res
}

define i32 @umax(<4 x i32> %in) {
; CHECK-LABEL: @umax
; CHECK: %res = call i32 @llvm.vector.reduce.umax.v4i32(<4 x i32> %in)
  %res = call i32 @llvm.experimental.vector.reduce.umax.v4i32(<4 x i32> %in)
  ret i32 %res
}


declare float @llvm.experimental.vector.reduce.v2.fadd.f32.v4f32(float, <4 x float>)
declare float @llvm.experimental.vector.reduce.v2.fmul.f32.v4f32(float, <4 x float>)

declare float @llvm.experimental.vector.reduce.fmin.v4f32(<4 x float>)
; CHECK: declare float @llvm.vector.reduce.fmin.v4f32(<4 x float>)

declare float @llvm.experimental.vector.reduce.fmax.v4f32(<4 x float>)
; CHECK: declare float @llvm.vector.reduce.fmax.v4f32(<4 x float>)

declare i32 @llvm.experimental.vector.reduce.and.v4i32(<4 x i32>)
; CHECK: declare i32 @llvm.vector.reduce.and.v4i32(<4 x i32>)

declare i32 @llvm.experimental.vector.reduce.or.v4i32(<4 x i32>)
; CHECK: declare i32 @llvm.vector.reduce.or.v4i32(<4 x i32>)

declare i32 @llvm.experimental.vector.reduce.xor.v4i32(<4 x i32>)
; CHECK: declare i32 @llvm.vector.reduce.xor.v4i32(<4 x i32>)

declare i32 @llvm.experimental.vector.reduce.smin.v4i32(<4 x i32>)
; CHECK: declare i32 @llvm.vector.reduce.smin.v4i32(<4 x i32>)

declare i32 @llvm.experimental.vector.reduce.smax.v4i32(<4 x i32>)
; CHECK: declare i32 @llvm.vector.reduce.smax.v4i32(<4 x i32>)

declare i32 @llvm.experimental.vector.reduce.umin.v4i32(<4 x i32>)
; CHECK: declare i32 @llvm.vector.reduce.umin.v4i32(<4 x i32>)

declare i32 @llvm.experimental.vector.reduce.umax.v4i32(<4 x i32>)
; CHECK: declare i32 @llvm.vector.reduce.umax.v4i32(<4 x i32>)





