; RUN: opt -S -dxil-intrinsic-expansion -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.0-pixel"

; Scalar float load: one LoadInput call, result forwarded directly.
; CHECK-LABEL: define float @load_scalar_f32
define float @load_scalar_f32() {
  ; CHECK: [[V:%.*]] = call float @dx.op.loadInput.f32(i32 4, i32 0, i32 1, i8 2, i32 0)
  ; CHECK-NEXT: ret float [[V]]
  ; CHECK-NOT: llvm.dx.load.input
  %v = call float @llvm.dx.load.input.f32(i32 99, i32 0, i32 1, i8 2, i32 0)
  ret float %v
}

; Vector float4 load: four per-component LoadInput calls reassembled into a vector.
; CHECK-LABEL: define <4 x float> @load_v4f32
define <4 x float> @load_v4f32() {
  ; CHECK: [[S0:%.*]] = call float @dx.op.loadInput.f32(i32 4, i32 1, i32 0, i8 0, i32 0)
  ; CHECK-NEXT: insertelement <4 x float> {{.*}}, float [[S0]], i32 0
  ; CHECK: [[S1:%.*]] = call float @dx.op.loadInput.f32(i32 4, i32 1, i32 0, i8 1, i32 0)
  ; CHECK-NEXT: insertelement {{.*}}, float [[S1]], i32 1
  ; CHECK: [[S2:%.*]] = call float @dx.op.loadInput.f32(i32 4, i32 1, i32 0, i8 2, i32 0)
  ; CHECK-NEXT: insertelement {{.*}}, float [[S2]], i32 2
  ; CHECK: [[S3:%.*]] = call float @dx.op.loadInput.f32(i32 4, i32 1, i32 0, i8 3, i32 0)
  ; CHECK-NEXT: insertelement {{.*}}, float [[S3]], i32 3
  ; CHECK-NOT: llvm.dx.load.input
  %v = call <4 x float> @llvm.dx.load.input.v4f32(i32 99, i32 1, i32 0, i8 0, i32 0)
  ret <4 x float> %v
}

; Vector float2 load with non-zero start column: col indices must be 2 and 3.
; CHECK-LABEL: define <2 x float> @load_v2f32_col2
define <2 x float> @load_v2f32_col2() {
  ; CHECK: [[S0:%.*]] = call float @dx.op.loadInput.f32(i32 4, i32 2, i32 0, i8 2, i32 0)
  ; CHECK-NEXT: insertelement <2 x float> {{.*}}, float [[S0]], i32 0
  ; CHECK: [[S1:%.*]] = call float @dx.op.loadInput.f32(i32 4, i32 2, i32 0, i8 3, i32 0)
  ; CHECK-NEXT: insertelement {{.*}}, float [[S1]], i32 1
  ; CHECK-NOT: llvm.dx.load.input
  %v = call <2 x float> @llvm.dx.load.input.v2f32(i32 99, i32 2, i32 0, i8 2, i32 0)
  ret <2 x float> %v
}

; Scalar int load: one LoadInput call, result forwarded directly.
; CHECK-LABEL: define i32 @load_scalar_i32
define i32 @load_scalar_i32() {
  ; CHECK: [[V:%.*]] = call i32 @dx.op.loadInput.i32(i32 4, i32 2, i32 0, i8 0, i32 0)
  ; CHECK-NEXT: ret i32 [[V]]
  ; CHECK-NOT: llvm.dx.load.input
  %v = call i32 @llvm.dx.load.input.i32(i32 99, i32 2, i32 0, i8 0, i32 0)
  ret i32 %v
}
