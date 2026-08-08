; RUN: opt -S -dxil-intrinsic-expansion -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.0-pixel"

; Scalar float store: one StoreOutput call, no residual intrinsic.
; CHECK-LABEL: define void @store_scalar_f32
define void @store_scalar_f32(float %val) {
  ; CHECK: call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 1, i8 2, float %val)
  ; CHECK-NOT: llvm.dx.store.output
  call void @llvm.dx.store.output.f32(i32 99, i32 0, i32 1, i8 2, float %val)
  ret void
}

; Vector float4 store: four per-component StoreOutput calls, col indices 0..3.
; CHECK-LABEL: define void @store_v4f32
define void @store_v4f32(<4 x float> %val) {
  ; CHECK: [[E0:%.*]] = extractelement <4 x float> %val, i32 0
  ; CHECK-NEXT: call void @dx.op.storeOutput.f32(i32 5, i32 1, i32 0, i8 0, float [[E0]])
  ; CHECK-NEXT: [[E1:%.*]] = extractelement <4 x float> %val, i32 1
  ; CHECK-NEXT: call void @dx.op.storeOutput.f32(i32 5, i32 1, i32 0, i8 1, float [[E1]])
  ; CHECK-NEXT: [[E2:%.*]] = extractelement <4 x float> %val, i32 2
  ; CHECK-NEXT: call void @dx.op.storeOutput.f32(i32 5, i32 1, i32 0, i8 2, float [[E2]])
  ; CHECK-NEXT: [[E3:%.*]] = extractelement <4 x float> %val, i32 3
  ; CHECK-NEXT: call void @dx.op.storeOutput.f32(i32 5, i32 1, i32 0, i8 3, float [[E3]])
  ; CHECK-NOT: llvm.dx.store.output
  call void @llvm.dx.store.output.v4f32(i32 99, i32 1, i32 0, i8 0, <4 x float> %val)
  ret void
}

; Vector float2 store with non-zero start column: col indices must be 2 and 3.
; CHECK-LABEL: define void @store_v2f32_col2
define void @store_v2f32_col2(<2 x float> %val) {
  ; CHECK: [[E0:%.*]] = extractelement <2 x float> %val, i32 0
  ; CHECK-NEXT: call void @dx.op.storeOutput.f32(i32 5, i32 2, i32 0, i8 2, float [[E0]])
  ; CHECK-NEXT: [[E1:%.*]] = extractelement <2 x float> %val, i32 1
  ; CHECK-NEXT: call void @dx.op.storeOutput.f32(i32 5, i32 2, i32 0, i8 3, float [[E1]])
  ; CHECK-NOT: llvm.dx.store.output
  call void @llvm.dx.store.output.v2f32(i32 99, i32 2, i32 0, i8 2, <2 x float> %val)
  ret void
}

; Scalar int store: one StoreOutput call, no residual intrinsic.
; CHECK-LABEL: define void @store_scalar_i32
define void @store_scalar_i32(i32 %val) {
  ; CHECK: call void @dx.op.storeOutput.i32(i32 5, i32 2, i32 0, i8 0, i32 %val)
  ; CHECK-NOT: llvm.dx.store.output
  call void @llvm.dx.store.output.i32(i32 99, i32 2, i32 0, i8 0, i32 %val)
  ret void
}
