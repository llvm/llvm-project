// RUN: mlir-opt %s -convert-nvgpu-to-nvvm | FileCheck %s

// Large-vector smoke tests for nvgpu.truncf.

// CHECK-LABEL: @cvt_large_f32_to_f16(
// CHECK-SAME: %[[IN:.+]]: vector<400xf32>
func.func @cvt_large_f32_to_f16(%in : vector<400xf32>) -> vector<400xf16> {
  // CHECK: llvm.bitcast %[[IN]] : vector<400xf32> to vector<400xi32>
  // CHECK: llvm.mlir.undef : vector<200xi32>
  // CHECK-COUNT-200: nvvm.convert.f32x2.to.f16x2
  // CHECK-NOT: nvvm.convert.f32x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<200xi32> to vector<400xf16>
  %out = nvgpu.truncf %in : vector<400xf32> to vector<400xf16>
  return %out : vector<400xf16>
}

// CHECK-LABEL: @cvt_large_f32_to_bf16(
func.func @cvt_large_f32_to_bf16(%in : vector<400xf32>) -> vector<400xbf16> {
  // CHECK: llvm.bitcast %{{.*}} : vector<400xf32> to vector<400xi32>
  // CHECK-COUNT-200: nvvm.convert.f32x2.to.bf16x2
  // CHECK-NOT: nvvm.convert.f32x2.to.bf16x2
  // CHECK: llvm.bitcast {{.*}} : vector<200xi32> to vector<400xbf16>
  %out = nvgpu.truncf %in : vector<400xf32> to vector<400xbf16>
  return %out : vector<400xbf16>
}

// CHECK-LABEL: @cvt_large_f32_to_f8(
func.func @cvt_large_f32_to_f8(%in : vector<400xf32>) -> vector<400xf8E4M3FN> {
  // CHECK: llvm.bitcast %{{.*}} : vector<400xf32> to vector<400xi32>
  // CHECK: llvm.mlir.undef : vector<100xi32>
  // CHECK-COUNT-200: nvvm.convert.f32x2.to.f8x2
  // CHECK-NOT: nvvm.convert.f32x2.to.f8x2
  // CHECK: llvm.bitcast {{.*}} : vector<100xi32> to vector<400xi8>
  %out = nvgpu.truncf %in : vector<400xf32> to vector<400xf8E4M3FN>
  return %out : vector<400xf8E4M3FN>
}

// CHECK-LABEL: @cvt_large_f32_to_f6(
func.func @cvt_large_f32_to_f6(%in : vector<400xf32>) -> vector<400xf6E2M3FN> {
  // CHECK: llvm.bitcast %{{.*}} : vector<400xf32> to vector<400xi32>
  // CHECK-COUNT-200: nvvm.convert.f32x2.to.f6x2
  // CHECK-NOT: nvvm.convert.f32x2.to.f6x2
  // CHECK: llvm.bitcast {{.*}} : vector<100xi32> to vector<400xi8>
  // CHECK: llvm.trunc {{.*}} : vector<400xi8> to vector<400xi6>
  %out = nvgpu.truncf %in : vector<400xf32> to vector<400xf6E2M3FN>
  return %out : vector<400xf6E2M3FN>
}

// CHECK-LABEL: @cvt_large_f32_to_f4(
func.func @cvt_large_f32_to_f4(%in : vector<400xf32>) -> vector<400xf4E2M1FN> {
  // CHECK: llvm.bitcast %{{.*}} : vector<400xf32> to vector<400xi32>
  // CHECK: llvm.mlir.undef : vector<50xi32>
  // CHECK-COUNT-200: nvvm.convert.f32x2.to.f4x2
  // CHECK-NOT: nvvm.convert.f32x2.to.f4x2
  %out = nvgpu.truncf %in : vector<400xf32> to vector<400xf4E2M1FN>
  return %out : vector<400xf4E2M1FN>
}

// CHECK-LABEL: @cvt_large_f16_to_f8(
func.func @cvt_large_f16_to_f8(%in : vector<400xf16>) -> vector<400xf8E4M3FN> {
  // CHECK: llvm.bitcast %{{.*}} : vector<400xf16> to vector<200xi32>
  // CHECK-COUNT-200: nvvm.convert.f16x2.to.f8x2
  // CHECK-NOT: nvvm.convert.f16x2.to.f8x2
  %out = nvgpu.truncf %in : vector<400xf16> to vector<400xf8E4M3FN>
  return %out : vector<400xf8E4M3FN>
}

// CHECK-LABEL: @cvt_large_bf16_to_f6(
func.func @cvt_large_bf16_to_f6(%in : vector<400xbf16>) -> vector<400xf6E3M2FN> {
  // CHECK: llvm.bitcast %{{.*}} : vector<400xbf16> to vector<200xi32>
  // CHECK-COUNT-200: nvvm.convert.bf16x2.to.f6x2
  // CHECK-NOT: nvvm.convert.bf16x2.to.f6x2
  // CHECK: llvm.trunc {{.*}} : vector<400xi8> to vector<400xi6>
  %out = nvgpu.truncf %in : vector<400xbf16> to vector<400xf6E3M2FN>
  return %out : vector<400xf6E3M2FN>
}
