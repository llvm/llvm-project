// RUN: mlir-opt %s -convert-nvgpu-to-nvvm | FileCheck %s

// Large-vector smoke tests for nvgpu.convert.fpext

// CHECK-LABEL: @cvt_large_f8_to_f16(
// CHECK-SAME: %[[IN:.+]]: vector<400xf8E4M3FN>
func.func @cvt_large_f8_to_f16(%in : vector<400xf8E4M3FN>) -> vector<400xf16> {
  // CHECK: builtin.unrealized_conversion_cast %[[IN]] : vector<400xf8E4M3FN> to vector<400xi8>
  // CHECK: llvm.bitcast {{.*}} : vector<400xi8> to vector<100xi32>
  // CHECK-COUNT-200: nvvm.convert.f8x2.to.f16x2
  // CHECK-NOT: nvvm.convert.f8x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<200xi32> to vector<400xf16>
  %out = nvgpu.convert.fpext %in : vector<400xf8E4M3FN> to vector<400xf16>
  return %out : vector<400xf16>
}

// CHECK-LABEL: @cvt_large_e8m0_to_bf16(
func.func @cvt_large_e8m0_to_bf16(%in : vector<400xf8E8M0FNU>) -> vector<400xbf16> {
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : vector<400xf8E8M0FNU> to vector<400xi8>
  // CHECK-COUNT-200: nvvm.convert.f8x2.to.bf16x2
  // CHECK-NOT: nvvm.convert.f8x2.to.bf16x2
  // CHECK: llvm.bitcast {{.*}} : vector<200xi32> to vector<400xbf16>
  %out = nvgpu.convert.fpext %in : vector<400xf8E8M0FNU> to vector<400xbf16>
  return %out : vector<400xbf16>
}

// CHECK-LABEL: @cvt_large_f6_to_f16(
// CHECK-SAME: %[[IN:.+]]: vector<400xf6E2M3FN>
func.func @cvt_large_f6_to_f16(%in : vector<400xf6E2M3FN>) -> vector<400xf16> {
  // CHECK: builtin.unrealized_conversion_cast %[[IN]] : vector<400xf6E2M3FN> to vector<400xi6>
  // CHECK: llvm.zext {{.*}} : vector<400xi6> to vector<400xi8>
  // CHECK: llvm.bitcast {{.*}} : vector<400xi8> to vector<100xi32>
  // CHECK-COUNT-200: nvvm.convert.f6x2.to.f16x2
  // CHECK-NOT: nvvm.convert.f6x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<200xi32> to vector<400xf16>
  %out = nvgpu.convert.fpext %in : vector<400xf6E2M3FN> to vector<400xf16>
  return %out : vector<400xf16>
}

// CHECK-LABEL: @cvt_large_f4_to_f16(
func.func @cvt_large_f4_to_f16(%in : vector<400xf4E2M1FN>) -> vector<400xf16> {
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : vector<400xf4E2M1FN> to vector<400xi4>
  // CHECK: llvm.bitcast {{.*}} : vector<400xi4> to vector<50xi32>
  // CHECK-COUNT-200: nvvm.convert.f4x2.to.f16x2
  // CHECK-NOT: nvvm.convert.f4x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<200xi32> to vector<400xf16>
  %out = nvgpu.convert.fpext %in : vector<400xf4E2M1FN> to vector<400xf16>
  return %out : vector<400xf16>
}

// CHECK-LABEL: @cvt_large_f8_to_f32(
func.func @cvt_large_f8_to_f32(%in : vector<400xf8E5M2>) -> vector<400xf32> {
  // CHECK-COUNT-200: nvvm.convert.f8x2.to.f16x2
  // CHECK-NOT: nvvm.convert.f8x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<200xi32> to vector<400xf16>
  // CHECK: llvm.fpext {{.*}} : vector<400xf16> to vector<400xf32>
  %out = nvgpu.convert.fpext %in : vector<400xf8E5M2> to vector<400xf32>
  return %out : vector<400xf32>
}

// CHECK-LABEL: @cvt_large_f6_to_f32(
func.func @cvt_large_f6_to_f32(%in : vector<400xf6E2M3FN>) -> vector<400xf32> {
  // CHECK: llvm.zext {{.*}} : vector<400xi6> to vector<400xi8>
  // CHECK-COUNT-200: nvvm.convert.f6x2.to.f16x2
  // CHECK-NOT: nvvm.convert.f6x2.to.f16x2
  // CHECK: llvm.fpext {{.*}} : vector<400xf16> to vector<400xf32>
  %out = nvgpu.convert.fpext %in : vector<400xf6E2M3FN> to vector<400xf32>
  return %out : vector<400xf32>
}

// CHECK-LABEL: @cvt_large_f16_to_f32(
// CHECK-SAME: %[[IN:.+]]: vector<400xf16>
func.func @cvt_large_f16_to_f32(%in : vector<400xf16>) -> vector<400xf32> {
  // CHECK-NOT: nvvm.convert
  // CHECK: llvm.fpext %[[IN]] : vector<400xf16> to vector<400xf32>
  %out = nvgpu.convert.fpext %in : vector<400xf16> to vector<400xf32>
  return %out : vector<400xf32>
}
