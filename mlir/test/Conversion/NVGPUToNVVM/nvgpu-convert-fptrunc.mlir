// RUN: mlir-opt %s -convert-nvgpu-to-nvvm | FileCheck %s
// RUN: mlir-opt %s -convert-nvgpu-to-nvvm -convert-vector-to-llvm | FileCheck %s --check-prefix=CHECK-E2E

// Basic aligned vector inputs.

// CHECK-LABEL: @cvt_float_f32_to_f16(
// CHECK-SAME: %[[IN:.+]]: vector<4xf32>
func.func @cvt_float_f32_to_f16(%in : vector<4xf32>) {
  // CHECK: llvm.bitcast %[[IN]] : vector<4xf32> to vector<4xi32>
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK-SAME: rnd = #nvvm.fp_rnd_mode<rn>
  // CHECK-SAME: : vector<2xf16>
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK-SAME: rnd = #nvvm.fp_rnd_mode<rn>
  // CHECK-SAME: : vector<2xf16>
  // CHECK: llvm.bitcast {{.*}} : vector<2xi32> to vector<4xf16>
  %out = nvgpu.convert.float %in : vector<4xf32> to vector<4xf16>
  return 
}

// CHECK-LABEL: @cvt_float_f32_to_f16_v8(
// CHECK-SAME: %[[IN:.+]]: vector<8xf32>
func.func @cvt_float_f32_to_f16_v8(%in : vector<8xf32>) {
  // CHECK: llvm.bitcast %[[IN]] : vector<8xf32> to vector<8xi32>
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<4xi32> to vector<8xf16>
  %out = nvgpu.convert.float %in : vector<8xf32> to vector<8xf16>
  return 
}

// CHECK-LABEL: @cvt_float_f32_to_bf16(
// CHECK-SAME: %[[IN:.+]]: vector<4xf32>
func.func @cvt_float_f32_to_bf16(%in : vector<4xf32>) {
  // CHECK: llvm.bitcast %[[IN]] : vector<4xf32> to vector<4xi32>
  // CHECK: nvvm.convert.f32x2.to.bf16x2
  // CHECK-SAME: rnd = #nvvm.fp_rnd_mode<rn>
  // CHECK-SAME: : vector<2xbf16>
  // CHECK: nvvm.convert.f32x2.to.bf16x2
  // CHECK: llvm.bitcast {{.*}} : vector<2xi32> to vector<4xbf16>
  %out = nvgpu.convert.float %in : vector<4xf32> to vector<4xbf16>
  return 
}

// CHECK-LABEL: @cvt_float_f32_to_e4m3(
// CHECK-SAME: %[[IN:.+]]: vector<8xf32>
func.func @cvt_float_f32_to_e4m3(%in : vector<8xf32>) {
  // CHECK: %[[IN_I32:.+]] = llvm.bitcast %[[IN]] : vector<8xf32> to vector<8xi32>
  // CHECK: %[[OUT_I32:.+]] = llvm.mlir.undef : vector<2xi32>
  // CHECK: %[[IDX_0:.+]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[SUB_VEC_0:.+]] = llvm.mlir.undef : vector<2xi16>
  // CHECK: nvvm.convert.f32x2.to.f8x2
  // CHECK-SAME: sat = #nvvm.sat_mode<satfinite>
  // CHECK-SAME: : i16(f8E4M3FN)
  // CHECK: llvm.insertelement {{.*}} : vector<2xi16>
  // CHECK: nvvm.convert.f32x2.to.f8x2
  // CHECK-SAME: : i16(f8E4M3FN)
  // CHECK: llvm.insertelement {{.*}} : vector<2xi16>
  // CHECK: llvm.bitcast {{.*}} : vector<2xi16> to i32
  // CHECK: llvm.insertelement {{.*}} : vector<2xi32>
  // CHECK: llvm.mlir.undef : vector<2xi16>
  // CHECK: nvvm.convert.f32x2.to.f8x2
  // CHECK: nvvm.convert.f32x2.to.f8x2
  // CHECK: llvm.bitcast {{.*}} : vector<2xi16> to i32
  // CHECK: llvm.insertelement {{.*}} : vector<2xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<2xi32> to vector<8xi8>
  %out = nvgpu.convert.float %in : vector<8xf32> to vector<8xf8E4M3FN>
  return 
}

// CHECK-LABEL: @cvt_float_f16_to_e2m3(
// CHECK-SAME: %[[IN:.+]]: vector<8xf16>
func.func @cvt_float_f16_to_e2m3(%in : vector<8xf16>) {
  // CHECK: llvm.bitcast %[[IN]] : vector<8xf16> to vector<4xi32>
  // CHECK: llvm.mlir.undef : vector<2xi32>
  // CHECK: llvm.mlir.undef : vector<2xi16>
  // CHECK: nvvm.convert.f16x2.to.f6x2
  // CHECK-SAME: : vector<2xf16> -> i16(f6E2M3FN)
  // CHECK: llvm.insertelement {{.*}} : vector<2xi16>
  // CHECK: nvvm.convert.f16x2.to.f6x2
  // CHECK-SAME: : vector<2xf16> -> i16(f6E2M3FN)
  // CHECK: llvm.insertelement {{.*}} : vector<2xi16>
  // CHECK: llvm.bitcast {{.*}} : vector<2xi16> to i32
  // CHECK: llvm.bitcast {{.*}} : vector<2xi32> to vector<8xi8>
  // CHECK: llvm.trunc {{.*}} : vector<8xi8> to vector<8xi6>
  %out = nvgpu.convert.float %in : vector<8xf16> to vector<8xf6E2M3FN>
  return
}

// CHECK-LABEL: @cvt_float_bf16_to_e3m2(
// CHECK-SAME: %[[IN:.+]]: vector<8xbf16>
func.func @cvt_float_bf16_to_e3m2(%in : vector<8xbf16>) {
  // CHECK: llvm.bitcast %[[IN]] : vector<8xbf16> to vector<4xi32>
  // CHECK: nvvm.convert.bf16x2.to.f6x2
  // CHECK-SAME: : vector<2xbf16> -> i16(f6E3M2FN)
  // CHECK: nvvm.convert.bf16x2.to.f6x2
  // CHECK-SAME: : vector<2xbf16> -> i16(f6E3M2FN)
  // CHECK: llvm.bitcast {{.*}} : vector<2xi32> to vector<8xi8>
  // CHECK: llvm.trunc {{.*}} : vector<8xi8> to vector<8xi6>
  %out = nvgpu.convert.float %in : vector<8xbf16> to vector<8xf6E3M2FN>
  return
}

// CHECK-LABEL: @cvt_float_f32_to_e2m3(
// CHECK-SAME: %[[IN:.+]]: vector<8xf32>
func.func @cvt_float_f32_to_e2m3(%in : vector<8xf32>) {
  // CHECK: llvm.bitcast %[[IN]] : vector<8xf32> to vector<8xi32>
  // CHECK: llvm.mlir.undef : vector<2xi32>
  // CHECK: llvm.mlir.undef : vector<2xi16>
  // CHECK: nvvm.convert.f32x2.to.f6x2
  // CHECK-SAME: : i16(f6E2M3FN)
  // CHECK: llvm.insertelement {{.*}} : vector<2xi16>
  // CHECK: nvvm.convert.f32x2.to.f6x2
  // CHECK-SAME: : i16(f6E2M3FN)
  // CHECK: llvm.insertelement {{.*}} : vector<2xi16>
  // CHECK: llvm.bitcast {{.*}} : vector<2xi16> to i32
  // CHECK: llvm.insertelement {{.*}} : vector<2xi32>
  // CHECK: llvm.mlir.undef : vector<2xi16>
  // CHECK: nvvm.convert.f32x2.to.f6x2
  // CHECK: nvvm.convert.f32x2.to.f6x2
  // CHECK: llvm.bitcast {{.*}} : vector<2xi16> to i32
  // CHECK: llvm.insertelement {{.*}} : vector<2xi32>
  // CHECK: llvm.bitcast {{.*}} : vector<2xi32> to vector<8xi8>
  // CHECK: llvm.trunc {{.*}} : vector<8xi8> to vector<8xi6>
  %out = nvgpu.convert.float %in : vector<8xf32> to vector<8xf6E2M3FN>
  return 
}

// CHECK-LABEL: @cvt_float_f32_to_e8m0_rz(
// CHECK-SAME: %[[IN:.+]]: vector<8xf32>
func.func @cvt_float_f32_to_e8m0_rz(%in : vector<8xf32>) {
  // CHECK: nvvm.convert.f32x2.to.f8x2
  // CHECK-SAME: rnd = #nvvm.fp_rnd_mode<rz>
  // CHECK-SAME: : i16(f8E8M0FNU)
  %out = nvgpu.convert.float %in {rnd = #nvvm.fp_rnd_mode<rz>}
      : vector<8xf32> to vector<8xf8E8M0FNU>
  return
}

// CHECK-LABEL: @cvt_float_f32_to_e8m0_rp(
// CHECK-SAME: %[[IN:.+]]: vector<8xf32>
func.func @cvt_float_f32_to_e8m0_rp(%in : vector<8xf32>) {
  // CHECK: nvvm.convert.f32x2.to.f8x2
  // CHECK-SAME: rnd = #nvvm.fp_rnd_mode<rp>
  // CHECK-SAME: : i16(f8E8M0FNU)
  %out = nvgpu.convert.float %in {rnd = #nvvm.fp_rnd_mode<rp>}
      : vector<8xf32> to vector<8xf8E8M0FNU>
  return
}

// Scalar inputs (canonicalize: broadcast + pad + extract).

// CHECK-LABEL: @fptrunc_scalar_f32_to_f16
// CHECK-SAME: %[[IN:.+]]: f32
func.func @fptrunc_scalar_f32_to_f16(%in : f32) -> f16 {
  // CHECK: vector.broadcast %[[IN]] : f32 to vector<1xf32>
  // CHECK: vector.insert_strided_slice
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK: vector.extract_strided_slice
  // CHECK: vector.extract {{.*}}[0] : f16 from vector<1xf16>
  %out = nvgpu.convert.float %in : f32 to f16
  return %out : f16
}

// CHECK-LABEL: @fptrunc_scalar_f32_to_bf16
// CHECK-SAME: %[[IN:.+]]: f32
func.func @fptrunc_scalar_f32_to_bf16(%in : f32) -> bf16 {
  // CHECK: vector.broadcast %[[IN]]
  // CHECK: nvvm.convert.f32x2.to.bf16x2
  // CHECK: vector.extract
  %out = nvgpu.convert.float %in : f32 to bf16
  return %out : bf16
}

// CHECK-LABEL: @fptrunc_scalar_f64_to_f32
func.func @fptrunc_scalar_f64_to_f32(%arg0: f64) -> f32 {
  // CHECK: vector.broadcast
  // CHECK: llvm.fptrunc
  // CHECK: vector.extract
  %out = nvgpu.convert.float %arg0 : f64 to f32
  return %out : f32
}

// Multi-rank vectors (canonicalize: shape_cast flatten).

// CHECK-LABEL: @fptrunc_v2x4_f32_to_f16
// CHECK-SAME: %[[IN:.+]]: vector<2x4xf32>
func.func @fptrunc_v2x4_f32_to_f16(%in : vector<2x4xf32>) -> vector<2x4xf16> {
  // CHECK: vector.shape_cast %[[IN]] : vector<2x4xf32> to vector<8xf32>
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK: vector.shape_cast {{.*}} to vector<2x4xf16>
  %out = nvgpu.convert.float %in : vector<2x4xf32> to vector<2x4xf16>
  return %out : vector<2x4xf16>
}

// CHECK-LABEL: @fptrunc_v4x2_f32_to_f8
// CHECK-SAME: %[[IN:.+]]: vector<4x2xf32>
func.func @fptrunc_v4x2_f32_to_f8(%in : vector<4x2xf32>) -> vector<4x2xf8E4M3FN> {
  // CHECK: vector.shape_cast %[[IN]] : vector<4x2xf32> to vector<8xf32>
  // CHECK: nvvm.convert.f32x2.to.f8x2
  // CHECK: vector.shape_cast {{.*}} to vector<4x2xf8E4M3FN>
  %out = nvgpu.convert.float %in : vector<4x2xf32> to vector<4x2xf8E4M3FN>
  return %out : vector<4x2xf8E4M3FN>
}

// Non-aligned 1-D vectors (canonicalize: pad via insert/extract_strided_slice).

// CHECK-LABEL: @fptrunc_v1f32_to_v1f16
// CHECK-SAME: %[[IN:.+]]: vector<1xf32>
func.func @fptrunc_v1f32_to_v1f16(%in : vector<1xf32>) -> vector<1xf16> {
  // CHECK: vector.insert_strided_slice %[[IN]]
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK: vector.extract_strided_slice
  %out = nvgpu.convert.float %in : vector<1xf32> to vector<1xf16>
  return %out : vector<1xf16>
}

// CHECK-LABEL: @fptrunc_v3f16_to_v3f8
// CHECK-SAME: %[[IN:.+]]: vector<3xf16>
func.func @fptrunc_v3f16_to_v3f8(%in : vector<3xf16>) -> vector<3xf8E4M3FN> {
  // CHECK: vector.insert_strided_slice %[[IN]]
  // CHECK: nvvm.convert.f16x2.to.f8x2
  // CHECK: vector.extract_strided_slice
  %out = nvgpu.convert.float %in : vector<3xf16> to vector<3xf8E4M3FN>
  return %out : vector<3xf8E4M3FN>
}

// Multi-rank + padding combined.

// CHECK-LABEL: @fptrunc_v3x1_f32_to_f16
// CHECK-SAME: %[[IN:.+]]: vector<3x1xf32>
func.func @fptrunc_v3x1_f32_to_f16(%in : vector<3x1xf32>) -> vector<3x1xf16> {
  // CHECK: vector.shape_cast %[[IN]] : vector<3x1xf32> to vector<3xf32>
  // CHECK: vector.insert_strided_slice
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK: vector.extract_strided_slice
  // CHECK: vector.shape_cast {{.*}} to vector<3x1xf16>
  %out = nvgpu.convert.float %in : vector<3x1xf32> to vector<3x1xf16>
  return %out : vector<3x1xf16>
}

// f64 source truncation.

// CHECK-LABEL: @fptrunc_f64_to_f32
func.func @fptrunc_f64_to_f32(%arg0: vector<4xf64>) -> vector<4xf32> {
  // CHECK: llvm.fptrunc %{{.*}} : vector<4xf64> to vector<4xf32>
  // CHECK-NOT: nvvm
  %out = nvgpu.convert.float %arg0 : vector<4xf64> to vector<4xf32>
  return %out : vector<4xf32>
}

// CHECK-LABEL: @fptrunc_f64_to_f16
func.func @fptrunc_f64_to_f16(%arg0: vector<2xf64>) -> vector<2xf16> {
  // CHECK: vector.insert_strided_slice
  // CHECK: llvm.fptrunc %{{.*}} : vector<4xf64> to vector<4xf16>
  // CHECK-NOT: nvvm.convert
  // CHECK: vector.extract_strided_slice
  %out = nvgpu.convert.float %arg0 : vector<2xf64> to vector<2xf16>
  return %out : vector<2xf16>
}

// CHECK-LABEL: @fptrunc_f64_to_bf16
func.func @fptrunc_f64_to_bf16(%arg0: vector<2xf64>) -> vector<2xbf16> {
  // CHECK: vector.insert_strided_slice
  // CHECK: llvm.fptrunc %{{.*}} : vector<4xf64> to vector<4xbf16>
  // CHECK-NOT: nvvm.convert
  // CHECK: vector.extract_strided_slice
  %out = nvgpu.convert.float %arg0 : vector<2xf64> to vector<2xbf16>
  return %out : vector<2xbf16>
}

// CHECK-LABEL: @fptrunc_f64_to_f8
func.func @fptrunc_f64_to_f8(%arg0: vector<4xf64>) -> vector<4xf8E4M3FN> {
  // CHECK: vector.insert_strided_slice
  // CHECK: llvm.fptrunc %{{.*}} : vector<8xf64> to vector<8xf32>
  // CHECK: nvvm.convert.f32x2.to.f8x2
  // CHECK: vector.extract_strided_slice
  %out = nvgpu.convert.float %arg0 : vector<4xf64> to vector<4xf8E4M3FN>
  return %out : vector<4xf8E4M3FN>
}

// Saturation and relu attributes.

// CHECK-LABEL: @fptrunc_f32_to_f8_satfinite
func.func @fptrunc_f32_to_f8_satfinite(%in : vector<8xf32>) {
  // CHECK: nvvm.convert.f32x2.to.f8x2
  // CHECK-SAME: sat = #nvvm.sat_mode<satfinite>
  %out = nvgpu.convert.float %in {sat = #nvvm.sat_mode<satfinite>}
      : vector<8xf32> to vector<8xf8E4M3FN>
  return
}

// CHECK-LABEL: @fptrunc_f32_to_f16_relu
func.func @fptrunc_f32_to_f16_relu(%in : vector<4xf32>) {
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK-SAME: relu = true
  %out = nvgpu.convert.float %in {relu = true}
      : vector<4xf32> to vector<4xf16>
  return
}

// Default SATFINITE behavior.

// CHECK-LABEL: @fptrunc_f32_to_f8_default_sat
func.func @fptrunc_f32_to_f8_default_sat(%in : vector<8xf32>) {
  // CHECK: nvvm.convert.f32x2.to.f8x2
  // CHECK-SAME: sat = #nvvm.sat_mode<satfinite>
  %out = nvgpu.convert.float %in : vector<8xf32> to vector<8xf8E4M3FN>
  return
}

// CHECK-LABEL: @fptrunc_f32_to_f16_default_sat
func.func @fptrunc_f32_to_f16_default_sat(%in : vector<4xf32>) {
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK-SAME: sat = #nvvm.sat_mode<satfinite>
  %out = nvgpu.convert.float %in : vector<4xf32> to vector<4xf16>
  return
}

// CHECK-LABEL: @fptrunc_f32_to_f16_explicit_none
func.func @fptrunc_f32_to_f16_explicit_none(%in : vector<4xf32>) {
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK-NOT: satfinite
  %out = nvgpu.convert.float %in {sat = #nvvm.sat_mode<none>}
      : vector<4xf32> to vector<4xf16>
  return
}

// Stochastic rounding (RS + random_bits).

// CHECK-LABEL: @fptrunc_f32_to_f16_rs
func.func @fptrunc_f32_to_f16_rs(%in : vector<4xf32>, %rbits : i32) {
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK-SAME: rnd = #nvvm.fp_rnd_mode<rs>
  %out = nvgpu.convert.float %in, %rbits {rnd = #nvvm.fp_rnd_mode<rs>}
      : vector<4xf32> to vector<4xf16>
  return
}

// CHECK-LABEL: @fptrunc_f32_to_bf16_rs
func.func @fptrunc_f32_to_bf16_rs(%in : vector<4xf32>, %rbits : i32) {
  // CHECK: nvvm.convert.f32x2.to.bf16x2
  // CHECK-SAME: rnd = #nvvm.fp_rnd_mode<rs>
  %out = nvgpu.convert.float %in, %rbits {rnd = #nvvm.fp_rnd_mode<rs>}
      : vector<4xf32> to vector<4xbf16>
  return
}

// CHECK-LABEL: @fptrunc_f32_to_f16_rz
func.func @fptrunc_f32_to_f16_rz(%in : vector<4xf32>) {
  // CHECK: nvvm.convert.f32x2.to.f16x2
  // CHECK-SAME: rnd = #nvvm.fp_rnd_mode<rz>
  %out = nvgpu.convert.float %in {rnd = #nvvm.fp_rnd_mode<rz>}
      : vector<4xf32> to vector<4xf16>
  return
}

// CHECK-LABEL: @fptrunc_f32_to_bf16_rz
func.func @fptrunc_f32_to_bf16_rz(%in : vector<4xf32>) {
  // CHECK: nvvm.convert.f32x2.to.bf16x2
  // CHECK-SAME: rnd = #nvvm.fp_rnd_mode<rz>
  %out = nvgpu.convert.float %in {rnd = #nvvm.fp_rnd_mode<rz>}
      : vector<4xf32> to vector<4xbf16>
  return
}

// End-to-end: no residual vector ops after full lowering.

// CHECK-E2E-LABEL: @e2e_scalar_f32_to_f16
// CHECK-E2E-NOT: vector.broadcast
// CHECK-E2E-NOT: vector.insert_strided_slice
// CHECK-E2E-NOT: vector.extract_strided_slice
// CHECK-E2E-NOT: vector.extract
// CHECK-E2E-NOT: vector.shape_cast
// CHECK-E2E: nvvm.convert.f32x2.to.f16x2
// CHECK-E2E: return
func.func @e2e_scalar_f32_to_f16(%in : f32) -> f16 {
  %out = nvgpu.convert.float %in : f32 to f16
  return %out : f16
}

// CHECK-E2E-LABEL: @e2e_v2x4_f32_to_f16
// CHECK-E2E-NOT: vector.shape_cast
// CHECK-E2E: nvvm.convert.f32x2.to.f16x2
// CHECK-E2E: return
func.func @e2e_v2x4_f32_to_f16(%in : vector<2x4xf32>) -> vector<2x4xf16> {
  %out = nvgpu.convert.float %in : vector<2x4xf32> to vector<2x4xf16>
  return %out : vector<2x4xf16>
}

// CHECK-E2E-LABEL: @e2e_v3f16_to_v3f8
// CHECK-E2E-NOT: vector.insert_strided_slice
// CHECK-E2E-NOT: vector.extract_strided_slice
// CHECK-E2E: nvvm.convert.f16x2.to.f8x2
// CHECK-E2E: return
func.func @e2e_v3f16_to_v3f8(%in : vector<3xf16>) -> vector<3xf8E4M3FN> {
  %out = nvgpu.convert.float %in : vector<3xf16> to vector<3xf8E4M3FN>
  return %out : vector<3xf8E4M3FN>
}
