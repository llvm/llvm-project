// RUN: mlir-opt %s -convert-nvgpu-to-nvvm | FileCheck %s
// RUN: mlir-opt %s -convert-nvgpu-to-nvvm -convert-vector-to-llvm | FileCheck %s --check-prefix=CHECK-E2E

// Basic aligned vector extension to f16/bf16.

// CHECK-LABEL: @cvt_float_e4m3fn_to_f16(
// CHECK-SAME: %[[IN:.+]]: vector<8xf8E4M3FN>
func.func @cvt_float_e4m3fn_to_f16(%in : vector<8xf8E4M3FN>) {
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[IN]] : vector<8xf8E4M3FN> to vector<8xi8>
  // CHECK: llvm.bitcast %[[CAST]] : vector<8xi8> to vector<2xi32>
  // CHECK: llvm.mlir.undef : vector<4xi32>
  // CHECK: llvm.bitcast {{.*}} : i32 to vector<2xi16>
  // CHECK: llvm.bitcast {{.*}} : i16 to vector<2xi8>
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK-SAME: : vector<2xi8>(f8E4M3FN)
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK-SAME: : vector<2xi8>(f8E4M3FN)
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<4xi32> to vector<8xf16>
  %out = nvgpu.convert.float %in : vector<8xf8E4M3FN> to vector<8xf16>
  return 
}

// -----

// CHECK-LABEL: @cvt_float_e5m2_to_f16(
// CHECK-SAME: %[[IN:.+]]: vector<8xf8E5M2>
func.func @cvt_float_e5m2_to_f16(%in : vector<8xf8E5M2>) {
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[IN]] : vector<8xf8E5M2> to vector<8xi8>
  // CHECK: %[[IN_I32:.+]] = llvm.bitcast %[[CAST]] : vector<8xi8> to vector<2xi32>
  // CHECK: %[[OUT_I32:.+]] = llvm.mlir.undef : vector<4xi32>
  // CHECK: llvm.extractelement %[[IN_I32]]
  // CHECK: llvm.bitcast {{.*}} : i32 to vector<2xi16>
  // CHECK: llvm.extractelement {{.*}} : vector<2xi16>
  // CHECK: llvm.bitcast {{.*}} : i16 to vector<2xi8>
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK-SAME: : vector<2xi8>(f8E5M2)
  // CHECK: llvm.bitcast {{.*}} : vector<2xf16> to i32
  // CHECK: llvm.insertelement {{.*}} : vector<4xi32>
  // CHECK: llvm.extractelement {{.*}} : vector<2xi16>
  // CHECK: llvm.bitcast {{.*}} : i16 to vector<2xi8>
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK-SAME: : vector<2xi8>(f8E5M2)
  // CHECK: llvm.bitcast {{.*}} : vector<2xf16> to i32
  // CHECK: llvm.insertelement {{.*}} : vector<4xi32>
  // CHECK: llvm.bitcast {{.*}} : i32 to vector<2xi16>
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<4xi32> to vector<8xf16>
  %out = nvgpu.convert.float %in : vector<8xf8E5M2> to vector<8xf16>
  return 
}

// -----

// CHECK-LABEL: @cvt_float_e8m0_to_bf16(
// CHECK-SAME: %[[IN:.+]]: vector<8xf8E8M0FNU>
func.func @cvt_float_e8m0_to_bf16(%in : vector<8xf8E8M0FNU>) {
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[IN]] : vector<8xf8E8M0FNU> to vector<8xi8>
  // CHECK: llvm.bitcast %[[CAST]] : vector<8xi8> to vector<2xi32>
  // CHECK: llvm.mlir.undef : vector<4xi32>
  // CHECK: llvm.bitcast {{.*}} : i32 to vector<2xi16>
  // CHECK: llvm.bitcast {{.*}} : i16 to vector<2xi8>
  // CHECK: nvvm.convert.f8x2.to.bf16x2
  // CHECK-SAME: : vector<2xi8>(f8E8M0FNU)
  // CHECK: llvm.bitcast {{.*}} : vector<2xbf16> to i32
  // CHECK: nvvm.convert.f8x2.to.bf16x2
  // CHECK-SAME: : vector<2xi8>(f8E8M0FNU)
  // CHECK: nvvm.convert.f8x2.to.bf16x2
  // CHECK: nvvm.convert.f8x2.to.bf16x2
  // CHECK: llvm.bitcast {{.*}} : vector<4xi32> to vector<8xbf16>
  %out = nvgpu.convert.float %in : vector<8xf8E8M0FNU> to vector<8xbf16>
  return 
}

// -----

// CHECK-LABEL: @cvt_float_e2m3_to_f16(
// CHECK-SAME: %[[IN:.+]]: vector<8xf6E2M3FN>
func.func @cvt_float_e2m3_to_f16(%in : vector<8xf6E2M3FN>) {
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[IN]] : vector<8xf6E2M3FN> to vector<8xi6>
  // CHECK: llvm.zext %[[CAST]] : vector<8xi6> to vector<8xi8>
  // CHECK: llvm.bitcast {{.*}} : vector<8xi8> to vector<2xi32>
  // CHECK: llvm.mlir.undef : vector<4xi32>
  // CHECK: llvm.bitcast {{.*}} : i32 to vector<2xi16>
  // CHECK: llvm.bitcast {{.*}} : i16 to vector<2xi8>
  // CHECK: nvvm.convert.f6x2.to.f16x2
  // CHECK-SAME: : vector<2xi8>(f6E2M3FN)
  // CHECK: nvvm.convert.f6x2.to.f16x2
  // CHECK-SAME: : vector<2xi8>(f6E2M3FN)
  // CHECK: nvvm.convert.f6x2.to.f16x2
  // CHECK: nvvm.convert.f6x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<4xi32> to vector<8xf16>
  %out = nvgpu.convert.float %in : vector<8xf6E2M3FN> to vector<8xf16>
  return 
}

// -----

// CHECK-LABEL: @cvt_float_e3m2_to_f16(
// CHECK-SAME: %[[IN:.+]]: vector<8xf6E3M2FN>
func.func @cvt_float_e3m2_to_f16(%in : vector<8xf6E3M2FN>) {
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[IN]] : vector<8xf6E3M2FN> to vector<8xi6>
  // CHECK: llvm.zext %[[CAST]] : vector<8xi6> to vector<8xi8>
  // CHECK: llvm.bitcast {{.*}} : vector<8xi8> to vector<2xi32>
  // CHECK: %[[OUT_I32:.+]] = llvm.mlir.undef : vector<4xi32>
  // CHECK: llvm.bitcast {{.*}} : i32 to vector<2xi16>
  // CHECK: llvm.bitcast {{.*}} : i16 to vector<2xi8>
  // CHECK: nvvm.convert.f6x2.to.f16x2
  // CHECK-SAME: : vector<2xi8>(f6E3M2FN)
  // CHECK: llvm.bitcast {{.*}} : vector<2xf16> to i32
  // CHECK: llvm.insertelement {{.*}} : vector<4xi32>
  // CHECK: nvvm.convert.f6x2.to.f16x2
  // CHECK-SAME: : vector<2xi8>(f6E3M2FN)
  // CHECK: llvm.insertelement {{.*}} : vector<4xi32>
  // CHECK: nvvm.convert.f6x2.to.f16x2
  // CHECK: nvvm.convert.f6x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<4xi32> to vector<8xf16>
  %out = nvgpu.convert.float %in : vector<8xf6E3M2FN> to vector<8xf16>
  return
}

// -----

// CHECK-LABEL: @cvt_float_e2m1_to_f16(
// CHECK-SAME: %[[IN:.+]]: vector<16xf4E2M1FN>
func.func @cvt_float_e2m1_to_f16(%in : vector<16xf4E2M1FN>) {
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[IN]] : vector<16xf4E2M1FN> to vector<16xi4>
  // CHECK: %[[IN_I32:.+]] = llvm.bitcast %[[CAST]] : vector<16xi4> to vector<2xi32>
  // CHECK: %[[OUT_I32:.+]] = llvm.mlir.undef : vector<8xi32>
  // CHECK: llvm.extractelement %[[IN_I32]]
  // CHECK: llvm.bitcast {{.*}} : i32 to vector<4xi8>
  // CHECK: llvm.extractelement {{.*}} : vector<4xi8>
  // CHECK: nvvm.convert.f4x2.to.f16x2
  // CHECK-SAME: : i8(f4E2M1FN)
  // CHECK: llvm.bitcast {{.*}} : vector<2xf16> to i32
  // CHECK: llvm.insertelement {{.*}} : vector<8xi32>
  // CHECK: llvm.extractelement {{.*}} : vector<4xi8>
  // CHECK: nvvm.convert.f4x2.to.f16x2
  // CHECK-SAME: : i8(f4E2M1FN)
  // CHECK: llvm.insertelement {{.*}} : vector<8xi32>
  // CHECK: nvvm.convert.f4x2.to.f16x2
  // CHECK: llvm.insertelement {{.*}} : vector<8xi32>
  // CHECK: nvvm.convert.f4x2.to.f16x2
  // CHECK: llvm.insertelement {{.*}} : vector<8xi32>
  // CHECK: llvm.bitcast {{.*}} : i32 to vector<4xi8>
  // CHECK: nvvm.convert.f4x2.to.f16x2
  // CHECK: nvvm.convert.f4x2.to.f16x2
  // CHECK: nvvm.convert.f4x2.to.f16x2
  // CHECK: nvvm.convert.f4x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<8xi32> to vector<16xf16>
  %out = nvgpu.convert.float %in : vector<16xf4E2M1FN> to vector<16xf16>
  return 
}

// -----

// CHECK-LABEL: @cvt_float_e2m3_to_bf16(
// CHECK-SAME: %[[IN:.+]]: vector<8xf6E2M3FN>
func.func @cvt_float_e2m3_to_bf16(%in : vector<8xf6E2M3FN>) {
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[IN]] : vector<8xf6E2M3FN> to vector<8xi6>
  // CHECK: llvm.zext %[[CAST]] : vector<8xi6> to vector<8xi8>
  // CHECK: llvm.bitcast {{.*}} : vector<8xi8> to vector<2xi32>
  // CHECK: nvvm.convert.f6x2.to.bf16x2
  // CHECK-SAME: : vector<2xi8>(f6E2M3FN)
  // CHECK: nvvm.convert.f6x2.to.bf16x2
  // CHECK: nvvm.convert.f6x2.to.bf16x2
  // CHECK: nvvm.convert.f6x2.to.bf16x2
  // CHECK: llvm.bitcast {{.*}} : vector<4xi32> to vector<8xbf16>
  %out = nvgpu.convert.float %in : vector<8xf6E2M3FN> to vector<8xbf16>
  return
}

// -----

// CHECK-LABEL: @cvt_float_e2m1_to_bf16(
// CHECK-SAME: %[[IN:.+]]: vector<16xf4E2M1FN>
func.func @cvt_float_e2m1_to_bf16(%in : vector<16xf4E2M1FN>) {
  // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[IN]] : vector<16xf4E2M1FN> to vector<16xi4>
  // CHECK: llvm.bitcast %[[CAST]] : vector<16xi4> to vector<2xi32>
  // CHECK: nvvm.convert.f4x2.to.bf16x2
  // CHECK-SAME: : i8(f4E2M1FN)
  // CHECK: nvvm.convert.f4x2.to.bf16x2
  // CHECK: llvm.bitcast {{.*}} : vector<8xi32> to vector<16xbf16>
  %out = nvgpu.convert.float %in : vector<16xf4E2M1FN> to vector<16xbf16>
  return
}

// Extension to f32 (two-step: narrow -> f16/bf16 -> f32).

// -----

// CHECK-LABEL: @fpext_e4m3fn_to_f32(
// CHECK-SAME: %[[IN:.+]]: vector<4xf8E4M3FN>
func.func @fpext_e4m3fn_to_f32(%in : vector<4xf8E4M3FN>) -> vector<4xf32> {
  // CHECK: builtin.unrealized_conversion_cast %[[IN]] : vector<4xf8E4M3FN> to vector<4xi8>
  // CHECK: llvm.bitcast {{.*}} : vector<4xi8> to vector<1xi32>
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<2xi32> to vector<4xf16>
  // CHECK: llvm.fpext {{.*}} : vector<4xf16> to vector<4xf32>
  %out = nvgpu.convert.float %in : vector<4xf8E4M3FN> to vector<4xf32>
  return %out : vector<4xf32>
}

// -----

// CHECK-LABEL: @fpext_e5m2_to_f32(
// CHECK-SAME: %[[IN:.+]]: vector<4xf8E5M2>
func.func @fpext_e5m2_to_f32(%in : vector<4xf8E5M2>) -> vector<4xf32> {
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<2xi32> to vector<4xf16>
  // CHECK: llvm.fpext {{.*}} : vector<4xf16> to vector<4xf32>
  %out = nvgpu.convert.float %in : vector<4xf8E5M2> to vector<4xf32>
  return %out : vector<4xf32>
}

// -----

// CHECK-LABEL: @fpext_e8m0_to_f32(
// CHECK-SAME: %[[IN:.+]]: vector<4xf8E8M0FNU>
func.func @fpext_e8m0_to_f32(%in : vector<4xf8E8M0FNU>) -> vector<4xf32> {
  // CHECK: nvvm.convert.f8x2.to.bf16x2
  // CHECK: nvvm.convert.f8x2.to.bf16x2
  // CHECK: llvm.bitcast {{.*}} : vector<2xi32> to vector<4xbf16>
  // CHECK: llvm.fpext {{.*}} : vector<4xbf16> to vector<4xf32>
  %out = nvgpu.convert.float %in : vector<4xf8E8M0FNU> to vector<4xf32>
  return %out : vector<4xf32>
}

// -----

// CHECK-LABEL: @fpext_e2m3_to_f32(
// CHECK-SAME: %[[IN:.+]]: vector<4xf6E2M3FN>
func.func @fpext_e2m3_to_f32(%in : vector<4xf6E2M3FN>) -> vector<4xf32> {
  // CHECK: builtin.unrealized_conversion_cast %[[IN]] : vector<4xf6E2M3FN> to vector<4xi6>
  // CHECK: llvm.zext {{.*}} : vector<4xi6> to vector<4xi8>
  // CHECK: nvvm.convert.f6x2.to.f16x2
  // CHECK: nvvm.convert.f6x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<2xi32> to vector<4xf16>
  // CHECK: llvm.fpext {{.*}} : vector<4xf16> to vector<4xf32>
  %out = nvgpu.convert.float %in : vector<4xf6E2M3FN> to vector<4xf32>
  return %out : vector<4xf32>
}

// -----

// CHECK-LABEL: @fpext_e2m1_to_f32(
// CHECK-SAME: %[[IN:.+]]: vector<8xf4E2M1FN>
func.func @fpext_e2m1_to_f32(%in : vector<8xf4E2M1FN>) -> vector<8xf32> {
  // CHECK: nvvm.convert.f4x2.to.f16x2
  // CHECK: llvm.bitcast {{.*}} : vector<4xi32> to vector<8xf16>
  // CHECK: llvm.fpext {{.*}} : vector<8xf16> to vector<8xf32>
  %out = nvgpu.convert.float %in : vector<8xf4E2M1FN> to vector<8xf32>
  return %out : vector<8xf32>
}

// -----

// CHECK-LABEL: @fpext_f16_to_f32(
// CHECK-SAME: %[[IN:.+]]: vector<4xf16>
func.func @fpext_f16_to_f32(%in : vector<4xf16>) -> vector<4xf32> {
  // CHECK: llvm.fpext %[[IN]] : vector<4xf16> to vector<4xf32>
  %out = nvgpu.convert.float %in : vector<4xf16> to vector<4xf32>
  return %out : vector<4xf32>
}

// -----

// CHECK-LABEL: @fpext_bf16_to_f32(
// CHECK-SAME: %[[IN:.+]]: vector<4xbf16>
func.func @fpext_bf16_to_f32(%in : vector<4xbf16>) -> vector<4xf32> {
  // CHECK: llvm.fpext %[[IN]] : vector<4xbf16> to vector<4xf32>
  %out = nvgpu.convert.float %in : vector<4xbf16> to vector<4xf32>
  return %out : vector<4xf32>
}

// Extension to f64.

// CHECK-LABEL: @fpext_f16_to_f64
func.func @fpext_f16_to_f64(%arg0: vector<4xf16>) -> vector<4xf64> {
  // CHECK: llvm.fpext %{{.*}} : vector<4xf16> to vector<4xf64>
  // CHECK-NOT: llvm.fpext
  %out = nvgpu.convert.float %arg0 : vector<4xf16> to vector<4xf64>
  return %out : vector<4xf64>
}

// CHECK-LABEL: @fpext_bf16_to_f64
func.func @fpext_bf16_to_f64(%arg0: vector<4xbf16>) -> vector<4xf64> {
  // CHECK: llvm.fpext %{{.*}} : vector<4xbf16> to vector<4xf64>
  // CHECK-NOT: llvm.fpext
  %out = nvgpu.convert.float %arg0 : vector<4xbf16> to vector<4xf64>
  return %out : vector<4xf64>
}

// CHECK-LABEL: @fpext_f32_to_f64
func.func @fpext_f32_to_f64(%arg0: vector<4xf32>) -> vector<4xf64> {
  // CHECK-NOT: nvvm
  // CHECK: llvm.fpext %{{.*}} : vector<4xf32> to vector<4xf64>
  %out = nvgpu.convert.float %arg0 : vector<4xf32> to vector<4xf64>
  return %out : vector<4xf64>
}

// CHECK-LABEL: @fpext_f8_to_f64
func.func @fpext_f8_to_f64(%arg0: vector<4xf8E4M3FN>) -> vector<4xf64> {
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: llvm.fpext {{.*}} to {{.*}}f64
  // CHECK-NOT: llvm.fpext {{.*}} to {{.*}}f32
  // CHECK: vector.extract_strided_slice
  %out = nvgpu.convert.float %arg0 : vector<4xf8E4M3FN> to vector<4xf64>
  return %out : vector<4xf64>
}

// CHECK-LABEL: @fpext_f4_to_f64
func.func @fpext_f4_to_f64(%arg0: vector<8xf4E2M1FN>) -> vector<8xf64> {
  // CHECK: nvvm.convert.f4x2.to.f16x2
  // CHECK: llvm.fpext {{.*}} to {{.*}}f64
  // CHECK-NOT: llvm.fpext {{.*}} to {{.*}}f32
  %out = nvgpu.convert.float %arg0 : vector<8xf4E2M1FN> to vector<8xf64>
  return %out : vector<8xf64>
}

// CHECK-LABEL: @fpext_e2m3_to_f64
func.func @fpext_e2m3_to_f64(%arg0: vector<4xf6E2M3FN>) -> vector<4xf64> {
  // CHECK: llvm.zext {{.*}} : vector<8xi6> to vector<8xi8>
  // CHECK: nvvm.convert.f6x2.to.f16x2
  // CHECK: llvm.fpext {{.*}} to {{.*}}f64
  // CHECK-NOT: llvm.fpext {{.*}} to {{.*}}f32
  %out = nvgpu.convert.float %arg0 : vector<4xf6E2M3FN> to vector<4xf64>
  return %out : vector<4xf64>
}

// Scalar inputs (canonicalize: broadcast + pad + extract).

// CHECK-LABEL: @fpext_scalar_f8_to_f16
// CHECK-SAME: %[[IN:.+]]: f8E4M3FN
func.func @fpext_scalar_f8_to_f16(%in : f8E4M3FN) -> f16 {
  // CHECK: vector.broadcast %[[IN]] : f8E4M3FN to vector<1xf8E4M3FN>
  // CHECK: vector.insert_strided_slice
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: vector.extract_strided_slice
  // CHECK: vector.extract {{.*}}[0] : f16 from vector<1xf16>
  %out = nvgpu.convert.float %in : f8E4M3FN to f16
  return %out : f16
}

// -----

// CHECK-LABEL: @fpext_scalar_f8_to_f32
// CHECK-SAME: %[[IN:.+]]: f8E4M3FN
func.func @fpext_scalar_f8_to_f32(%in : f8E4M3FN) -> f32 {
  // CHECK: vector.broadcast
  // CHECK: vector.insert_strided_slice
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: llvm.fpext
  // CHECK: vector.extract_strided_slice
  // CHECK: vector.extract {{.*}}[0] : f32 from vector<1xf32>
  %out = nvgpu.convert.float %in : f8E4M3FN to f32
  return %out : f32
}

// Multi-rank vectors (canonicalize: shape_cast flatten).

// CHECK-LABEL: @fpext_v2x4_f8_to_f16
// CHECK-SAME: %[[IN:.+]]: vector<2x4xf8E4M3FN>
func.func @fpext_v2x4_f8_to_f16(%in : vector<2x4xf8E4M3FN>) -> vector<2x4xf16> {
  // CHECK: vector.shape_cast %[[IN]] : vector<2x4xf8E4M3FN> to vector<8xf8E4M3FN>
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: vector.shape_cast {{.*}} to vector<2x4xf16>
  %out = nvgpu.convert.float %in : vector<2x4xf8E4M3FN> to vector<2x4xf16>
  return %out : vector<2x4xf16>
}

// -----

// CHECK-LABEL: @fpext_v2x4_f8_to_f32
// CHECK-SAME: %[[IN:.+]]: vector<2x4xf8E4M3FN>
func.func @fpext_v2x4_f8_to_f32(%in : vector<2x4xf8E4M3FN>) -> vector<2x4xf32> {
  // CHECK: vector.shape_cast {{.*}} to vector<8xf8E4M3FN>
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: llvm.fpext
  // CHECK: vector.shape_cast {{.*}} to vector<2x4xf32>
  %out = nvgpu.convert.float %in : vector<2x4xf8E4M3FN> to vector<2x4xf32>
  return %out : vector<2x4xf32>
}

// Non-aligned 1-D vectors (canonicalize: pad via insert/extract_strided_slice).

// CHECK-LABEL: @fpext_v3f8_to_v3f16
// CHECK-SAME: %[[IN:.+]]: vector<3xf8E5M2>
func.func @fpext_v3f8_to_v3f16(%in : vector<3xf8E5M2>) -> vector<3xf16> {
  // CHECK: vector.insert_strided_slice %[[IN]]
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: vector.extract_strided_slice
  %out = nvgpu.convert.float %in : vector<3xf8E5M2> to vector<3xf16>
  return %out : vector<3xf16>
}

// -----

// CHECK-LABEL: @fpext_v3_f8_to_f32
// CHECK-SAME: %[[IN:.+]]: vector<3xf8E5M2>
func.func @fpext_v3_f8_to_f32(%in : vector<3xf8E5M2>) -> vector<3xf32> {
  // CHECK: vector.insert_strided_slice
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: llvm.fpext
  // CHECK: vector.extract_strided_slice
  %out = nvgpu.convert.float %in : vector<3xf8E5M2> to vector<3xf32>
  return %out : vector<3xf32>
}

// Multi-rank + padding combined.

// CHECK-LABEL: @fpext_v3x1_f8_to_f16
// CHECK-SAME: %[[IN:.+]]: vector<3x1xf8E4M3FN>
func.func @fpext_v3x1_f8_to_f16(%in : vector<3x1xf8E4M3FN>) -> vector<3x1xf16> {
  // CHECK: vector.shape_cast %[[IN]] : vector<3x1xf8E4M3FN> to vector<3xf8E4M3FN>
  // CHECK: vector.insert_strided_slice
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK: vector.extract_strided_slice
  // CHECK: vector.shape_cast {{.*}} to vector<3x1xf16>
  %out = nvgpu.convert.float %in : vector<3x1xf8E4M3FN> to vector<3x1xf16>
  return %out : vector<3x1xf16>
}

// Relu attribute.

// CHECK-LABEL: @fpext_f8_to_f16_relu
func.func @fpext_f8_to_f16_relu(%in : vector<8xf8E4M3FN>) {
  // CHECK: nvvm.convert.f8x2.to.f16x2
  // CHECK-SAME: relu = true
  %out = nvgpu.convert.float %in {relu = true}
      : vector<8xf8E4M3FN> to vector<8xf16>
  return
}

// End-to-end: no residual vector ops after full lowering.

// CHECK-E2E-LABEL: @e2e_scalar_f8_to_f16
// CHECK-E2E-NOT: vector.broadcast
// CHECK-E2E-NOT: vector.insert_strided_slice
// CHECK-E2E-NOT: vector.extract_strided_slice
// CHECK-E2E-NOT: vector.extract
// CHECK-E2E-NOT: vector.shape_cast
// CHECK-E2E: nvvm.convert.f8x2.to.f16x2
// CHECK-E2E: return
func.func @e2e_scalar_f8_to_f16(%in : f8E4M3FN) -> f16 {
  %out = nvgpu.convert.float %in : f8E4M3FN to f16
  return %out : f16
}

// CHECK-E2E-LABEL: @e2e_v2x4_f8_to_f16
// CHECK-E2E-NOT: vector.shape_cast
// CHECK-E2E: nvvm.convert.f8x2.to.f16x2
// CHECK-E2E: return
func.func @e2e_v2x4_f8_to_f16(%in : vector<2x4xf8E4M3FN>) -> vector<2x4xf16> {
  %out = nvgpu.convert.float %in : vector<2x4xf8E4M3FN> to vector<2x4xf16>
  return %out : vector<2x4xf16>
}

// CHECK-E2E-LABEL: @e2e_v3f8_to_v3f16
// CHECK-E2E-NOT: vector.insert_strided_slice
// CHECK-E2E-NOT: vector.extract_strided_slice
// CHECK-E2E: nvvm.convert.f8x2.to.f16x2
// CHECK-E2E: return
func.func @e2e_v3f8_to_v3f16(%in : vector<3xf8E5M2>) -> vector<3xf16> {
  %out = nvgpu.convert.float %in : vector<3xf8E5M2> to vector<3xf16>
  return %out : vector<3xf16>
}

// CHECK-E2E-LABEL: @e2e_scalar_f8_to_f32
// CHECK-E2E-NOT: vector.broadcast
// CHECK-E2E-NOT: vector.insert_strided_slice
// CHECK-E2E-NOT: vector.extract_strided_slice
// CHECK-E2E-NOT: vector.extract
// CHECK-E2E-NOT: vector.shape_cast
// CHECK-E2E: nvvm.convert.f8x2.to.f16x2
// CHECK-E2E: llvm.fpext
// CHECK-E2E: return
func.func @e2e_scalar_f8_to_f32(%in : f8E4M3FN) -> f32 {
  %out = nvgpu.convert.float %in : f8E4M3FN to f32
  return %out : f32
}

// CHECK-E2E-LABEL: @e2e_v2x4_f8_to_f32
// CHECK-E2E-NOT: vector.shape_cast
// CHECK-E2E: nvvm.convert.f8x2.to.f16x2
// CHECK-E2E: llvm.fpext
// CHECK-E2E: return
func.func @e2e_v2x4_f8_to_f32(%in : vector<2x4xf8E4M3FN>) -> vector<2x4xf32> {
  %out = nvgpu.convert.float %in : vector<2x4xf8E4M3FN> to vector<2x4xf32>
  return %out : vector<2x4xf32>
}

// CHECK-E2E-LABEL: @e2e_v3f8_to_v3f32
// CHECK-E2E-NOT: vector.insert_strided_slice
// CHECK-E2E-NOT: vector.extract_strided_slice
// CHECK-E2E: nvvm.convert.f8x2.to.f16x2
// CHECK-E2E: llvm.fpext
// CHECK-E2E: return
func.func @e2e_v3f8_to_v3f32(%in : vector<3xf8E5M2>) -> vector<3xf32> {
  %out = nvgpu.convert.float %in : vector<3xf8E5M2> to vector<3xf32>
  return %out : vector<3xf32>
}
