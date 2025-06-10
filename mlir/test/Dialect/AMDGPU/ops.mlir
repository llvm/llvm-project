// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt -allow-unregistered-dialect %s | mlir-opt -allow-unregistered-dialect | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -allow-unregistered-dialect -mlir-print-op-generic %s | mlir-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @ext_packed_fp8_s
// CHECK: amdgpu.ext_packed_fp8 {{.*}} vector<4xf8E4M3FNUZ> to f32
func.func @ext_packed_fp8_s(%v: vector<4xf8E4M3FNUZ>) -> f32 {
  %ret = amdgpu.ext_packed_fp8 %v[0] : vector<4xf8E4M3FNUZ> to f32
  func.return %ret : f32
}

// CHECK-LABEL: func @ext_packed_fp8_v
// CHECK: amdgpu.ext_packed_fp8 {{.*}} vector<4xf8E4M3FNUZ> to vector<2xf32
func.func @ext_packed_fp8_v(%v: vector<4xf8E4M3FNUZ>) -> vector<2xf32> {
  %ret = amdgpu.ext_packed_fp8 %v[0] : vector<4xf8E4M3FNUZ> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func @packed_trunc_2xfp8
// CHECK: amdgpu.packed_trunc_2xfp8
func.func @packed_trunc_2xfp8(%v1: f32, %v2: f32, %others: vector<4xf8E5M2FNUZ>, %stoch: i32) -> vector<4xf8E5M2FNUZ> {
  %ret = amdgpu.packed_trunc_2xfp8 %v1, %v2 into %others[word 1] : f32 to vector<4xf8E5M2FNUZ> into vector<4xf8E5M2FNUZ>
  func.return %ret : vector<4xf8E5M2FNUZ>
}

// CHECK-LABEL: func @packed_stoch_round_fp8
// CHECK: amdgpu.packed_stoch_round_fp8
func.func @packed_stoch_round_fp8(%v1: f32, %stoch: i32, %others: vector<4xf8E5M2FNUZ>) -> vector<4xf8E5M2FNUZ> {
  %ret = amdgpu.packed_stoch_round_fp8 %v1 + %stoch into %others[2] : f32 to vector<4xf8E5M2FNUZ> into vector<4xf8E5M2FNUZ>
  func.return %ret : vector<4xf8E5M2FNUZ>
}

// CHECK-LABEL: func.func @scaled_ext_full_f8e4m3_f32
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_full_f8e4m3_f32(%v: vector<4xf8E4M3FN>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf8E4M3FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_full_f8e4m3_f16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_full_f8e4m3_f16(%v: vector<4xf8E4M3FN>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf8E4M3FN> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_full_f8e4m3_bf16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_full_f8e4m3_bf16(%v: vector<4xf8E4M3FN>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf8E4M3FN> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_half_f8e4m3_f32
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_half_f8e4m3_f32(%v: vector<2xf8E4M3FN>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E4M3FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_half_f8e4m3_f16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_half_f8e4m3_f16(%v: vector<2xf8E4M3FN>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E4M3FN> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_half_f8e4m3_bf16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_half_f8e4m3_bf16(%v: vector<2xf8E4M3FN>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E4M3FN> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f8e4m3_f32
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_scalar_f8e4m3_f32(%v: vector<2xf8E4M3FN>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E4M3FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f8e4m3_f16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_scalar_f8e4m3_f16(%v: vector<2xf8E4M3FN>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E4M3FN> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f8e4m3_bf16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_scalar_f8e4m3_bf16(%v: vector<2xf8E4M3FN>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E4M3FN> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_full_f8e5m2_f32
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_full_f8e5m2_f32(%v: vector<4xf8E5M2>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf8E5M2> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_full_f8e5m2_f16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_full_f8e5m2_f16(%v: vector<4xf8E5M2>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf8E5M2> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_full_f8e5m2_bf16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_full_f8e5m2_bf16(%v: vector<4xf8E5M2>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf8E5M2> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_half_f8e5m2_f32
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_half_f8e5m2_f32(%v: vector<2xf8E5M2>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E5M2> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_half_f8e5m2_f16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_half_f8e5m2_f16(%v: vector<2xf8E5M2>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E5M2> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_half_f8e5m2_bf16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_half_f8e5m2_bf16(%v: vector<2xf8E5M2>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E5M2> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f8e5m2_f32
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_scalar_f8e5m2_f32(%v: vector<2xf8E5M2>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E5M2> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f8e5m2_f16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_scalar_f8e5m2_f16(%v: vector<2xf8E5M2>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E5M2> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f8e5m2_bf16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_scalar_f8e5m2_bf16(%v: vector<2xf8E5M2>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf8E5M2> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_full_f4e2m1_f32
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_full_f4e2m1_f32(%v: vector<8xf4E2M1FN>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<8xf4E2M1FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_full_f4e2m1_f16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_full_f4e2m1_f16(%v: vector<8xf4E2M1FN>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<8xf4E2M1FN> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_full_f4e2m1_bf16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_full_f4e2m1_bf16(%v: vector<8xf4E2M1FN>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<8xf4E2M1FN> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_half_f4e2m1_f32
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_half_f4e2m1_f32(%v: vector<8xf4E2M1FN>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<8xf4E2M1FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_half_f4e2m1_f16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_half_f4e2m1_f16(%v: vector<4xf4E2M1FN>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf4E2M1FN> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_half_f4e2m1_bf16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_half_f4e2m1_bf16(%v: vector<4xf4E2M1FN>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<4xf4E2M1FN> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f4e2m1_f32
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_scalar_f4e2m1_f32(%v: vector<2xf4E2M1FN>, %scale: f32) -> vector<2xf32> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf4E2M1FN> to vector<2xf32>
  func.return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f4e2m1_f16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_scalar_f4e2m1_f16(%v: vector<2xf4E2M1FN>, %scale: f32) -> vector<2xf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf4E2M1FN> to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: func.func @scaled_ext_scalar_f4e2m1_bf16
// CHECK: amdgpu.scaled_ext_packed
func.func @scaled_ext_scalar_f4e2m1_bf16(%v: vector<2xf4E2M1FN>, %scale: f32) -> vector<2xbf16> {
  %ret = amdgpu.scaled_ext_packed %v[0], %scale : vector<2xf4E2M1FN> to vector<2xbf16>
  func.return %ret : vector<2xbf16>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e4m3_f32
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_f8e4m3_f32(%v: vector<2xf32>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xf32> to vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e4m3_f32
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_into_f8e4m3_f32(%v: vector<2xf32>, %existing: vector<4xf8E4M3FN>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xf32> to vector<4xf8E4M3FN> into vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e4m3_f16
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_f8e4m3_f16(%v: vector<2xf16>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xf16> to vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e4m3_f16
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_into_f8e4m3_f16(%v: vector<2xf16>, %existing: vector<4xf8E4M3FN>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xf16> to vector<4xf8E4M3FN> into vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e4m3_bf16
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_f8e4m3_bf16(%v: vector<2xbf16>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xbf16> to vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e4m3_bf16
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_into_f8e4m3_bf16(%v: vector<2xbf16>, %existing: vector<4xf8E4M3FN>, %scale: f32) -> vector<4xf8E4M3FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xbf16> to vector<4xf8E4M3FN> into vector<4xf8E4M3FN>
  func.return %ret : vector<4xf8E4M3FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e5m2_f32
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_f8e5m2_f32(%v: vector<2xf32>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xf32> to vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e5m2_f32
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_into_f8e5m2_f32(%v: vector<2xf32>, %existing: vector<4xf8E5M2>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xf32> to vector<4xf8E5M2> into vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e5m2_f16
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_f8e5m2_f16(%v: vector<2xf16>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xf16> to vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e5m2_f16
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_into_f8e5m2_f16(%v: vector<2xf16>, %existing: vector<4xf8E5M2>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xf16> to vector<4xf8E5M2> into vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f8e5m2_bf16
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_f8e5m2_bf16(%v: vector<2xbf16>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xbf16> to vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f8e5m2_bf16
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_into_f8e5m2_bf16(%v: vector<2xbf16>, %existing: vector<4xf8E5M2>, %scale: f32) -> vector<4xf8E5M2> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xbf16> to vector<4xf8E5M2> into vector<4xf8E5M2>
  func.return %ret : vector<4xf8E5M2>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f4e2m1_f32
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_f4e2m1_f32(%v: vector<2xf32>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xf32> to vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f4e2m1_f32
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_into_f4e2m1_f32(%v: vector<2xf32>, %existing: vector<8xf4E2M1FN>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xf32> to vector<8xf4E2M1FN> into vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f4e2m1_f16
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_f4e2m1_f16(%v: vector<2xf16>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xf16> to vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f4e2m1_f16
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_into_f4e2m1_f16(%v: vector<2xf16>, %existing: vector<8xf4E2M1FN>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xf16> to vector<8xf4E2M1FN> into vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_f4e2m1_bf16
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_f4e2m1_bf16(%v: vector<2xbf16>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into undef[0], %scale : vector<2xbf16> to vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func.func @packed_scaled_trunc_into_f4e2m1_bf16
// CHECK: amdgpu.packed_scaled_trunc
func.func @packed_scaled_trunc_into_f4e2m1_bf16(%v: vector<2xbf16>, %existing: vector<8xf4E2M1FN>, %scale: f32) -> vector<8xf4E2M1FN> {
  %ret = amdgpu.packed_scaled_trunc %v into %existing[0], %scale : vector<2xbf16> to vector<8xf4E2M1FN> into vector<8xf4E2M1FN>
  func.return %ret : vector<8xf4E2M1FN>
}

// CHECK-LABEL: func @fat_raw_buffer_cast_easy
// CHECK: amdgpu.fat_raw_buffer_cast
func.func @fat_raw_buffer_cast_easy(%m: memref<8xi32>) -> memref<8xi32, #amdgpu.address_space<fat_raw_buffer>> {
  %ret = amdgpu.fat_raw_buffer_cast %m : memref<8xi32> to memref<8xi32, #amdgpu.address_space<fat_raw_buffer>>
  func.return %ret : memref<8xi32, #amdgpu.address_space<fat_raw_buffer>>
}

// CHECK-LABEL: func @fat_raw_buffer_cast
// CHECK: amdgpu.fat_raw_buffer_cast
// CHECK-SAME: validBytes(%{{[^)]*}})
// CHECK-SAME: cacheSwizzleStride(%{{[^)]*}})
// CHECK-SAME: boundsCheck(false)
// CHECK-SAME: resetOffset
func.func @fat_raw_buffer_cast(%m: memref<8xi32, strided<[1], offset: ?>>, %validBytes: i32, %cacheSwizzle: i14) -> memref<8xi32, strided<[1]>, #amdgpu.address_space<fat_raw_buffer>> {
  %ret = amdgpu.fat_raw_buffer_cast %m validBytes(%validBytes) cacheSwizzleStride(%cacheSwizzle) boundsCheck(false) resetOffset
    : memref<8xi32, strided<[1], offset: ?>> to memref<8xi32, strided<[1]>, #amdgpu.address_space<fat_raw_buffer>>
  func.return %ret : memref<8xi32, strided<[1]>, #amdgpu.address_space<fat_raw_buffer>>
}

// CHECK-LABEL: func @raw_buffer_load_f32_from_rank_1
func.func @raw_buffer_load_f32_from_rank_1(%src : memref<128xf32>, %offset : i32, %idx0 : i32) -> f32 {
  // CHECK: amdgpu.raw_buffer_load {indexOffset = 1 : i32} %{{.*}}[{{.*}}] sgprOffset %{{.*}} : memref<128xf32>, i32 -> f32
  %0 = amdgpu.raw_buffer_load {boundsCheck = true, indexOffset = 1 : i32} %src[%idx0] sgprOffset %offset : memref<128xf32>, i32 -> f32
  func.return %0 : f32
}

// CHECK-LABEL: func @raw_buffer_load_f32_from_rank_4
func.func @raw_buffer_load_f32_from_rank_4(%src : memref<128x64x32x16xf32>, %offset : i32, %idx0 : i32, %idx1 : i32, %idx2 : i32, %idx3 : i32) -> f32 {
  // CHECK: amdgpu.raw_buffer_load {indexOffset = 1 : i32} %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] sgprOffset %{{.*}} : memref<128x64x32x16xf32>, i32, i32, i32, i32 -> f32
  %0 = amdgpu.raw_buffer_load {boundsCheck = true, indexOffset = 1 : i32} %src[%idx0, %idx1, %idx2, %idx3] sgprOffset %offset : memref<128x64x32x16xf32>, i32, i32, i32, i32 -> f32
  func.return %0 : f32
}

// CHECK-LABEL: func @raw_buffer_load_scalar
func.func @raw_buffer_load_scalar(%src : memref<f32>) -> f32 {
  // CHECK: amdgpu.raw_buffer_load {indexOffset = 1 : i32} %{{.*}}[] : memref<f32> -> f32
  %0 = amdgpu.raw_buffer_load {indexOffset = 1 : i32} %src[] : memref<f32> -> f32
  func.return %0 : f32
}

// CHECK-LABEL: func @raw_buffer_load_4xf32_from_rank_4
func.func @raw_buffer_load_4xf32_from_rank_4(%src : memref<128x64x32x16xf32>, %offset : i32, %idx0 : i32, %idx1 : i32, %idx2 : i32, %idx3 : i32) -> vector<4xf32> {
  // CHECK: amdgpu.raw_buffer_load {indexOffset = 1 : i32} %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] sgprOffset %{{.*}} : memref<128x64x32x16xf32>, i32, i32, i32, i32 -> vector<4xf32>
  %0 = amdgpu.raw_buffer_load {boundsCheck = true, indexOffset = 1 : i32} %src[%idx0, %idx1, %idx2, %idx3] sgprOffset %offset : memref<128x64x32x16xf32>, i32, i32, i32, i32 -> vector<4xf32>
  func.return %0 : vector<4xf32>
}

// CHECK-LABEL: func @raw_buffer_store_f32_to_rank_1
func.func @raw_buffer_store_f32_to_rank_1(%value : f32, %dst : memref<128xf32>, %offset : i32, %idx0 : i32) {
  // CHECK: amdgpu.raw_buffer_store {indexOffset = 1 : i32} %{{.*}} -> %{{.*}}[{{.*}}] sgprOffset %{{.*}} : f32 -> memref<128xf32>, i32
  amdgpu.raw_buffer_store {boundsCheck = true, indexOffset = 1 : i32} %value -> %dst[%idx0] sgprOffset %offset : f32 -> memref<128xf32>, i32
  func.return
}

// CHECK-LABEL: func @raw_buffer_store_f32_to_rank_4
func.func @raw_buffer_store_f32_to_rank_4(%value : f32, %dst : memref<128x64x32x16xf32>, %offset : i32, %idx0 : i32, %idx1 : i32, %idx2 : i32, %idx3 : i32) {
  // CHECK: amdgpu.raw_buffer_store {indexOffset = 1 : i32} %{{.*}} -> %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] sgprOffset %{{.*}} : f32 -> memref<128x64x32x16xf32>, i32, i32, i32, i32
  amdgpu.raw_buffer_store {boundsCheck = true, indexOffset = 1 : i32} %value -> %dst[%idx0, %idx1, %idx2, %idx3] sgprOffset %offset : f32 -> memref<128x64x32x16xf32>, i32, i32, i32, i32
  func.return
}

// CHECK-LABEL: func @raw_buffer_store_4xf32_to_rank_4
func.func @raw_buffer_store_4xf32_to_rank_4(%value : vector<4xf32>, %dst : memref<128x64x32x16xf32>, %offset : i32, %idx0 : i32, %idx1 : i32, %idx2 : i32, %idx3 : i32) {
  // CHECK: amdgpu.raw_buffer_store {indexOffset = 1 : i32} %{{.*}} -> %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] sgprOffset %{{.*}} : vector<4xf32> -> memref<128x64x32x16xf32>, i32, i32, i32, i32
  amdgpu.raw_buffer_store {boundsCheck = true, indexOffset = 1 : i32} %value -> %dst[%idx0, %idx1, %idx2, %idx3] sgprOffset %offset : vector<4xf32> -> memref<128x64x32x16xf32>, i32, i32, i32, i32
  func.return
}

// CHECK-LABEL: func @raw_buffer_store_scalar
func.func @raw_buffer_store_scalar(%value : f32, %dst : memref<f32>) {
  // CHECK: amdgpu.raw_buffer_store {indexOffset = 1 : i32} %{{.*}} -> %{{.*}}[] : f32 -> memref<f32>
  amdgpu.raw_buffer_store {indexOffset = 1 : i32} %value -> %dst[] : f32 -> memref<f32>
  func.return
}

// CHECK-LABEL: func @raw_buffer_atomic_fadd_f32_to_rank_1
func.func @raw_buffer_atomic_fadd_f32_to_rank_1(%value : f32, %dst : memref<128xf32>, %offset : i32, %idx0 : i32) {
  // CHECK: amdgpu.raw_buffer_atomic_fadd {indexOffset = 1 : i32} %{{.*}} -> %{{.*}}[{{.*}}] sgprOffset %{{.*}} : f32 -> memref<128xf32>, i32
  amdgpu.raw_buffer_atomic_fadd {boundsCheck = true, indexOffset = 1 : i32} %value -> %dst[%idx0] sgprOffset %offset : f32 -> memref<128xf32>, i32
  func.return
}

// CHECK-LABEL: func @raw_buffer_atomic_fadd_f32_to_rank_4
func.func @raw_buffer_atomic_fadd_f32_to_rank_4(%value : f32, %dst : memref<128x64x32x16xf32>, %offset : i32, %idx0 : i32, %idx1 : i32, %idx2 : i32, %idx3 : i32) {
  // CHECK: amdgpu.raw_buffer_atomic_fadd {indexOffset = 1 : i32} %{{.*}} -> %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] sgprOffset %{{.*}} : f32 -> memref<128x64x32x16xf32>, i32, i32, i32, i32
  amdgpu.raw_buffer_atomic_fadd {boundsCheck = true, indexOffset = 1 : i32} %value -> %dst[%idx0, %idx1, %idx2, %idx3] sgprOffset %offset : f32 -> memref<128x64x32x16xf32>, i32, i32, i32, i32
  func.return
}

// CHECK-LABEL: func @raw_buffer_atomic_cmpswap_f32
func.func @raw_buffer_atomic_cmpswap_f32(%src : f32, %cmp : f32, %dst : memref<128x64x32x16xf32>, %offset : i32, %idx0 : i32, %idx1 : i32, %idx2 : i32, %idx3 : i32) {
  // CHECK: amdgpu.raw_buffer_atomic_cmpswap {indexOffset = 1 : i32} %{{.*}}, %{{.*}} -> %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] sgprOffset %{{.*}} : f32 -> memref<128x64x32x16xf32>, i32, i32, i32, i32
  amdgpu.raw_buffer_atomic_cmpswap {boundsCheck = true, indexOffset = 1 : i32} %src, %cmp -> %dst[%idx0, %idx1, %idx2, %idx3] sgprOffset %offset : f32 -> memref<128x64x32x16xf32>, i32, i32, i32, i32
  func.return
}

// CHECK-LABEL: func @lds_barrier
func.func @lds_barrier() {
  // CHECK: amdgpu.lds_barrier
  amdgpu.lds_barrier
  func.return
}

// CHECK-LABEL: func @sched_barrier
func.func @sched_barrier() {
  // CHECK: amdgpu.sched_barrier allow = <none>
  amdgpu.sched_barrier allow = <none>
  // CHECK: amdgpu.sched_barrier allow = <valu|all_vmem>
  amdgpu.sched_barrier allow = <valu|all_vmem>
  func.return
}

// CHECK-LABEL: func @mfma
func.func @mfma(%arg0 : f32, %arg1 : vector<32xf32>) -> vector<32xf32> {
  // CHECK: amdgpu.mfma
  %0 = amdgpu.mfma %arg0 * %arg0 + %arg1 { abid = 1 : i32, cbsz = 1 : i32, k = 1 : i32, m = 32 : i32, n = 32 : i32, blocks = 2 : i32 } blgp = bcast_second_32 : f32, f32, vector<32xf32>
  func.return %0 : vector<32xf32>
}

// CHECK-LABEL: func @wmma
func.func @wmma(%arg0 : vector<16xf16>, %arg1 : vector<8xf16>) -> vector<8xf16> {
  // CHECK: amdgpu.wmma
  %0 = amdgpu.wmma %arg0 * %arg0 + %arg1 : vector<16xf16>, vector<16xf16>, vector<8xf16>
  func.return %0 : vector<8xf16>
}

// CHECK-LABEL: func @swizzle_bitmode
func.func @swizzle_bitmode(%arg0 : f32) -> f32 {
  // CHECK: amdgpu.swizzle_bitmode
  %0 = amdgpu.swizzle_bitmode %arg0 1 2 4 : f32
  func.return %0 : f32
}

// CHECK-LABEL: func @scaled_mfma
func.func @scaled_mfma(%arg0 : f8E8M0FNU, %arg1 : vector<32xf6E2M3FN>, %arg2 : vector<16xf32>) -> vector<16xf32> {
  // CHECK: amdgpu.scaled_mfma
  %0 = amdgpu.scaled_mfma(%arg0[0] * %arg1) * (%arg0[1] * %arg1) + %arg2 { k = 64 : i32, m = 32 : i32, n = 32 : i32 } : f8E8M0FNU, vector<32xf6E2M3FN>, f8E8M0FNU, vector<32xf6E2M3FN>, vector<16xf32>
  func.return %0 : vector<16xf32>
}
