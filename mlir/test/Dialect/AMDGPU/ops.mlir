// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt -allow-unregistered-dialect %s | mlir-opt -allow-unregistered-dialect | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -allow-unregistered-dialect -mlir-print-op-generic %s | mlir-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @ext_packed_fp8
// CHECK: amdgpu.ext_packed_fp8
func.func @ext_packed_fp8(%v: vector<4xf8E4M3FNUZ>) -> f32 {
  %ret = amdgpu.ext_packed_fp8 %v[0] : vector<4xf8E4M3FNUZ> to f32
  func.return %ret : f32
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
