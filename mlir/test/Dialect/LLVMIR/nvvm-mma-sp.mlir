// RUN: mlir-opt %s -split-input-file | FileCheck %s

// This file contains tests for all sparse MMA (mma.sp.sync) operations in the NVVM dialect
// Based on PTX ISA documentation:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-sparse-mma
//
// Sparse MMA operations follow 2:4 structured sparsity where 2 out of every 4 elements
// in the A operand are non-zero. The A operand is provided in compressed form,
// and sparseMetadata provides the sparsity indices.
//
// NOTE: These tests use the default (standard) metadata ordering.
// For ordered metadata tests (PTX ISA 8.5+, sm_90+), see nvvm-mma-sp-ordered.mlir.

// =============================================================================
// F16 Sparse MMA Operations (m16n8k16)
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_m16n8k16_f16_f16
func.func @nvvm_mma_sp_m16n8k16_f16_f16(
    %a0 : vector<2xf16>, %a1 : vector<2xf16>,
    %b0 : vector<2xf16>, %b1 : vector<2xf16>,
    %c0 : vector<2xf16>, %c1 : vector<2xf16>,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1]
                        sparseMetadata[%meta] selector[%sel]
                        {shape = #nvvm.shape<m = 16, n = 8, k = 16>}
      : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_m16n8k16_f16_f32
func.func @nvvm_mma_sp_m16n8k16_f16_f32(
    %a0 : vector<2xf16>, %a1 : vector<2xf16>,
    %b0 : vector<2xf16>, %b1 : vector<2xf16>,
    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {shape = #nvvm.shape<m = 16, n = 8, k = 16>}
      : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return
}

// =============================================================================
// F16 Sparse MMA Operations (m16n8k32)
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_m16n8k32_f16_f16
func.func @nvvm_mma_sp_m16n8k32_f16_f16(
    %a0 : vector<2xf16>, %a1 : vector<2xf16>, %a2 : vector<2xf16>, %a3 : vector<2xf16>,
    %b0 : vector<2xf16>, %b1 : vector<2xf16>, %b2 : vector<2xf16>, %b3 : vector<2xf16>,
    %c0 : vector<2xf16>, %c1 : vector<2xf16>,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}, {{.*}}, {{.*}}, {{.*}}] C[{{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {shape = #nvvm.shape<m = 16, n = 8, k = 32>} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1]
                        sparseMetadata[%meta] selector[%sel]
                        {shape = #nvvm.shape<m = 16, n = 8, k = 32>}
      : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_m16n8k32_f16_f32
func.func @nvvm_mma_sp_m16n8k32_f16_f32(
    %a0 : vector<2xf16>, %a1 : vector<2xf16>, %a2 : vector<2xf16>, %a3 : vector<2xf16>,
    %b0 : vector<2xf16>, %b1 : vector<2xf16>, %b2 : vector<2xf16>, %b3 : vector<2xf16>,
    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}, {{.*}}, {{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {shape = #nvvm.shape<m = 16, n = 8, k = 32>} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {shape = #nvvm.shape<m = 16, n = 8, k = 32>}
      : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return
}

// =============================================================================
// BF16 Sparse MMA Operations
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_m16n8k16_bf16_f32
func.func @nvvm_mma_sp_m16n8k16_bf16_f32(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {multiplicandAPtxType = #nvvm.mma_type<bf16>, multiplicandBPtxType = #nvvm.mma_type<bf16>, shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<bf16>,
                         multiplicandBPtxType = #nvvm.mma_type<bf16>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 16>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_m16n8k32_bf16_f32
func.func @nvvm_mma_sp_m16n8k32_bf16_f32(
    %a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32,
    %b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32,
    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}, {{.*}}, {{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {multiplicandAPtxType = #nvvm.mma_type<bf16>, multiplicandBPtxType = #nvvm.mma_type<bf16>, shape = #nvvm.shape<m = 16, n = 8, k = 32>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<bf16>,
                         multiplicandBPtxType = #nvvm.mma_type<bf16>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 32>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return
}

// =============================================================================
// TF32 Sparse MMA Operations
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_m16n8k8_tf32_f32
func.func @nvvm_mma_sp_m16n8k8_tf32_f32(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {multiplicandAPtxType = #nvvm.mma_type<tf32>, multiplicandBPtxType = #nvvm.mma_type<tf32>, shape = #nvvm.shape<m = 16, n = 8, k = 8>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<tf32>,
                         multiplicandBPtxType = #nvvm.mma_type<tf32>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 8>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_m16n8k16_tf32_f32
func.func @nvvm_mma_sp_m16n8k16_tf32_f32(
    %a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32,
    %b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32,
    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}, {{.*}}, {{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {multiplicandAPtxType = #nvvm.mma_type<tf32>, multiplicandBPtxType = #nvvm.mma_type<tf32>, shape = #nvvm.shape<m = 16, n = 8, k = 16>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<tf32>,
                         multiplicandBPtxType = #nvvm.mma_type<tf32>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 16>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return
}

// =============================================================================
// Integer (s8) Sparse MMA Operations
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_m16n8k32_s8_s32
func.func @nvvm_mma_sp_m16n8k32_s8_s32(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>, multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<s8>, shape = #nvvm.shape<m = 16, n = 8, k = 32>} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<s8>,
                         multiplicandBPtxType = #nvvm.mma_type<s8>,
                         intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 32>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_m16n8k32_s8_s32_satfinite
func.func @nvvm_mma_sp_m16n8k32_s8_s32_satfinite(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<satfinite>, multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<s8>, shape = #nvvm.shape<m = 16, n = 8, k = 32>} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<s8>,
                         multiplicandBPtxType = #nvvm.mma_type<s8>,
                         intOverflowBehavior = #nvvm.mma_int_overflow<satfinite>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 32>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_m16n8k64_s8_s32
func.func @nvvm_mma_sp_m16n8k64_s8_s32(
    %a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32,
    %b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32,
    %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}, {{.*}}, {{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>, multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<s8>, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<s8>,
                         multiplicandBPtxType = #nvvm.mma_type<s8>,
                         intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  return
}

// =============================================================================
// Integer (u8) Sparse MMA Operations
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_m16n8k32_u8_s32
func.func @nvvm_mma_sp_m16n8k32_u8_s32(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>, multiplicandAPtxType = #nvvm.mma_type<u8>, multiplicandBPtxType = #nvvm.mma_type<u8>, shape = #nvvm.shape<m = 16, n = 8, k = 32>} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<u8>,
                         multiplicandBPtxType = #nvvm.mma_type<u8>,
                         intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 32>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_m16n8k64_u8_s32
func.func @nvvm_mma_sp_m16n8k64_u8_s32(
    %a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32,
    %b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32,
    %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}, {{.*}}, {{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>, multiplicandAPtxType = #nvvm.mma_type<u8>, multiplicandBPtxType = #nvvm.mma_type<u8>, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<u8>,
                         multiplicandBPtxType = #nvvm.mma_type<u8>,
                         intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  return
}

// =============================================================================
// Sub-byte Integer (s4) Sparse MMA Operations
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_m16n8k64_s4_s32
func.func @nvvm_mma_sp_m16n8k64_s4_s32(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>, multiplicandAPtxType = #nvvm.mma_type<s4>, multiplicandBPtxType = #nvvm.mma_type<s4>, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<s4>,
                         multiplicandBPtxType = #nvvm.mma_type<s4>,
                         intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_m16n8k128_s4_s32
func.func @nvvm_mma_sp_m16n8k128_s4_s32(
    %a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32,
    %b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32,
    %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}, {{.*}}, {{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>, multiplicandAPtxType = #nvvm.mma_type<s4>, multiplicandBPtxType = #nvvm.mma_type<s4>, shape = #nvvm.shape<m = 16, n = 8, k = 128>} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<s4>,
                         multiplicandBPtxType = #nvvm.mma_type<s4>,
                         intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 128>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  return
}

// =============================================================================
// Sub-byte Integer (u4) Sparse MMA Operations
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_m16n8k64_u4_s32
func.func @nvvm_mma_sp_m16n8k64_u4_s32(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>, multiplicandAPtxType = #nvvm.mma_type<u4>, multiplicandBPtxType = #nvvm.mma_type<u4>, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<u4>,
                         multiplicandBPtxType = #nvvm.mma_type<u4>,
                         intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_m16n8k128_u4_s32
func.func @nvvm_mma_sp_m16n8k128_u4_s32(
    %a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32,
    %b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32,
    %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}, {{.*}}, {{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>, multiplicandAPtxType = #nvvm.mma_type<u4>, multiplicandBPtxType = #nvvm.mma_type<u4>, shape = #nvvm.shape<m = 16, n = 8, k = 128>} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<u4>,
                         multiplicandBPtxType = #nvvm.mma_type<u4>,
                         intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 128>}
      : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  return
}

// =============================================================================
// FP8 (e4m3) Sparse MMA Operations
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_m16n8k64_e4m3_f16
func.func @nvvm_mma_sp_m16n8k64_e4m3_f16(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : vector<2xf16>, %c1 : vector<2xf16>,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {multiplicandAPtxType = #nvvm.mma_type<e4m3>, multiplicandBPtxType = #nvvm.mma_type<e4m3>, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                         multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_m16n8k64_e4m3_f32
func.func @nvvm_mma_sp_m16n8k64_e4m3_f32(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {multiplicandAPtxType = #nvvm.mma_type<e4m3>, multiplicandBPtxType = #nvvm.mma_type<e4m3>, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                         multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return
}

// =============================================================================
// FP8 (e5m2) Sparse MMA Operations
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_m16n8k64_e5m2_f16
func.func @nvvm_mma_sp_m16n8k64_e5m2_f16(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : vector<2xf16>, %c1 : vector<2xf16>,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {multiplicandAPtxType = #nvvm.mma_type<e5m2>, multiplicandBPtxType = #nvvm.mma_type<e5m2>, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                         multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_m16n8k64_e5m2_f32
func.func @nvvm_mma_sp_m16n8k64_e5m2_f32(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {multiplicandAPtxType = #nvvm.mma_type<e5m2>, multiplicandBPtxType = #nvvm.mma_type<e5m2>, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                         multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return
}

