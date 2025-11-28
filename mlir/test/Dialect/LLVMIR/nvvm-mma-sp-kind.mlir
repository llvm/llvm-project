// RUN: mlir-opt %s -split-input-file | FileCheck %s

// This file contains tests for sparse MMA (mma.sp.sync) operations with KIND variants.
// The kind::f8f6f4 variant was introduced in PTX ISA 8.7 for sm_90+ architectures.
//
// Based on PTX ISA documentation:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-sparse-mma
//
// KIND::F8F6F4 enables:
// - Additional FP8 types: e3m2, e2m3, e2m1
// - F16 accumulator for m16n8k64 FP8 operations
// - Mixed-precision FP8 computations
//
// Requirements:
// - ONLY works with ordered metadata (sp::ordered_metadata)
// - ONLY for shape m16n8k64
// - ONLY for FP8 types (not integers or other floats)

// =============================================================================
// FP8 e4m3 Sparse MMA with KIND (m16n8k64)
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e4m3_f16
func.func @nvvm_mma_sp_kind_m16n8k64_e4m3_f16(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : vector<2xf16>, %c1 : vector<2xf16>,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {kind = #nvvm.mma_kind<f8f6f4>, multiplicandAPtxType = #nvvm.mma_type<e4m3>, multiplicandBPtxType = #nvvm.mma_type<e4m3>, orderedMetadata, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1]
                        sparseMetadata[%meta] selector[%sel]
                        {kind = #nvvm.mma_kind<f8f6f4>,
                         orderedMetadata,
                         multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                         multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e4m3_f32
func.func @nvvm_mma_sp_kind_m16n8k64_e4m3_f32(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {kind = #nvvm.mma_kind<f8f6f4>, multiplicandAPtxType = #nvvm.mma_type<e4m3>, multiplicandBPtxType = #nvvm.mma_type<e4m3>, orderedMetadata, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {kind = #nvvm.mma_kind<f8f6f4>,
                         orderedMetadata,
                         multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                         multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return
}

// =============================================================================
// FP8 e5m2 Sparse MMA with KIND (m16n8k64)
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e5m2_f16
func.func @nvvm_mma_sp_kind_m16n8k64_e5m2_f16(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : vector<2xf16>, %c1 : vector<2xf16>,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {kind = #nvvm.mma_kind<f8f6f4>, multiplicandAPtxType = #nvvm.mma_type<e5m2>, multiplicandBPtxType = #nvvm.mma_type<e5m2>, orderedMetadata, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1]
                        sparseMetadata[%meta] selector[%sel]
                        {kind = #nvvm.mma_kind<f8f6f4>,
                         orderedMetadata,
                         multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                         multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e5m2_f32
func.func @nvvm_mma_sp_kind_m16n8k64_e5m2_f32(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {kind = #nvvm.mma_kind<f8f6f4>, multiplicandAPtxType = #nvvm.mma_type<e5m2>, multiplicandBPtxType = #nvvm.mma_type<e5m2>, orderedMetadata, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {kind = #nvvm.mma_kind<f8f6f4>,
                         orderedMetadata,
                         multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                         multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return
}

// =============================================================================
// FP8 e3m2 Sparse MMA with KIND (m16n8k64)
// NOTE: e3m2 is ONLY available with kind::f8f6f4
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e3m2_f16
func.func @nvvm_mma_sp_kind_m16n8k64_e3m2_f16(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : vector<2xf16>, %c1 : vector<2xf16>,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {kind = #nvvm.mma_kind<f8f6f4>, multiplicandAPtxType = #nvvm.mma_type<e3m2>, multiplicandBPtxType = #nvvm.mma_type<e3m2>, orderedMetadata, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1]
                        sparseMetadata[%meta] selector[%sel]
                        {kind = #nvvm.mma_kind<f8f6f4>,
                         orderedMetadata,
                         multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                         multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e3m2_f32
func.func @nvvm_mma_sp_kind_m16n8k64_e3m2_f32(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {kind = #nvvm.mma_kind<f8f6f4>, multiplicandAPtxType = #nvvm.mma_type<e3m2>, multiplicandBPtxType = #nvvm.mma_type<e3m2>, orderedMetadata, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {kind = #nvvm.mma_kind<f8f6f4>,
                         orderedMetadata,
                         multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                         multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return
}

// =============================================================================
// FP8 e2m3 Sparse MMA with KIND (m16n8k64)
// NOTE: e2m3 is ONLY available with kind::f8f6f4
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e2m3_f16
func.func @nvvm_mma_sp_kind_m16n8k64_e2m3_f16(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : vector<2xf16>, %c1 : vector<2xf16>,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {kind = #nvvm.mma_kind<f8f6f4>, multiplicandAPtxType = #nvvm.mma_type<e2m3>, multiplicandBPtxType = #nvvm.mma_type<e2m3>, orderedMetadata, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1]
                        sparseMetadata[%meta] selector[%sel]
                        {kind = #nvvm.mma_kind<f8f6f4>,
                         orderedMetadata,
                         multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                         multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e2m3_f32
func.func @nvvm_mma_sp_kind_m16n8k64_e2m3_f32(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {kind = #nvvm.mma_kind<f8f6f4>, multiplicandAPtxType = #nvvm.mma_type<e2m3>, multiplicandBPtxType = #nvvm.mma_type<e2m3>, orderedMetadata, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {kind = #nvvm.mma_kind<f8f6f4>,
                         orderedMetadata,
                         multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                         multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return
}

// =============================================================================
// FP8 e2m1 Sparse MMA with KIND (m16n8k64)
// NOTE: e2m1 is ONLY available with kind::f8f6f4
// =============================================================================

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e2m1_f16
func.func @nvvm_mma_sp_kind_m16n8k64_e2m1_f16(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : vector<2xf16>, %c1 : vector<2xf16>,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {kind = #nvvm.mma_kind<f8f6f4>, multiplicandAPtxType = #nvvm.mma_type<e2m1>, multiplicandBPtxType = #nvvm.mma_type<e2m1>, orderedMetadata, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1]
                        sparseMetadata[%meta] selector[%sel]
                        {kind = #nvvm.mma_kind<f8f6f4>,
                         orderedMetadata,
                         multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                         multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  return
}

// CHECK-LABEL: @nvvm_mma_sp_kind_m16n8k64_e2m1_f32
func.func @nvvm_mma_sp_kind_m16n8k64_e2m1_f32(
    %a0 : i32, %a1 : i32,
    %b0 : i32, %b1 : i32,
    %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32,
    %meta : i32, %sel : i32) {
  // CHECK: nvvm.mma.sp.sync A[{{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] {kind = #nvvm.mma_kind<f8f6f4>, multiplicandAPtxType = #nvvm.mma_type<e2m1>, multiplicandBPtxType = #nvvm.mma_type<e2m1>, orderedMetadata, shape = #nvvm.shape<m = 16, n = 8, k = 64>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                        sparseMetadata[%meta] selector[%sel]
                        {kind = #nvvm.mma_kind<f8f6f4>,
                         orderedMetadata,
                         multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                         multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                         shape = #nvvm.shape<m = 16, n = 8, k = 64>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return
}

