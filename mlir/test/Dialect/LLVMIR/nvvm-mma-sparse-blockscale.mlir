// RUN: mlir-opt %s -split-input-file | FileCheck %s

// This file contains tests for all sparse MMA block scale operations in the NVVM dialect
// Based on PTX ISA documentation:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-with-block-scaling
//
// Sparse MMA block scale operations perform matrix multiply-accumulate with block scaling
// on sparse matrices: D = matmul(A * SF_A, B * SF_B) + C
// where A follows 2:4 structured sparsity and SF_A, SF_B are scaling factors.

// =============================================================================
// MXF8F6F4 Sparse Block Scale MMA Operations (m16n8k64) - All Type Combinations
// =============================================================================

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e2m1
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e2m1(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                                 multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e2m3
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e2m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                                 multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e3m2
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e3m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                                 multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e4m3
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e4m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                                 multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e5m2
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m1_e5m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                                 multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e2m1
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e2m1(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                                 multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e2m3
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e2m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                                 multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e3m2
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e3m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                                 multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e4m3
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e4m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                                 multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e5m2
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e2m3_e5m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                                 multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e2m1
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e2m1(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                                 multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e2m3
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e2m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                                 multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e3m2
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e3m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                                 multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e4m3
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e4m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                                 multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e5m2
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e3m2_e5m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                                 multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e2m1
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e2m1(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                                 multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e2m3
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e2m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                                 multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e3m2
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e3m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                                 multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e4m3
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e4m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                                 multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e5m2
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e4m3_e5m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                                 multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e2m1
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e2m1(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                                 multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e2m3
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e2m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                                 multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e3m2
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e3m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                                 multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e4m3
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e4m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                                 multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e5m2
func.func @nvvm_mxf8f6f4_sp_blockscale_mma_e5m2_e5m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                                 multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                                 multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                                 scaleVecSize = #nvvm.scale_vec_size<x1>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf8f6f4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// =============================================================================
// MXF4 Sparse Block Scale MMA Operations (m16n8k128)
// =============================================================================

// CHECK-LABEL: @nvvm_mxf4_sp_blockscale_mma
func.func @nvvm_mxf4_sp_blockscale_mma(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 128>,
                                 multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                                 multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                                 scaleVecSize = #nvvm.scale_vec_size<x2>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// =============================================================================
// MXF4NVF4 Sparse Block Scale MMA Operations (m16n8k128)
// =============================================================================

// CHECK-LABEL: @nvvm_mxf4nvf4_sp_blockscale_mma_ue8m0
func.func @nvvm_mxf4nvf4_sp_blockscale_mma_ue8m0(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 128>,
                                 multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                                 multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                                 scaleVecSize = #nvvm.scale_vec_size<x2>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf4nvf4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf4nvf4_sp_blockscale_mma_ue4m3
func.func @nvvm_mxf4nvf4_sp_blockscale_mma_ue4m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 128>,
                                 multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                                 multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                                 scaleVecSize = #nvvm.scale_vec_size<x4>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue4m3>,
                                 kind = #nvvm.block_scale_kind<mxf4nvf4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf4nvf4_sp_blockscale_mma_ue8m0_x4
func.func @nvvm_mxf4nvf4_sp_blockscale_mma_ue8m0_x4(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32, %b2: i32, %b3: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %sparseMetadata: i32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  %sparsitySelector = llvm.mlir.constant(0 : i32) : i32
  // CHECK: nvvm.mma.sp.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] sparseMetadata[{{.*}}] selector[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.sp.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1, %b2, %b3] C[%c0, %c1, %c2, %c3]
                                sparseMetadata[%sparseMetadata]
                                selector[%sparsitySelector]
                                scaleA[%scaleAData, %byteIdA, %threadIdA]
                                scaleB[%scaleBData, %byteIdB, %threadIdB]
                                {shape = #nvvm.shape<m = 16, n = 8, k = 128>,
                                 multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                                 multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                                 scaleVecSize = #nvvm.scale_vec_size<x4>,
                                 blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                 kind = #nvvm.block_scale_kind<mxf4nvf4>,
                                 orderedMetadata}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}
