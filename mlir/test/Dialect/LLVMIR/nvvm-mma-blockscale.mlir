// RUN: mlir-opt %s -split-input-file | FileCheck %s

// This file contains tests for all dense MMA block scale operations in the NVVM dialect
// Based on PTX ISA documentation:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-with-block-scaling
//
// MMA block scale operations perform matrix multiply-accumulate with block scaling:
// D = matmul(A * SF_A, B * SF_B) + C
// where SF_A and SF_B are scaling factors with dimensions based on scale vector size.

// =============================================================================
// MXF8F6F4 Block Scale MMA Operations (m16n8k32) - All Type Combinations
// =============================================================================

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m1_e2m1
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m1_e2m1(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m1_e2m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m1_e2m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m1_e3m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m1_e3m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m1_e4m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m1_e4m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m1_e5m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m1_e5m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m3_e2m1
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m3_e2m1(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m3_e2m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m3_e2m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m3_e3m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m3_e3m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m3_e4m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m3_e4m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m3_e5m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m3_e5m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e3m2_e2m1
func.func @nvvm_mxf8f6f4_blockscale_mma_e3m2_e2m1(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e3m2_e2m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e3m2_e2m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e3m2_e3m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e3m2_e3m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e3m2_e4m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e3m2_e4m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e3m2_e5m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e3m2_e5m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e4m3_e2m1
func.func @nvvm_mxf8f6f4_blockscale_mma_e4m3_e2m1(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e4m3_e2m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e4m3_e2m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e4m3_e3m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e4m3_e3m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e4m3_e4m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e4m3_e4m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e4m3_e5m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e4m3_e5m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e5m2_e2m1
func.func @nvvm_mxf8f6f4_blockscale_mma_e5m2_e2m1(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e5m2_e2m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e5m2_e2m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e5m2_e3m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e5m2_e3m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e5m2_e4m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e5m2_e4m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e5m2_e5m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e5m2_e5m2(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// =============================================================================
// MXF4 Block Scale MMA Operations (m16n8k64)
// =============================================================================

// CHECK-LABEL: @nvvm_mxf4_blockscale_mma
func.func @nvvm_mxf4_blockscale_mma(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x2>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// =============================================================================
// MXF4NVF4 Block Scale MMA Operations (m16n8k64)
// =============================================================================

// CHECK-LABEL: @nvvm_mxf4nvf4_blockscale_mma_ue8m0
func.func @nvvm_mxf4nvf4_blockscale_mma_ue8m0(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x2>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf4nvf4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf4nvf4_blockscale_mma_ue4m3
func.func @nvvm_mxf4nvf4_blockscale_mma_ue4m3(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x4>,
                              blockScaleFormat = #nvvm.block_scale_format<ue4m3>,
                              kind = #nvvm.block_scale_kind<mxf4nvf4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mxf4nvf4_blockscale_mma_ue8m0_x4
func.func @nvvm_mxf4nvf4_blockscale_mma_ue8m0_x4(%a0: i32, %a1: i32, %a2: i32, %a3: i32,
    %b0: i32, %b1: i32,
    %c0: f32, %c1: f32, %c2: f32, %c3: f32,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16)
    -> !llvm.struct<(f32, f32, f32, f32)> {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x4>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf4nvf4>}
      : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  return %0 : !llvm.struct<(f32, f32, f32, f32)>
}
