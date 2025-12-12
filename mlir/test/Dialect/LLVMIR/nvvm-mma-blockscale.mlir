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
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m1_e2m1(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m1_e2m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m1_e2m3(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m1_e3m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m1_e3m2(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m1_e4m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m1_e4m3(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m1_e5m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m1_e5m2(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m3_e2m1
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m3_e2m1(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m3_e2m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m3_e2m3(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m3_e3m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m3_e3m2(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m3_e4m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m3_e4m3(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e2m3_e5m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e2m3_e5m2(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e3m2_e2m1
func.func @nvvm_mxf8f6f4_blockscale_mma_e3m2_e2m1(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e3m2_e2m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e3m2_e2m3(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e3m2_e3m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e3m2_e3m2(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e3m2_e4m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e3m2_e4m3(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e3m2_e5m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e3m2_e5m2(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e3m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e4m3_e2m1
func.func @nvvm_mxf8f6f4_blockscale_mma_e4m3_e2m1(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e4m3_e2m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e4m3_e2m3(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e4m3_e3m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e4m3_e3m2(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e4m3_e4m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e4m3_e4m3(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e4m3_e5m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e4m3_e5m2(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e4m3>,
                              multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e5m2_e2m1
func.func @nvvm_mxf8f6f4_blockscale_mma_e5m2_e2m1(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e5m2_e2m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e5m2_e2m3(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e5m2_e3m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e5m2_e3m2(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e3m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e5m2_e4m3
func.func @nvvm_mxf8f6f4_blockscale_mma_e5m2_e4m3(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e4m3>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf8f6f4_blockscale_mma_e5m2_e5m2
func.func @nvvm_mxf8f6f4_blockscale_mma_e5m2_e5m2(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 32>,
                              multiplicandAPtxType = #nvvm.mma_type<e5m2>,
                              multiplicandBPtxType = #nvvm.mma_type<e5m2>,
                              scaleVecSize = #nvvm.scale_vec_size<x1>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf8f6f4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// =============================================================================
// MXF4 Block Scale MMA Operations (m16n8k64)
// =============================================================================

// CHECK-LABEL: @nvvm_mxf4_blockscale_mma
func.func @nvvm_mxf4_blockscale_mma(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x2>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// =============================================================================
// MXF4NVF4 Block Scale MMA Operations (m16n8k64)
// =============================================================================

// CHECK-LABEL: @nvvm_mxf4nvf4_blockscale_mma_ue8m0
func.func @nvvm_mxf4nvf4_blockscale_mma_ue8m0(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x2>,
                              blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                              kind = #nvvm.block_scale_kind<mxf4nvf4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}

// CHECK-LABEL: @nvvm_mxf4nvf4_blockscale_mma_ue4m3
func.func @nvvm_mxf4nvf4_blockscale_mma_ue4m3(%a: vector<4xi32>, %b: vector<2xi32>, %c: vector<4xf32>,
    %scaleAData: i32, %byteIdA: i16, %threadIdA: i16,
    %scaleBData: i32, %byteIdB: i16, %threadIdB: i16) {
  // CHECK: nvvm.mma.block_scale A[{{.*}}] B[{{.*}}] C[{{.*}}] scaleA[{{.*}}, {{.*}}, {{.*}}] scaleB[{{.*}}, {{.*}}, {{.*}}]
  %0 = nvvm.mma.block_scale A[%a] B[%b] C[%c]
                             scaleA[%scaleAData, %byteIdA, %threadIdA]
                             scaleB[%scaleBData, %byteIdB, %threadIdB]
                             {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                              multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                              multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                              scaleVecSize = #nvvm.scale_vec_size<x4>,
                              blockScaleFormat = #nvvm.block_scale_format<ue4m3>,
                              kind = #nvvm.block_scale_kind<mxf4nvf4>}
      : (vector<4xi32>, vector<2xi32>, vector<4xf32>) -> !llvm.struct<(vector<4xf32>)>
  return
}
