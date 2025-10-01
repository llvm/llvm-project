// RUN: mlir-opt --xevm-attach-target='module=xevm_* chip=pvc' -xegpu-subgroup-distribute \
// RUN: -allow-unregistered-dialect -canonicalize -cse -split-input-file %s | FileCheck %s

// CHECK-LABEL: gpu.func @load_dpas_postop_store
// CHECK:         (%[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xf16>,
// CHECK-SAME:      %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK:         %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG0]] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
// CHECK:         %[[T3:.*]] = xegpu.load_nd %[[T2]][%{{.*}}]  : !xegpu.tensor_desc<8x16xf16> -> vector<8xf16>
// CHECK:         %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG1]] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK:         %[[T1:.*]] = xegpu.load_nd %[[T0]][%{{.*}}] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK-DAG:     %[[T4:.*]] = xegpu.dpas %[[T3]], %[[T1]] : vector<8xf16>, vector<16xf16> -> vector<8xf32>
// CHECK:         %[[T5:.*]] = vector.shape_cast %[[T4]] : vector<8xf32> to vector<8x1xf32>
// CHECK:         %[[T6:.*]] = math.exp %[[T5]] {{{.*}}} : vector<8x1xf32>
// CHECK-DAG:     %[[T8:.*]] = vector.shape_cast %[[T6]] : vector<8x1xf32> to vector<8xf32>
// CHECK-DAG:     %[[T7:.*]] = xegpu.create_nd_tdesc %[[ARG2]] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK:         xegpu.store_nd %[[T8]], %[[T7]][{{.*}}] : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
gpu.module @xevm_module{
  gpu.func @load_dpas_postop_store(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
    %c0 = arith.constant 0 : index
    %0 = xegpu.create_nd_tdesc %arg0 : memref<8x16xf16>
      -> !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %1 = xegpu.load_nd %0[%c0, %c0]
      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} :
      !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf16>

    %2 = xegpu.create_nd_tdesc %arg1: memref<16x16xf16>
      -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
    %3 = xegpu.load_nd %2[%c0, %c0]
      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
      : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
      -> vector<16x16xf16>

    %4 = xegpu.dpas %1, %3
      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>

    %5 = math.exp %4
      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<8x16xf32>

    %6 = xegpu.create_nd_tdesc %arg2 : memref<8x16xf32> ->
      !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %5, %6[%c0, %c0] : vector<8x16xf32>,
      !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @gemm
// CHECK:         (%[[ARG0:[0-9a-zA-Z]+]]: memref<1024x1024xbf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<1024x1024xbf16>,
// CHECK-SAME:     %[[ARG2:[0-9a-zA-Z]+]]: memref<1024x1024xf32>) {
// CHECK-DAG:         %[[BLOCK_ID_X:.*]] = gpu.block_id x
// CHECK-DAG:         %[[BLOCK_ID_Y:.*]] = gpu.block_id y
// CHECK-DAG:         %[[Y_COORD:.*]] = arith.muli %[[BLOCK_ID_Y]], %c16 : index
// CHECK-DAG:         %[[X_COORD:.*]] = arith.muli %[[BLOCK_ID_X]], %c8 : index
// CHECK:             %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG2]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK-NEXT:        %[[T3:.*]] = xegpu.load_nd %[[T2]][%[[X_COORD]], %[[Y_COORD]]] : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>
// CHECK-NEXT:        %[[T4:.*]] = vector.shape_cast %[[T3]] : vector<8xf32> to vector<8x1xf32>
// CHECK:             %[[T5:.*]] = scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG4:.*]] = %[[T4]])
// CHECK-SAME:          -> (vector<8x1xf32>) {
// CHECK-DAG:           %[[T10:.*]] = xegpu.create_nd_tdesc %[[ARG1]] : memref<1024x1024xbf16> -> !xegpu.tensor_desc<16x16xbf16>
// CHECK-DAG:           %[[T11:.*]] = xegpu.load_nd %[[T10]][%[[K]], %[[Y_COORD]]] <{packed}> : !xegpu.tensor_desc<16x16xbf16> -> vector<16xbf16>
// CHECK-DAG:           %[[T12:.*]] = xegpu.create_nd_tdesc %[[ARG0]] : memref<1024x1024xbf16> -> !xegpu.tensor_desc<8x16xbf16>
// CHECK-DAG:           %[[T13:.*]] = xegpu.load_nd %[[T12]][%[[X_COORD]], %[[K]]] : !xegpu.tensor_desc<8x16xbf16> -> vector<8xbf16>
// CHECK-DAG:           %[[T14:.*]] = vector.shape_cast %[[ARG4]] : vector<8x1xf32> to vector<8xf32>
// CHECK-NEXT:          %[[T15:.*]] = xegpu.dpas %[[T13]], %[[T11]], %[[T14]]
// CHECK-SAME:            : vector<8xbf16>, vector<16xbf16>, vector<8xf32> -> vector<8xf32>
// CHECK-NEXT:          %[[T16:.*]] = vector.shape_cast %[[T15]] : vector<8xf32> to vector<8x1xf32>
// CHECK-NEXT:          scf.yield %[[T16]] : vector<8x1xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[T9:.*]] = vector.shape_cast %[[T5]] : vector<8x1xf32> to vector<8xf32>
// CHECK-NEXT:        xegpu.store_nd %[[T9]], %[[T2]][%[[X_COORD]], %[[Y_COORD]]] : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
gpu.module @xevm_module{
gpu.func @gemm(%arg0: memref<1024x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xf32>){
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %block_id_x = gpu.block_id  x
  %block_id_y = gpu.block_id  y
  %0 = arith.muli %block_id_x, %c8 : index
  %1 = arith.muli %block_id_y, %c16 : index
  %2 = xegpu.create_nd_tdesc %arg2 : memref<1024x1024xf32> ->
    !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %3 = xegpu.load_nd %2[%0, %1]
    {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf32>

  %4 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %3) -> (vector<8x16xf32>) {

    %5 = xegpu.create_nd_tdesc %arg0: memref<1024x1024xbf16>
      -> !xegpu.tensor_desc<8x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %6 = xegpu.create_nd_tdesc %arg1 : memref<1024x1024xbf16>
      -> !xegpu.tensor_desc<16x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>

    %7 = xegpu.load_nd %5[%0, %arg3]
      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<8x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xbf16>
    %8 = xegpu.load_nd %6[%arg3, %1]
      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
      : !xegpu.tensor_desc<16x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xbf16>

    %9 = xegpu.dpas %7, %8, %arg4
      {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<8x16xbf16>, vector<16x16xbf16>, vector<8x16xf32> -> vector<8x16xf32>

    scf.yield %9 : vector<8x16xf32>
  } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}

  xegpu.store_nd %4, %2[%0, %1] : vector<8x16xf32>,
    !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @scatter_ops_scf_yield
// CHECK:         (%{{.*}}: memref<256xf16>, %[[PREDICATE:[a-zA-Z0-9]+]]: i1) {
// CHECK-DAG:      %[[CST:.*]] = arith.constant dense<1.200000e+01> : vector<1x8xf16>
// CHECK-DAG:      %[[OFFSET:.*]] = arith.constant dense<12> : vector<1xindex>
// CHECK-DAG:      %[[MASK:.*]] = arith.constant dense<true> : vector<1xi1>
// CHECK:          %[[IF:.*]] = scf.if %[[PREDICATE]] -> (vector<1x8xf16>) {
// CHECK-NEXT:        %[[LD:.*]] = xegpu.load %{{.*}}[%[[OFFSET]]], %[[MASK]] <{chunk_size = 8 : i64}>
// CHECK-SAME:          : memref<256xf16>, vector<1xindex>, vector<1xi1> -> vector<8xf16>
// CHECK-NEXT:        %[[LD_CAST:.*]] = vector.shape_cast %[[LD]] : vector<8xf16> to vector<1x8xf16>
// CHECK-NEXT:        scf.yield %[[LD_CAST]] : vector<1x8xf16>
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %[[CST]] : vector<1x8xf16>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[IF_CAST:.*]] = vector.shape_cast %[[IF]] : vector<1x8xf16> to vector<8xf16>
// CHECK-NEXT:      xegpu.store %[[IF_CAST]], %{{.*}}[%[[OFFSET]]], %[[MASK]] <{chunk_size = 8 : i64}>
// CHECK-SAME:        vector<8xf16>, memref<256xf16>, vector<1xindex>, vector<1xi1>
gpu.module @xevm_module{
  gpu.func @scatter_ops_scf_yield(%src: memref<256xf16>, %pred : i1) {
    %1 = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<1>: vector<16xi1>
    %offset = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<12> : vector<16xindex>
    %loaded = scf.if %pred -> (vector<16x8xf16>) {
      %3 = xegpu.load %src[%offset], %1 <{chunk_size=8}> {
        layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>
      } : memref<256xf16>, vector<16xindex>, vector<16xi1> -> vector<16x8xf16>
      scf.yield %3 : vector<16x8xf16>
    } else {
      %3 = arith.constant {
        layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>
      } dense<12.> : vector<16x8xf16>
      scf.yield %3 : vector<16x8xf16>
    } { layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]> }
    xegpu.store %loaded, %src[%offset], %1 <{chunk_size=8}> : vector<16x8xf16>, memref<256xf16>, vector<16xindex>, vector<16xi1>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @scatter_ops_scf_non_yield({{.*}}) {
// CHECK:         %[[OFFSET:.*]] = arith.constant dense<12> : vector<1xindex>
// CHECK:         %[[MASK:.*]] = arith.constant dense<true> : vector<1xi1>
// CHECK:         %[[PREDICATE:.*]] = llvm.mlir.poison : i1
// CHECK:         scf.if %[[PREDICATE]] {
// CHECK-NEXT:      %[[LOADED:.*]] = xegpu.load %arg0[%[[OFFSET]]], %[[MASK]] <{chunk_size = 8 : i64}>
// CHECK-SAME:         memref<256xf16>, vector<1xindex>, vector<1xi1> -> vector<8xf16>
// CHECK-NEXT:      xegpu.store %[[LOADED]], %arg0[%[[OFFSET]]], %[[MASK]] <{chunk_size = 8 : i64}>
// CHECK-SAME:         vector<8xf16>, memref<256xf16>, vector<1xindex>, vector<1xi1>
// CHECK-NEXT:    }
gpu.module @xevm_module{
  gpu.func @scatter_ops_scf_non_yield(%src: memref<256xf16>) {
    %pred = llvm.mlir.poison : i1
    %1 = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<1>: vector<16xi1>
    %offset = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<12> : vector<16xindex>
    scf.if %pred  {
      %3 = xegpu.load %src[%offset], %1 <{chunk_size=8}> {
        layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>
      } : memref<256xf16>, vector<16xindex>, vector<16xi1> -> vector<16x8xf16>
      xegpu.store %3, %src[%offset], %1 <{chunk_size=8}> : vector<16x8xf16>, memref<256xf16>, vector<16xindex>, vector<16xi1>
    }
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @mma_transpose_b(
// CHECK: %[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x8xi32>, %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK-DAG:     %[[ADESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
// CHECK-DAG:     %[[BDESC:.*]] = xegpu.create_nd_tdesc %[[ARG1]] : memref<16x8xi32> -> !xegpu.tensor_desc<16x8xi32>
// CHECK-DAG:     %[[A:.*]] = xegpu.load_nd %[[ADESC]][%{{.*}}] : !xegpu.tensor_desc<8x16xf16> -> vector<8xf16>
// CHECK-DAG:     %[[B:.*]] = xegpu.load_nd %[[BDESC]][%{{.*}}] <{transpose = array<i64: 1, 0>}>
// CHECK-SAME:      !xegpu.tensor_desc<16x8xi32> -> vector<8xi32>
// CHECK-NEXT:    %[[BCAST0:.*]] = vector.shape_cast %[[B]] : vector<8xi32> to vector<1x8xi32>
// CHECK-NEXT:    %[[BCAST1:.*]] = vector.bitcast %[[BCAST0]] : vector<1x8xi32> to vector<1x16xf16>
// CHECK-NEXT:    %[[BCAST2:.*]] = vector.shape_cast %[[BCAST1]] : vector<1x16xf16> to vector<16xf16>
// CHECK-NEXT:    %[[C:.*]] = xegpu.dpas %[[A]], %[[BCAST2]] : vector<8xf16>, vector<16xf16> -> vector<8xf32>
gpu.module @xevm_module{
  gpu.func @mma_transpose_b(%arg0: memref<8x16xf16>, %arg1: memref<16x8xi32>, %arg2: memref<8x16xf32>) {
    %c0 = arith.constant 0 : index
    %0 = xegpu.create_nd_tdesc %arg0 : memref<8x16xf16>
      -> !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %1 = xegpu.load_nd %0[%c0, %c0]  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf16>
    %2 = xegpu.create_nd_tdesc %arg1 : memref<16x8xi32>
      -> !xegpu.tensor_desc<16x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>>
    %3 = xegpu.load_nd %2[%c0, %c0]  {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<16x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>> -> vector<16x8xi32>
    %4 = vector.bitcast %3 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>}
      : vector<16x8xi32> to vector<16x16xf16>
    %5 = vector.transpose %4, [1, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
      : vector<16x16xf16> to vector<16x16xf16>
    %6 = xegpu.dpas %1, %5 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    %7 = xegpu.create_nd_tdesc %arg2 : memref<8x16xf32>
      -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %6, %7[%c0, %c0] : vector<8x16xf32>,
      !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return

  }
}
