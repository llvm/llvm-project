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
      {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} :
      !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf16>

    %2 = xegpu.create_nd_tdesc %arg1: memref<16x16xf16>
      -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
    %3 = xegpu.load_nd %2[%c0, %c0]
      {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
      : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
      -> vector<16x16xf16>

    %4 = xegpu.dpas %1, %3
      {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
       layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>,
       layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
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
    {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
    : !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf32>

  %4 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %3) -> (vector<8x16xf32>) {

    %5 = xegpu.create_nd_tdesc %arg0: memref<1024x1024xbf16>
      -> !xegpu.tensor_desc<8x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %6 = xegpu.create_nd_tdesc %arg1 : memref<1024x1024xbf16>
      -> !xegpu.tensor_desc<16x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>

    %7 = xegpu.load_nd %5[%0, %arg3]
      {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<8x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xbf16>
    %8 = xegpu.load_nd %6[%arg3, %1]
      {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
      : !xegpu.tensor_desc<16x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xbf16>

    %9 = xegpu.dpas %7, %8, %arg4
      {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
       layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>,
       layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
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
        layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>
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
        layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>
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
    %1 = xegpu.load_nd %0[%c0, %c0]  {layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf16>
    %2 = xegpu.create_nd_tdesc %arg1 : memref<16x8xi32>
      -> !xegpu.tensor_desc<16x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>>
    %3 = xegpu.load_nd %2[%c0, %c0]  {layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}
      : !xegpu.tensor_desc<16x8xi32, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>> -> vector<16x8xi32>
    %4 = vector.bitcast %3 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2]>}
      : vector<16x8xi32> to vector<16x16xf16>
    %5 = vector.transpose %4, [1, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}
      : vector<16x16xf16> to vector<16x16xf16>
    %6 = xegpu.dpas %1, %5
      {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
       layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>,
       layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    %7 = xegpu.create_nd_tdesc %arg2 : memref<8x16xf32>
      -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %6, %7[%c0, %c0] : vector<8x16xf32>,
      !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return

  }
}

// -----
// CHECK-LABEL: gpu.func @warp_scf_for_unused_uniform_for_result(
// CHECK:         %[[W:.*]]:2 = gpu.warp_execute_on_lane_0(%{{.*}})[16] args(%{{.*}} : index,
// CHECK-SAME:      !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>,
// CHECK-SAME:      memref<16x16xf32>) -> (vector<16x1xf32>, vector<16x1xf32>) {
// CHECK:           gpu.yield %{{.*}}, {{.*}} : vector<16x16xf32>, vector<16x1xf32>
// CHECK:         }
// CHECK:         %{{.*}}:2 = scf.for {{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK-SAME:      (%{{.*}} = %[[W]]#0, %{{.*}} = %[[W]]#1) -> (vector<16x1xf32>, vector<16x1xf32>) {
// CHECK:           %[[W1:.*]]:2 = gpu.warp_execute_on_lane_0(%{{.*}})[16]
// CHECK-SAME:        args(%{{.*}} : vector<16x1xf32>, vector<16x1xf32>) -> (vector<16x1xf32>, vector<16x1xf32>) {
// CHECK:             gpu.yield %{{.*}}, %{{.*}} : vector<16x16xf32>, vector<16x1xf32>
// CHECK:           }
// CHECK:           scf.yield %[[W1]]#0, %[[W1]]#1 : vector<16x1xf32>, vector<16x1xf32>
// CHECK:         }
gpu.module @xevm_module{
  gpu.func @warp_scf_for_unused_uniform_for_result(%arg0: index,
    %arg1: !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>,
    %arg2: memref<16x16xf32>) {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %ini = "some_def"() {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : () -> (vector<16x1xf32>)
    %ini2 = "some_def"() {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
      : () -> (vector<16x16xf32>)
    %3:2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %ini2, %arg5 = %ini) -> (vector<16x16xf32>, vector<16x1xf32>) {
      %1  = "some_def"(%arg5)
        {
          layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
          layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
        }
        : (vector<16x1xf32>) -> (vector<16x1xf32>)
      %acc = "some_def"(%arg4, %1)
        {
          layout_operand_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
          layout_operand_1 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>,
          layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
        }
        : (vector<16x16xf32>, vector<16x1xf32>) -> (vector<16x16xf32>)
      scf.yield %acc, %1 : vector<16x16xf32>, vector<16x1xf32>
    }
    {
      layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>
    }
    xegpu.store_nd %3#0, %arg1[%c0, %c0]
      : vector<16x16xf32>, !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @load_store_matrix_1({{.*}}) {
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[LANE_ID:.*]] = gpu.lane_id
// CHECK: %[[REMU1:.*]] = arith.remui %[[LANE_ID]], %[[C8]]
// CHECK: %[[DIVU:.*]] = arith.divui %[[LANE_ID]], %[[C8]]
// CHECK: %[[REMU2:.*]] = arith.remui %[[DIVU]], %[[C2]]
// CHECK: %[[REMU3:.*]] = arith.remui %[[REMU2]], %[[C2]]
// CHECK: %[[REMU4:.*]] = arith.remui %[[REMU1]], %[[C8]]
// CHECK: %[[MAT:.*]] = xegpu.load_matrix %arg0[%[[REMU3]], %[[REMU4]]] : !xegpu.mem_desc<32x32xf32>, index, index -> vector<1x1xf32>
// CHECK: xegpu.store_matrix %[[MAT]], %arg0[%[[REMU3]], %[[REMU4]]] : vector<1x1xf32>, !xegpu.mem_desc<32x32xf32>, index, index
gpu.module @xevm_module{
  gpu.func @load_store_matrix_1(%arg0: !xegpu.mem_desc<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %1 = xegpu.load_matrix %arg0[%c0, %c0] <{layout = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>}> : !xegpu.mem_desc<32x32xf32>, index, index -> vector<2x8xf32>
    xegpu.store_matrix %1, %arg0[%c0, %c0] <{layout = #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>}> : vector<2x8xf32>, !xegpu.mem_desc<32x32xf32>, index, index
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @load_store_matrix_2({{.*}}) {
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[LANE_ID:.*]] = gpu.lane_id
// CHECK: %[[REMU1:.*]] = arith.remui %[[LANE_ID]], %[[C4]]
// CHECK: %[[DIVU:.*]] = arith.divui %[[LANE_ID]], %[[C4]]
// CHECK: %[[REMU2:.*]] = arith.remui %[[DIVU]], %[[C4]]
// CHECK: %[[MUL:.*]] = arith.muli %[[REMU2]], %[[C2]]
// CHECK: %[[REMU3:.*]] = arith.remui %[[MUL]], %[[C8]]
// CHECK: %[[REMU4:.*]] = arith.remui %[[REMU1]], %[[C4]]
// CHECK: %[[ADD:.*]] = arith.addi %[[REMU4]], %[[C1]]
// CHECK: %[[MAT:.*]] = xegpu.load_matrix %arg0[%[[REMU3]], %[[ADD]]] : !xegpu.mem_desc<32x32xf32>, index, index -> vector<2x1xf32>
// CHECK: xegpu.store_matrix %[[MAT]], %arg0[%[[REMU3]], %[[ADD]]] : vector<2x1xf32>, !xegpu.mem_desc<32x32xf32>, index, index
gpu.module @xevm_module{
  gpu.func @load_store_matrix_2(%arg0: !xegpu.mem_desc<32x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = xegpu.load_matrix %arg0[%c0, %c1] <{layout = #xegpu.layout<lane_layout = [4, 4], lane_data = [2, 1]>}> : !xegpu.mem_desc<32x32xf32>, index, index -> vector<8x4xf32>
    xegpu.store_matrix %1, %arg0[%c0, %c1] <{layout = #xegpu.layout<lane_layout = [4, 4], lane_data = [2, 1]>}> : vector<8x4xf32>, !xegpu.mem_desc<32x32xf32>, index, index
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @load_store_matrix_3({{.*}}) {
// CHECK: %[[MAT:.*]] = xegpu.load_matrix %arg0[%{{.*}}, %{{.*}}] <{subgroup_block_io}>:
// CHECK-SAME: !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<block = [16, 1], stride = [1, 32]>>, index, index -> vector<1x2xf32>
// CHECK: xegpu.store_matrix %[[MAT]], %arg0[%{{.*}}, %{{.*}}] <{subgroup_block_io}>:
// CHECK-SAME: vector<1x2xf32>, !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<block = [16, 1], stride = [1, 32]>>, index, index
gpu.module @xevm_module{
  gpu.func @load_store_matrix_3(%arg0: !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<stride = [1, 32], block = [16, 1]>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %1 = xegpu.load_matrix %arg0[%c0, %c1] {subgroup_block_io, layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>} :
      !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<stride = [1, 32], block = [16, 1]>>, index, index -> vector<16x2xf32>
    xegpu.store_matrix %1, %arg0[%c0, %c1] {subgroup_block_io, layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>} :
      vector<16x2xf32>, !xegpu.mem_desc<32x32xf32, #xegpu.mem_layout<stride = [1, 32], block = [16, 1]>>, index, index
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @vector_broadcast_1d_to_2d_broadcast_within_lane({{.*}}) {
gpu.module @xevm_module{
   gpu.func  @vector_broadcast_1d_to_2d_broadcast_within_lane(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>} dense<0.000000e+00> : vector<16xf16>
    %tdesc0 = xegpu.create_nd_tdesc %arg0 : memref<16x16xf16>
      -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %tdesc1 = xegpu.create_nd_tdesc %arg1 : memref<16x16xf16>
      -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %0 = xegpu.load_nd %tdesc0[%c0, %c0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf16>
    %1 = vector.multi_reduction <add>, %0, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>} [0] : vector<16x16xf16> to vector<16xf16>
    // CHECK: %[[BCAST:.*]] = vector.broadcast %{{.*}} : f16 to vector<16xf16>
    %2 = vector.broadcast %1 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<16xf16> to vector<16x16xf16>
    xegpu.store_nd %2, %tdesc1[%c0, %c0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @vector_broadcast_2d_to_2d_across_lane_lower_to_noop_case({{.*}}) {
gpu.module @xevm_module{
   gpu.func  @vector_broadcast_2d_to_2d_across_lane_lower_to_noop_case(%arg0: memref<16xf16>, %arg1: memref<16x16xf16>) {
    %c0 = arith.constant 0 : index
    %mask = vector.constant_mask [16] {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}: vector<16xi1>
    %1 = xegpu.load %arg0[%c0], %mask {layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}: memref<16xf16>, index, vector<16xi1> -> vector<16xf16>

    %11 = vector.shape_cast %1 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<16xf16> to vector<16x1xf16>
    %2 = vector.broadcast %11 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<16x1xf16> to vector<16x16xf16>
    // CHECK-NOT: vector.broadcast
    // CHECK-NOT: vector.shape_cast

    %tdesc1 = xegpu.create_nd_tdesc %arg1 : memref<16x16xf16>
      -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    // CHECK: xegpu.store_nd {{.*}}, {{.*}}[{{.*}}, {{.*}}]
    // CHECK-SAME: : vector<16xf16>, !xegpu.tensor_desc<16x16xf16>

    xegpu.store_nd %2, %tdesc1[%c0, %c0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @vector_shape_cast_scalar_to_vector({{.*}}) {
gpu.module @xevm_module{
   gpu.func  @vector_shape_cast_scalar_to_vector(%arg0: memref<16xf16>, %arg1: memref<16x16xf16>) {
    %c0 = arith.constant 0 : index
    %9 = gpu.block_id  x
    %10 = arith.index_cast %9 : index to i16
    %11 = arith.bitcast %10 : i16 to f16
    // CHECK: vector.broadcast {{.*}} : f16 to vector<16xf16>
    %2 = vector.broadcast %11 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : f16 to vector<16x16xf16>
    %tdesc1 = xegpu.create_nd_tdesc %arg1 : memref<16x16xf16>
      -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %2, %tdesc1[%c0, %c0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----
gpu.module @xevm_test {
    // CHECK-LABEL: gpu.func @vector_reduce_2d
    // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
    // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
    // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
    // CHECK-DAG: %[[C16:.*]] = arith.constant 16 : i32
    // CHECK-DAG: %[[CST_1:.*]] = arith.constant 1.000000e+00 : f32
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[CST_0:.*]] = arith.constant dense<true> : vector<1xi1>
    // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0> : vector<1xindex>
    // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %arg0 : memref<4x16xf32> -> !xegpu.tensor_desc<4x16xf32>
    // CHECK: %[[LOADED:.*]] = xegpu.load_nd %[[TDESC]][0, 0] : !xegpu.tensor_desc<4x16xf32> -> vector<4xf32>
    // CHECK: %[[LOADED_REDUCED:.*]] = vector.reduction <add>, %[[LOADED]], %[[CST_1]] : vector<4xf32> into f32
    // CHECK: %[[SHUFFLE_0:.*]], %{{.*}} = gpu.shuffle xor %[[LOADED_REDUCED]], %[[C1]], %[[C16]] : f32
    // CHECK: %[[VEC_RED_0:.*]] = arith.addf %[[LOADED_REDUCED]], %[[SHUFFLE_0]] : f32
    // CHECK: %[[SHUFFLE_1:.*]], %{{.*}} = gpu.shuffle xor %[[VEC_RED_0]], %[[C2]], %[[C16]] : f32
    // CHECK: %[[VEC_RED_1:.*]] = arith.addf %[[VEC_RED_0]], %[[SHUFFLE_1]] : f32
    // CHECK: %[[SHUFFLE_2:.*]], %{{.*}} = gpu.shuffle xor %[[VEC_RED_1]], %[[C4]], %[[C16]] : f32
    // CHECK: %[[VEC_RED_2:.*]] = arith.addf %[[VEC_RED_1]], %[[SHUFFLE_2]] : f32
    // CHECK: %[[SHUFFLE_3:.*]], %{{.*}} = gpu.shuffle xor %[[VEC_RED_2]], %[[C8]], %[[C16]] : f32
    // CHECK: %[[VEC_RED_3:.*]] = arith.addf %[[VEC_RED_2]], %[[SHUFFLE_3]] : f32
    // CHECK: %[[VEC_RED:.*]] = vector.broadcast %[[VEC_RED_3]] : f32 to vector<1xf32>
    // CHECK: xegpu.store %[[VEC_RED]], %arg1[%[[CST]]], %[[CST_0]] : vector<1xf32>, memref<256xf32>, vector<1xindex>, vector<1xi1>
  gpu.func @vector_reduce_2d(%arg0: memref<4x16xf32>, %arg1: memref<256xf32>) {
      %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0, 1]>} 1.000000e+00 : f32
      %0 = xegpu.create_nd_tdesc %arg0 : memref<4x16xf32> -> !xegpu.tensor_desc<4x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
      %1 = xegpu.load_nd %0[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<4x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<4x16xf32>
      %2 = vector.broadcast %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>} : f32 to vector<16xf32>
      %3 = vector.multi_reduction <add>, %1, %2 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>} [0] : vector<4x16xf32> to vector<16xf32>
      %4 = vector.reduction <add>, %3 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0, 1]>} : vector<16xf32> into f32
      %5 = vector.broadcast %4 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>} : f32 to vector<16xf32>
      %cst_0 = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<0> : vector<16xindex>
      %cst_1 = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<true> : vector<16xi1>
      xegpu.store %5, %arg1[%cst_0], %cst_1 <{layout = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [0]>}> : vector<16xf32>, memref<256xf32>, vector<16xindex>, vector<16xi1>
    gpu.return
  }
}
