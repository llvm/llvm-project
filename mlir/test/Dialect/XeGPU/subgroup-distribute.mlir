// RUN: mlir-opt -xegpu-subgroup-distribute -allow-unregistered-dialect -canonicalize -cse -split-input-file %s | FileCheck %s

// CHECK-LABEL: gpu.func @store_nd_1d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<16xf32>) {
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<1xf32>
// CHECK-DAG: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
// CHECK: xegpu.store_nd %[[CST]], %[[T0]]  : vector<1xf32>, !xegpu.tensor_desc<16xf32>
// CHECK: gpu.return
gpu.module @test {
  gpu.func @store_nd_1d(%arg0: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<1.000000e+00> : vector<16xf32>
    %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<16xf32> -> !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    xegpu.store_nd %cst, %0  : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @store_nd_2d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<16x16xf16>) {
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<16xf16>
// CHECK-DAG: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK: xegpu.store_nd %[[CST]], %[[T0]]  : vector<16xf16>, !xegpu.tensor_desc<16x16xf16>
gpu.module @test {
  gpu.func @store_nd_2d(%arg0: memref<16x16xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} dense<1.000000e+00> : vector<16x16xf16>
    %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %cst, %0 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}



// -----
// CHECK-LABEL: gpu.func @load_nd_1d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<16xf32>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16xf32>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
// CHECK-DAG: %[[T1:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<16xf32> -> vector<1xf32>
// CHECK-DAG: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%{{.*}}] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
// CHECK: xegpu.store_nd %[[T1]], %[[T2]]  : vector<1xf32>, !xegpu.tensor_desc<16xf32>
gpu.module @test {
  gpu.func @load_nd_1d(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
    %c0 = arith.constant 0 : index
    %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<16xf32> -> !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    %1 = xegpu.load_nd %0  {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} : !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>> -> vector<16xf32>
    %2 = xegpu.create_nd_tdesc %arg1[%c0] : memref<16xf32> -> !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    xegpu.store_nd %1, %2 : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @load_nd_2d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<16x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xf16>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK-DAG: %[[T1:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK-DAG: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK: xegpu.store_nd %[[T1]], %[[T2]]  : vector<16xf16>, !xegpu.tensor_desc<16x16xf16>
gpu.module @test {
  gpu.func @load_nd_2d(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) {
    %c0 = arith.constant 0 : index
    %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %1 = xegpu.load_nd %0  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf16>
    %2 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %1, %2 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @load_nd_array_length
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<16x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xf16>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<32xf16>
// CHECK: %[[T2:.*]] = vector.shape_cast %[[T1]] : vector<32xf16> to vector<2x16x1xf16>
// CHECK: %[[T3:.*]] = vector.extract %[[T2]][0] : vector<16x1xf16> from vector<2x16x1xf16>
// CHECK-DAG: %[[T4:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK-DAG: %[[T5:.*]] = vector.shape_cast %[[T3]] : vector<16x1xf16> to vector<16xf16>
// CHECK: xegpu.store_nd %[[T5]], %[[T4]]  : vector<16xf16>, !xegpu.tensor_desc<16x16xf16>
gpu.module @test {
  gpu.func @load_nd_array_length(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>) {
    %c0 = arith.constant 0 : index
    %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %1 = xegpu.load_nd %0  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<2x16x16xf16>
    %2 = vector.extract %1[%c0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<16x16xf16> from vector<2x16x16xf16>
    %3 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %2, %3 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @load_dpas_store
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xf16>, %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
// CHECK: %[[T3:.*]] = xegpu.load_nd %[[T2]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8xf16>
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK-DAG: %[[T4:.*]] = xegpu.dpas %[[T3]], %[[T1]] : vector<8xf16>, vector<16xf16> -> vector<8xf32>
// CHECK-DAG: %[[T5:.*]] = xegpu.create_nd_tdesc %[[ARG2]][%{{.*}}] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK: xegpu.store_nd %[[T4]], %[[T5]]  : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
gpu.module @test {
  gpu.func @load_dpas_store(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
    %c0 = arith.constant 0 : index
    %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %1 = xegpu.load_nd %0  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf16>
    %2 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
    %3 = xegpu.load_nd %2  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xf16>
    %4 = xegpu.dpas %1, %3 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    %5 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %4, %5  : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}


// -----
// CHECK-LABEL: gpu.func @load_dpas_postop_store
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xf16>, %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
// CHECK: %[[T3:.*]] = xegpu.load_nd %[[T2]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8xf16>
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK-DAG: %[[T4:.*]] = xegpu.dpas %[[T3]], %[[T1]] : vector<8xf16>, vector<16xf16> -> vector<8xf32>
// CHECK: %[[T5:.*]] = vector.shape_cast %[[T4]] : vector<8xf32> to vector<8x1xf32>
// CHECK: %[[T6:.*]] = math.exp %[[T5]] {{{.*}}} : vector<8x1xf32>
// CHECK-DAG: %[[T8:.*]] = vector.shape_cast %[[T6]] : vector<8x1xf32> to vector<8xf32>
// CHECK-DAG: %[[T7:.*]] = xegpu.create_nd_tdesc %[[ARG2]][{{.*}}] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK: xegpu.store_nd %[[T8]], %[[T7]]  : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
gpu.module @test {
  gpu.func @load_dpas_postop_store(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<8x16xf32>) {
    %c0 = arith.constant 0 : index
    %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %1 = xegpu.load_nd %0  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : !xegpu.tensor_desc<8x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf16>
    %2 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
    %3 = xegpu.load_nd %2  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xf16>
    %4 = xegpu.dpas %1, %3 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
    %5 = math.exp %4 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8x16xf32>
    %6 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %5, %6 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @create_nd_tdesc_non_memref
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: ui64, %[[ARG1:[0-9a-zA-Z]+]]: ui64, %[[ARG2:[0-9a-zA-Z]+]]: index,
// CHECK-SAME: %[[ARG3:[0-9a-zA-Z]+]]: index, %[[ARG4:[0-9a-zA-Z]+]]: index,
// CHECK-SAME: %[[ARG5:[0-9a-zA-Z]+]]: index, %[[ARG6:[0-9a-zA-Z]+]]: index, %[[ARG7:[0-9a-zA-Z]+]]: index) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][{{.*}}], shape : [%[[ARG2]], %[[ARG3]]], strides : [%[[ARG4]], %[[ARG5]]] : ui64 -> !xegpu.tensor_desc<16x16xf16>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG1]][{{.*}}], shape : [%[[ARG2]], %[[ARG3]]], strides : [%[[ARG4]], %[[ARG5]]] : ui64 -> !xegpu.tensor_desc<16x16xf16>
// CHECK: xegpu.store_nd %[[T1]], %[[T2]]  : vector<16xf16>, !xegpu.tensor_desc<16x16xf16>
gpu.module @test {
  gpu.func @create_nd_tdesc_non_memref(%arg0: ui64, %arg1: ui64, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) {
    %c0 = arith.constant 0 : index
    %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0], shape:[%arg2, %arg3], strides:[%arg4, %arg5] : ui64 -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %1 = xegpu.load_nd %0  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<16x16xf16>
    %2 = xegpu.create_nd_tdesc %arg1[%c0, %c0], shape:[%arg2, %arg3], strides:[%arg4, %arg5] : ui64 -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %1, %2 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----
// TODO: gemm does not use update_nd_offset because of an issue in scf-for distribution.
// CHECK-LABEL: gpu.func @gemm
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<1024x1024xbf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<1024x1024xbf16>, %[[ARG2:[0-9a-zA-Z]+]]: memref<1024x1024xf32>) {
// CHECK-DAG: %[[BLOCK_ID_X:.*]] = gpu.block_id x
// CHECK-DAG: %[[BLOCK_ID_Y:.*]] = gpu.block_id y
// CHECK-DAG: %[[Y_COORD:.*]] = arith.muli %[[BLOCK_ID_Y]], %c16 : index
// CHECK-DAG: %[[X_COORD:.*]] = arith.muli %[[BLOCK_ID_X]], %c8 : index
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG2]][%[[X_COORD]], %[[Y_COORD]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK-NEXT: %[[T3:.*]] = xegpu.load_nd %[[T2]] : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>
// CHECK-NEXT: %[[T4:.*]] = vector.shape_cast %[[T3]] : vector<8xf32> to vector<8x1xf32>
// CHECK: %[[T5:.*]] = scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG4:.*]] = %[[T4]]) -> (vector<8x1xf32>) {
// CHECK-DAG: %[[T10:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%[[K]], %[[Y_COORD]]] : memref<1024x1024xbf16> -> !xegpu.tensor_desc<16x16xbf16>
// CHECK-DAG: %[[T11:.*]] = xegpu.load_nd %[[T10]] <{packed}> : !xegpu.tensor_desc<16x16xbf16> -> vector<16xbf16>
// CHECK-DAG: %[[T12:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[X_COORD]], %[[K]]] : memref<1024x1024xbf16> -> !xegpu.tensor_desc<8x16xbf16>
// CHECK-DAG: %[[T13:.*]] = xegpu.load_nd %[[T12]] : !xegpu.tensor_desc<8x16xbf16> -> vector<8xbf16>
// CHECK-DAG: %[[T14:.*]] = vector.shape_cast %[[ARG4]] : vector<8x1xf32> to vector<8xf32>
// CHECK-NEXT: %[[T15:.*]] = xegpu.dpas %[[T13]], %[[T11]], %[[T14]] : vector<8xbf16>, vector<16xbf16>, vector<8xf32> -> vector<8xf32>
// CHECK-NEXT: %[[T16:.*]] = vector.shape_cast %[[T15]] : vector<8xf32> to vector<8x1xf32>
// CHECK-NEXT: scf.yield %[[T16]] : vector<8x1xf32>
// CHECK-NEXT: }
// CHECK-NEXT: %[[T9:.*]] = vector.shape_cast %[[T5]] : vector<8x1xf32> to vector<8xf32>
// CHECK-NEXT: xegpu.store_nd %[[T9]], %[[T2]] : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
gpu.module @test {
gpu.func @gemm(%arg0: memref<1024x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xf32>){
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %block_id_x = gpu.block_id  x
  %block_id_y = gpu.block_id  y
  %0 = arith.muli %block_id_x, %c8 : index
  %1 = arith.muli %block_id_y, %c16 : index
  %2 = xegpu.create_nd_tdesc %arg2[%0, %1] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  %3 = xegpu.load_nd %2  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xf32>
  %4 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %3) -> (vector<8x16xf32>) {
    %5 = xegpu.create_nd_tdesc %arg0[%0, %arg3] : memref<1024x1024xbf16> -> !xegpu.tensor_desc<8x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %6 = xegpu.create_nd_tdesc %arg1[%arg3, %1] : memref<1024x1024xbf16> -> !xegpu.tensor_desc<16x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
    %7 = xegpu.load_nd %5  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : !xegpu.tensor_desc<8x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<8x16xbf16>
    %8 = xegpu.load_nd %6  {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} : !xegpu.tensor_desc<16x16xbf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<16x16xbf16>
    %9 = xegpu.dpas %7, %8, %arg4 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<8x16xbf16>, vector<16x16xbf16>, vector<8x16xf32> -> vector<8x16xf32>
    scf.yield %9 : vector<8x16xf32>
  } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}
  xegpu.store_nd %4, %2 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @update_nd_offset_1d(
// CHECK: %[[ARG0:[0-9a-zA-Z]+]]: memref<256xf32>) {
// CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<1xf32>
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<256xf32> -> !xegpu.tensor_desc<16xf32>
// CHECK: %[[T1:.*]] = xegpu.update_nd_offset %[[T0]], [%c32] : !xegpu.tensor_desc<16xf32>
// CHECK: xegpu.store_nd %[[CST]], %[[T1]]  : vector<1xf32>, !xegpu.tensor_desc<16xf32>
gpu.module @test {
  gpu.func @update_nd_offset_1d(%arg0: memref<256xf32>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} dense<1.000000e+00> : vector<16xf32>
    %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<256xf32> -> !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    %1 = xegpu.update_nd_offset %0, [%c32] : !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    xegpu.store_nd %cst, %1  : vector<16xf32>, !xegpu.tensor_desc<16xf32, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @update_nd_offset_2d
// CHECK: %[[ARG0:[0-9a-zA-Z]+]]: memref<256x256xf32>) {
// CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<16xf32>
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<256x256xf32> -> !xegpu.tensor_desc<16x16xf32>
// CHECK: %[[T1:.*]] = xegpu.update_nd_offset %[[T0]], [%c32, %c32] : !xegpu.tensor_desc<16x16xf32>
// CHECK: xegpu.store_nd %[[CST]], %[[T1]]  : vector<16xf32>, !xegpu.tensor_desc<16x16xf32>
gpu.module @test {
  gpu.func @update_nd_offset_2d(%arg0: memref<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} dense<1.000000e+00> : vector<16x16xf32>
    %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<256x256xf32> -> !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %1 = xegpu.update_nd_offset %0, [%c32, %c32] : !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %cst, %1  : vector<16x16xf32>, !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @prefetch_2d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<256x256xf16>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<256x256xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK: xegpu.prefetch_nd %[[T0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x16xf16>
gpu.module @test {
  gpu.func @prefetch_2d(%arg0: memref<256x256xf16>) {
    %c0 = arith.constant 0 : index
    %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<256x256xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.prefetch_nd %0 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x16xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----
// Explicitly check that update_nd_offset op's source retain layout when yielded from the warp op (PR150545)
// CHECK-LABEL: gpu.func @check_update_nd_offset_distributed_tensor_desc
// CHECK:      %[[W:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[16] ->
// CHECK-SAME:    (!xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>) {
// CHECK:         %[[T0:.*]] = "some_op"() : () -> !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
// CHECK:         gpu.yield %[[T0]] : !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
// CHECK:       }
// CHECK:      %[[T1:.*]] = builtin.unrealized_conversion_cast %[[W]] :
// CHECK-SAME:    !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> to !xegpu.tensor_desc<16x16xf32> {resolve_simt_type_mismatch}
// CHECK:      xegpu.update_nd_offset %[[T1]], [%{{.*}}] : !xegpu.tensor_desc<16x16xf32>
gpu.module @test {
  gpu.func @check_update_nd_offset_distributed_tensor_desc() {
    %c32 = arith.constant 32 : index
    %cst = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} dense<1.000000e+00> : vector<16x16xf32>
    %0 = "some_op"() : () -> !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %1 = xegpu.update_nd_offset %0, [%c32, %c32] : !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %cst, %1  : vector<16x16xf32>, !xegpu.tensor_desc<16x16xf32, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @prefetch_1d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<256xf16>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<256xf16> -> !xegpu.tensor_desc<16xf16>
// CHECK: xegpu.prefetch_nd %[[T0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16xf16>
gpu.module @test {
  gpu.func @prefetch_1d(%arg0: memref<256xf16>) {
    %c0 = arith.constant 0 : index
    %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<256xf16> -> !xegpu.tensor_desc<16xf16, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    xegpu.prefetch_nd %0 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16xf16, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    gpu.return
  }
}

// -----
// CHECK-LABEL: gpu.func @gpu_barrier({{.*}}) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<256xf16> -> !xegpu.tensor_desc<16xf16>
// CHECK-NEXT: %[[T1:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<16xf16> -> vector<1xf16>
// CHECK-NEXT: gpu.barrier
// CHECK-NEXT: %[[T2:.*]] = xegpu.create_nd_tdesc %{{.*}} : memref<256xf16> -> !xegpu.tensor_desc<16xf16>
// CHECK-NEXT: xegpu.store_nd %[[T1]], %[[T2]] : vector<1xf16>, !xegpu.tensor_desc<16xf16>
gpu.module @test {
  gpu.func @gpu_barrier(%arg0: memref<256xf16>, %arg1: memref<256xf16>) {
    %c0 = arith.constant 0 : index
    %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<256xf16> -> !xegpu.tensor_desc<16xf16, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    %1 = xegpu.load_nd %0  {layout_result_0 = #xegpu.layout<lane_layout = [16], lane_data = [1]>} : !xegpu.tensor_desc<16xf16, #xegpu.layout<lane_layout = [16], lane_data = [1]>> -> vector<16xf16>
    gpu.barrier
    %2 = xegpu.create_nd_tdesc %arg1[%c0] : memref<256xf16> -> !xegpu.tensor_desc<16xf16, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    xegpu.store_nd %1, %2 : vector<16xf16>, !xegpu.tensor_desc<16xf16, #xegpu.layout<lane_layout = [16], lane_data = [1]>>
    gpu.return
  }
}

// -----
// CHECK-PROP-LABEL: func.func @vector_multi_reduction_col_reduce
// CHECK-PROP      :   %[[W:.*]]:2 = gpu.warp_execute_on_lane_0({{.*}})[32] -> (vector<32x2xf32>, vector<2xf32>) {
// CHECK-PROP      :     %[[SOURCE:.*]] = "some_def"() : () -> vector<32x64xf32>
// CHECK-PROP      :     %[[ACC:.*]] = "some_def"() : () -> vector<64xf32>
// CHECK-PROP      :     gpu.yield %[[SOURCE]], %[[ACC]] : vector<32x64xf32>, vector<64xf32>
// CHECK-PROP      :   }
// CHECK-PROP      :   %[[COL0:.*]] = vector.extract_strided_slice %[[W]]#0
// CHECK-PROP-SAME :     {offsets = [0, 0], sizes = [32, 1], strides = [1, 1]} : vector<32x2xf32> to vector<32x1xf32>
// CHECK-PROP      :   %[[COL0CAST:.*]] = vector.shape_cast %[[COL0]] : vector<32x1xf32> to vector<32xf32>
// CHECK-PROP      :   %[[ACC0:.*]] = vector.extract %[[W]]#1[0] : f32 from vector<2xf32>
// CHECK-PROP      :   %[[REDUCE0:.*]] = vector.reduction <add>, %[[COL0CAST]], %[[ACC0]] : vector<32xf32> into f32
// CHECK-PROP      :   %[[COL1:.*]] = vector.extract_strided_slice %[[W]]#0
// CHECK-PROP-SAME :     {offsets = [0, 1], sizes = [32, 1], strides = [1, 1]} : vector<32x2xf32> to vector<32x1xf32>
// CHECK-PROP      :   %[[COL1CAST:.*]] = vector.shape_cast %[[COL1]] : vector<32x1xf32> to vector<32xf32>
// CHECK-PROP      :   %[[ACC1:.*]] = vector.extract %[[W]]#1[1] : f32 from vector<2xf32>
// CHECK-PROP      :   %[[REDUCE1:.*]] = vector.reduction <add>, %[[COL1CAST]], %[[ACC1]] : vector<32xf32> into f32
// CHECK-PROP      :   %[[R:.*]] = vector.from_elements %[[REDUCE0]], %[[REDUCE1]] : vector<2xf32>
// CHECK-PROP      :   return %[[R]] : vector<2xf32>
gpu.module @test {
gpu.func @vector_multi_reduction_col_reduce(%laneid: index) -> vector<2xf32> {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<2xf32>) {
    %0 = "some_def"() : () -> (vector<32x64xf32>)
    %acc = "some_def"() : () -> (vector<64xf32>)
    %1 = vector.multi_reduction <add>, %0, %acc [0] : vector<32x64xf32> to vector<64xf32>
    gpu.yield %1 : vector<64xf32>
  }
  gpu.return %r : vector<2xf32>
}
}

// -----
// CHECK-PROP-LABEL:  func.func @vector_multi_reduction_row_reduce
// CHECK-PROP-DAG:    %[[C16:.*]] = arith.constant 16 : i32
// CHECK-PROP-DAG:    %[[C8:.*]] = arith.constant 8 : i32
// CHECK-PROP-DAG:    %[[C4:.*]] = arith.constant 4 : i32
// CHECK-PROP-DAG:    %[[C2:.*]] = arith.constant 2 : i32
// CHECK-PROP-DAG:    %[[C1:.*]] = arith.constant 1 : i32
// CHECK-PROP-DAG:    %[[C32:.*]] = arith.constant 32 : i32
// CHECK-PROP-DAG:    %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-PROP:        %[[W:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<2x1xf32>) {
// CHECK-PROP:          %[[SRC:.*]] = "some_def"() : () -> vector<2x32xf32>
// CHECK-PROP:          gpu.yield %[[SRC]] : vector<2x32xf32>
// CHECK-PROP:        }
// CHECK-PROP:        %[[T1:.*]] = vector.extract %[[W]][0, 0] : f32 from vector<2x1xf32>
// CHECK-PROP:        %[[SR:.*]], %{{.*}} = gpu.shuffle  xor %[[T1]], %[[C1]], %[[C32]] : f32
// CHECK-PROP:        %[[T2:.*]] = arith.addf %[[T1]], %[[SR]] : f32
// CHECK-PROP:        %[[SR0:.*]], %{{.*}} = gpu.shuffle  xor %[[T2]], %[[C2]], %[[C32]] : f32
// CHECK-PROP:        %[[T3:.*]] = arith.addf %[[T2]], %[[SR0]] : f32
// CHECK-PROP:        %[[SR2:.*]], %{{.*}} = gpu.shuffle  xor %[[T3]], %[[C4]], %[[C32]] : f32
// CHECK-PROP:        %[[T4:.*]] = arith.addf %[[T3]], %[[SR2]] : f32
// CHECK-PROP:        %[[SR4:.*]], %{{.*}} = gpu.shuffle  xor %[[T4]], %[[C8]], %[[C32]] : f32
// CHECK-PROP:        %[[T5:.*]] = arith.addf %[[T4]], %[[SR4]] : f32
// CHECK-PROP:        %[[SR6:.*]], %{{.*}} = gpu.shuffle  xor %[[T5]], %[[C16]], %[[C32]] : f32
// CHECK-PROP:        %[[T6:.*]] = arith.addf %[[T5]], %[[SR6]] : f32
// CHECK-PROP:        %[[R0:.*]] = arith.addf %[[T6]], %[[CST]] : f32
//
// CHECK-PROP:        %[[T8:.*]] = vector.extract %[[W]][1, 0] : f32 from vector<2x1xf32>
// CHECK-PROP:        %[[SR8:.*]], %{{.*}} = gpu.shuffle  xor %[[T8]], %[[C1]], %[[C32]] : f32
// CHECK-PROP:        %[[T9:.*]] = arith.addf %[[T8]], %[[SR8]] : f32
// CHECK-PROP:        %[[SR10:.*]], %{{.*}} = gpu.shuffle  xor %[[T9]], %[[C2]], %[[C32]] : f32
// CHECK-PROP:        %[[T10:.*]] = arith.addf %[[T9]], %[[SR10]] : f32
// CHECK-PROP:        %[[SR12:.*]], %{{.*}} = gpu.shuffle  xor %[[T10]], %[[C4]], %[[C32]] : f32
// CHECK-PROP:        %[[T11:.*]] = arith.addf %[[T10]], %[[SR12]] : f32
// CHECK-PROP:        %[[SR14:.*]], %{{.*}} = gpu.shuffle  xor %[[T11]], %[[C8]], %[[C32]] : f32
// CHECK-PROP:        %[[T12:.*]] = arith.addf %[[T11]], %[[SR14]] : f32
// CHECK-PROP:        %[[SR16:.*]], %{{.*}} = gpu.shuffle  xor %[[T12]], %[[C16]], %[[C32]] : f32
// CHECK-PROP:        %[[T13:.*]] = arith.addf %[[T12]], %[[SR16]] : f32
// CHECK-PROP:        %[[R1:.*]] = arith.addf %[[T13]], %[[CST]] : f32
// CHECK-PROP:        %[[R:.*]] = vector.from_elements %[[R0]], %[[R1]] : vector<2xf32>
// CHECK-PROP:        return %[[R]] : vector<2xf32>
gpu.module @test {
gpu.func @vector_multi_reduction_row_reduce(%laneid: index) -> vector<2xf32> {
  %zero = arith.constant dense<0.0> : vector<2xf32>
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<2xf32>) {
    %0 = "some_def"() : () -> (vector<2x32xf32>)
    %1 = vector.multi_reduction <add>, %0, %zero [1] : vector<2x32xf32> to vector<2xf32>
    gpu.yield %1 : vector<2xf32>
  }
  gpu.return %r : vector<2xf32>
}
}
