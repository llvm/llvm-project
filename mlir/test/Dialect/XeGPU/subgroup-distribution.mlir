// RUN: mlir-opt -xegpu-subgroup-distribute -split-input-file %s | FileCheck %s

// CHECK-LABEL: gpu.func @test_store_nd_1d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<16xf32>) {
// CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<1xf32>
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
// CHECK: xegpu.store_nd %[[CST]], %[[T0]]  : vector<1xf32>, !xegpu.tensor_desc<16xf32>
// CHECK: gpu.return
gpu.module @test {
gpu.func @test_store_nd_1d(%arg0: memref<16xf32>){
  %c0 = arith.constant 0 : index
  %1 = arith.constant dense<1.000000e+00> : vector<16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
  xegpu.store_nd %1, %0 : vector<16xf32>, !xegpu.tensor_desc<16xf32>
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @test_store_nd_2d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<16x16xf16>) {
// CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<16xf16>
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK: xegpu.store_nd %[[CST]], %[[T0]]  : vector<16xf16>, !xegpu.tensor_desc<16x16xf16>
gpu.module @test {
gpu.func @test_store_nd_2d(%arg0: memref<16x16xf16>){
  %c0 = arith.constant 0 : index
  %1 = arith.constant dense<1.000000e+00> : vector<16x16xf16>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  xegpu.store_nd %1, %0 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
  gpu.return
}
}



// -----
// CHECK-LABEL: gpu.func @test_load_nd_1d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<16xf32>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16xf32>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<16xf32> -> vector<1xf32>
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%{{.*}}] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
// CHECK: xegpu.store_nd %[[T1]], %[[T2]]  : vector<1xf32>, !xegpu.tensor_desc<16xf32>
gpu.module @test {
gpu.func @test_load_nd_1d(%arg0: memref<16xf32>, %arg1: memref<16xf32>){
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
  %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
  %2 = xegpu.create_nd_tdesc %arg1[%c0] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
  xegpu.store_nd %1, %2 : vector<16xf32>, !xegpu.tensor_desc<16xf32>
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @test_load_nd_2d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<16x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xf16>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK: xegpu.store_nd %[[T1]], %[[T2]]  : vector<16xf16>, !xegpu.tensor_desc<16x16xf16>
gpu.module @test {
gpu.func @test_load_nd_2d(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>){
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  xegpu.store_nd %1, %2 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @test_load_nd_array_length
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<16x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xf16>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<32xf16>
// CHECK: %[[T2:.*]] = vector.shape_cast %[[T1]] : vector<32xf16> to vector<2x16x1xf16>
// CHECK: %[[T3:.*]] = vector.extract %[[T2]][0] : vector<16x1xf16> from vector<2x16x1xf16>
// CHECK: %[[T4:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK: %[[T5:.*]] = vector.shape_cast %[[T3]] : vector<16x1xf16> to vector<16xf16>
// CHECK: xegpu.store_nd %[[T5]], %[[T4]]  : vector<16xf16>, !xegpu.tensor_desc<16x16xf16>
gpu.module @test {
gpu.func @test_load_nd_array_length(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>){
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<16x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<2x16x16xf16>
  %2 = vector.extract %1[%c0] : vector<16x16xf16> from vector<2x16x16xf16>
  %3 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  xegpu.store_nd %2, %3 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @test_dpas
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: vector<8x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: vector<16x16xf16>, %[[ARG2:[0-9a-zA-Z]+]]: vector<8x16xf32>, %[[ARG3:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK: %[[T1:.*]]:3 = gpu.warp_execute_on_lane_0(%{{.*}})[16] args(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]
// CHECK-SAME: vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32>, memref<8x16xf32>) -> (vector<8x1xf16>, vector<16x1xf16>, vector<8x1xf32>) {
// CHECK: ^bb0(%[[ARG4:[0-9a-zA-Z]+]]: vector<8x16xf16>, %[[ARG5:[0-9a-zA-Z]+]]: vector<16x16xf16>, %[[ARG6:[0-9a-zA-Z]+]]: vector<8x16xf32>, %[[ARG7:[0-9a-zA-Z]+]]: memref<8x16xf32>):
// CHECK:  gpu.yield %[[ARG4]], %[[ARG5]], %[[ARG6]] : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32>
// CHECK: }
// CHECK: %[[T2:.*]] = vector.shape_cast %[[T1]]#0 : vector<8x1xf16> to vector<8xf16>
// CHECK: %[[T3:.*]] = vector.shape_cast %[[T1]]#1 : vector<16x1xf16> to vector<16xf16>
// CHECK: %[[T4:.*]] = vector.shape_cast %[[T1]]#2 : vector<8x1xf32> to vector<8xf32>
// CHECK: %[[T5:.*]] = xegpu.dpas %[[T2]], %[[T3]], %[[T4]] : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
// CHECK: %[[T6:.*]] = xegpu.create_nd_tdesc %[[ARG3]][%{{.*}}] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK: xegpu.store_nd %[[T5]], %[[T6]]  : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
gpu.module @test {
gpu.func @test_dpas(%arg0: vector<8x16xf16>, %arg1: vector<16x16xf16>, %arg3: vector<8x16xf32>, %arg2: memref<8x16xf32>){
  %c0 = arith.constant 0 : index
  %0 = xegpu.dpas %arg0, %arg1, %arg3 : vector<8x16xf16>, vector<16x16xf16>, vector<8x16xf32> -> vector<8x16xf32>
  %3 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %0, %3 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @load_dpas_store
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xf16>, %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
// CHECK: %[[T3:.*]] = xegpu.load_nd %[[T2]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8xf16>
// CHECK: %[[T4:.*]] = xegpu.dpas %[[T3]], %[[T1]] : vector<8xf16>, vector<16xf16> -> vector<8xf32>
// CHECK: %[[T5:.*]] = xegpu.create_nd_tdesc %[[ARG2]][%{{.*}}] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK: xegpu.store_nd %[[T4]], %[[T5]]  : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
gpu.module @test {
gpu.func @load_dpas_store(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg3: memref<8x16xf32>){
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %2 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %3 = xegpu.load_nd %2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %4 = xegpu.dpas %1, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %5 = xegpu.create_nd_tdesc %arg3[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %4, %5 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.return
}
}
