// RUN: mlir-opt -xegpu-subgroup-distribute -cse -split-input-file %s | FileCheck %s

// CHECK-LABEL: gpu.func @store_nd_1d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<16xf32>) {
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<1xf32>
// CHECK-DAG: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
// CHECK: xegpu.store_nd %[[CST]], %[[T0]]  : vector<1xf32>, !xegpu.tensor_desc<16xf32>
// CHECK: gpu.return
gpu.module @test {
gpu.func @store_nd_1d(%arg0: memref<16xf32>){
  %c0 = arith.constant 0 : index
  %1 = arith.constant dense<1.000000e+00> : vector<16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
  xegpu.store_nd %1, %0 : vector<16xf32>, !xegpu.tensor_desc<16xf32>
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
gpu.func @store_nd_2d(%arg0: memref<16x16xf16>){
  %c0 = arith.constant 0 : index
  %1 = arith.constant dense<1.000000e+00> : vector<16x16xf16>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  xegpu.store_nd %1, %0 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
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
gpu.func @load_nd_1d(%arg0: memref<16xf32>, %arg1: memref<16xf32>){
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
  %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<16xf32> -> vector<16xf32>
  %2 = xegpu.create_nd_tdesc %arg1[%c0] : memref<16xf32> -> !xegpu.tensor_desc<16xf32>
  xegpu.store_nd %1, %2 : vector<16xf32>, !xegpu.tensor_desc<16xf32>
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
gpu.func @load_nd_2d(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>){
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  xegpu.store_nd %1, %2 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
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
gpu.func @load_nd_array_length(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>){
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
// CHECK-LABEL: gpu.func @load_dpas_store
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xf16>, %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
// CHECK: %[[T3:.*]] = xegpu.load_nd %[[T2]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8xf16>
// CHECK-DAG: %[[T4:.*]] = xegpu.dpas %[[T3]], %[[T1]] : vector<8xf16>, vector<16xf16> -> vector<8xf32>
// CHECK-DAG: %[[T5:.*]] = xegpu.create_nd_tdesc %[[ARG2]][%{{.*}}] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
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


// -----
// CHECK-LABEL: gpu.func @load_dpas_postop_store
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<8x16xf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<16x16xf16>, %[[ARG2:[0-9a-zA-Z]+]]: memref<8x16xf32>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%{{.*}}] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]] <{packed}> : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
// CHECK: %[[T3:.*]] = xegpu.load_nd %[[T2]]  : !xegpu.tensor_desc<8x16xf16> -> vector<8xf16>
// CHECK-DAG: %[[T4:.*]] = xegpu.dpas %[[T3]], %[[T1]] : vector<8xf16>, vector<16xf16> -> vector<8xf32>
// CHECK: %[[T5:.*]] = vector.shape_cast %[[T4]] : vector<8xf32> to vector<8x1xf32>
// CHECK: %[[T6:.*]] = math.exp %[[T5]] {{{.*}}} : vector<8x1xf32>
// CHECK-DAG: %[[T8:.*]] = vector.shape_cast %[[T6]] : vector<8x1xf32> to vector<8xf32>
// CHECK-DAG: %[[T7:.*]] = xegpu.create_nd_tdesc %[[ARG2]][{{.*}}] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK: xegpu.store_nd %[[T8]], %[[T7]]  : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
gpu.module @test {
gpu.func @load_dpas_postop_store(%arg0: memref<8x16xf16>, %arg1: memref<16x16xf16>, %arg3: memref<8x16xf32>){
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
  %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
  %2 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
  %3 = xegpu.load_nd %2 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %4 = xegpu.dpas %1, %3 : vector<8x16xf16>, vector<16x16xf16> -> vector<8x16xf32>
  %5 = math.exp %4 : vector<8x16xf32>
  %6 = xegpu.create_nd_tdesc %arg3[%c0, %c0] : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %5, %6 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
  gpu.return
}
}

// -----
gpu.module @test {
// CHECK-LABEL: gpu.func @create_nd_tdesc_non_memref
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: ui64, %[[ARG1:[0-9a-zA-Z]+]]: ui64, %[[ARG2:[0-9a-zA-Z]+]]: index,
// CHECK-SAME: %[[ARG3:[0-9a-zA-Z]+]]: index, %[[ARG4:[0-9a-zA-Z]+]]: index,
// CHECK-SAME: %[[ARG5:[0-9a-zA-Z]+]]: index, %[[ARG6:[0-9a-zA-Z]+]]: index, %[[ARG7:[0-9a-zA-Z]+]]: index) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][{{.*}}], [%[[ARG2]], %[[ARG3]]], [%[[ARG4]], %[[ARG5]]] : ui64 -> !xegpu.tensor_desc<16x16xf16>
// CHECK: %[[T1:.*]] = xegpu.load_nd %[[T0]]  : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG1]][{{.*}}], [%[[ARG2]], %[[ARG3]]], [%[[ARG4]], %[[ARG5]]] : ui64 -> !xegpu.tensor_desc<16x16xf16>
// CHECK: xegpu.store_nd %[[T1]], %[[T2]]  : vector<16xf16>, !xegpu.tensor_desc<16x16xf16>
gpu.func @create_nd_tdesc_non_memref(%arg0: ui64, %arg1: ui64,
  %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) {
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0 [%c0, %c0], [%arg2, %arg3], [%arg4, %arg5] : ui64 -> !xegpu.tensor_desc<16x16xf16>
  %1 = xegpu.load_nd %0 : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %2 = xegpu.create_nd_tdesc %arg1[%c0, %c0], [%arg2, %arg3], [%arg4, %arg5] : ui64 -> !xegpu.tensor_desc<16x16xf16>
  xegpu.store_nd %1, %2 : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @gemm_loop
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<1024x1024xbf16>, %[[ARG1:[0-9a-zA-Z]+]]: memref<1024x1024xbf16>, %[[ARG2:[0-9a-zA-Z]+]]: memref<1024x1024xf32>) {
// CHECK: %[[BLOCK_ID_X:.*]] = gpu.block_id x
// CHECK: %[[BLOCK_ID_Y:.*]] = gpu.block_id y
// CHECK: %[[Y_COORD:.*]] = arith.muli %[[BLOCK_ID_Y]], %c16 : index
// CHECK: %[[X_COORD:.*]] = arith.muli %[[BLOCK_ID_X]], %c8 : index
// CHECK: %[[T2:.*]] = xegpu.create_nd_tdesc %[[ARG2]][%[[X_COORD]], %[[Y_COORD]]] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK: %[[T3:.*]] = xegpu.load_nd %[[T2]]  : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>
// CHECK-DAG: %[[C_INIT:.*]] = vector.shape_cast %[[T3]] : vector<8xf32> to vector<8x1xf32>
// CHECK-DAG: %[[B_TILE:.*]] = xegpu.create_nd_tdesc %[[ARG1]][%{{.*}}, %[[Y_COORD]]] : memref<1024x1024xbf16> -> !xegpu.tensor_desc<16x16xbf16>
// CHECK-DAG: %[[A_TILE:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%[[X_COORD]], %{{.*}}] : memref<1024x1024xbf16> -> !xegpu.tensor_desc<8x16xbf16>
// CHECK: %[[T7:.*]]:3 = scf.for {{.*}} iter_args(%[[C_VAL:.*]] = %[[C_INIT]], %[[A_ARG:.*]] = %[[A_TILE]], %[[B_ARG:.*]] = %[[B_TILE]]) -> (vector<8x1xf32>, !xegpu.tensor_desc<8x16xbf16>, !xegpu.tensor_desc<16x16xbf16>) {
// CHECK-DAG: %[[B_NEXT:.*]] = xegpu.update_nd_offset %[[B_ARG]], [{{.*}}] : !xegpu.tensor_desc<16x16xbf16>
// CHECK-DAG: %[[A_NEXT:.*]] = xegpu.update_nd_offset %[[A_ARG]], [{{.*}}] : !xegpu.tensor_desc<8x16xbf16>
// CHECK-DAG: %[[B:.*]] = xegpu.load_nd %[[B_ARG]] <{packed}> : !xegpu.tensor_desc<16x16xbf16> -> vector<16xbf16>
// CHECK-DAG: %[[A:.*]] = xegpu.load_nd %[[A_ARG]]  : !xegpu.tensor_desc<8x16xbf16> -> vector<8xbf16>
// CHECK-DAG: %[[C:.*]] = vector.shape_cast %[[C_VAL]] : vector<8x1xf32> to vector<8xf32>
// CHECK-NEXT: %[[T8:.*]] = xegpu.dpas %[[A]], %[[B]], %[[C]] : vector<8xbf16>, vector<16xbf16>, vector<8xf32> -> vector<8xf32>
// CHECK-NEXT: %[[C_OUT:.*]] = vector.shape_cast %[[T8]] : vector<8xf32> to vector<8x1xf32>
// CHECK-NEXT: scf.yield %[[C_OUT]], %[[A_NEXT]], %[[B_NEXT]] : vector<8x1xf32>, !xegpu.tensor_desc<8x16xbf16>, !xegpu.tensor_desc<16x16xbf16>
// CHECK-NEXT:}
// CHECK-NEXT: %[[C_FINAL:.*]] = vector.shape_cast %[[T7]]#0 : vector<8x1xf32> to vector<8xf32>
// CHECK-NEXT: xegpu.store_nd %[[C_FINAL]], %[[T2]]  : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
gpu.module @test {
gpu.func @gemm_loop(%arg0: memref<1024x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xf32>){
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %0 = gpu.block_id x
  %1 = gpu.block_id y
  %2 = arith.muli %0, %c8 : index
  %3 = arith.muli %1, %c16 : index
  %4 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  %5 = xegpu.load_nd %4 : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
  %7 = xegpu.create_nd_tdesc %arg0[%2, %c0] : memref<1024x1024xbf16> -> !xegpu.tensor_desc<8x16xbf16>
  %8 = xegpu.create_nd_tdesc %arg1[%c0, %3] : memref<1024x1024xbf16> -> !xegpu.tensor_desc<16x16xbf16>
  %6:3 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %5, %arg5 = %7, %arg6 = %8) -> (vector<8x16xf32>, !xegpu.tensor_desc<8x16xbf16>, !xegpu.tensor_desc<16x16xbf16>) {
    %9 = xegpu.load_nd %arg5 : !xegpu.tensor_desc<8x16xbf16> -> vector<8x16xbf16>
    %10 = xegpu.load_nd %arg6 : !xegpu.tensor_desc<16x16xbf16> -> vector<16x16xbf16>
    %12 = xegpu.update_nd_offset %arg5, [%c0, %c16] : !xegpu.tensor_desc<8x16xbf16>
    %13 = xegpu.update_nd_offset %arg6, [%c16, %c0] : !xegpu.tensor_desc<16x16xbf16>
    %11 = xegpu.dpas %9, %10, %arg4 : vector<8x16xbf16>, vector<16x16xbf16>, vector<8x16xf32> -> vector<8x16xf32>
    scf.yield %11, %12, %13 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xbf16>, !xegpu.tensor_desc<16x16xbf16>
  }
  %12 = xegpu.create_nd_tdesc %arg2[%2, %3] : memref<1024x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
  xegpu.store_nd %6#0, %12 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
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
gpu.func @update_nd_offset_1d(%arg0: memref<256xf32>){
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %1 = arith.constant dense<1.000000e+00> : vector<16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<256xf32> -> !xegpu.tensor_desc<16xf32>
  %2 = xegpu.update_nd_offset %0, [%c32] : !xegpu.tensor_desc<16xf32>
  xegpu.store_nd %1, %2 : vector<16xf32>, !xegpu.tensor_desc<16xf32>
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
gpu.func @update_nd_offset_2d(%arg0: memref<256x256xf32>){
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %1 = arith.constant dense<1.000000e+00> : vector<16x16xf32>
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<256x256xf32> -> !xegpu.tensor_desc<16x16xf32>
  %2 = xegpu.update_nd_offset %0, [%c32, %c32] : !xegpu.tensor_desc<16x16xf32>
  xegpu.store_nd %1, %2 : vector<16x16xf32>, !xegpu.tensor_desc<16x16xf32>
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @prefetch_2d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<256x256xf16>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<256x256xf16> -> !xegpu.tensor_desc<16x16xf16>
// CHECK: xegpu.prefetch_nd %[[T0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16x16xf16>
gpu.module @test {
gpu.func @prefetch_2d(%arg0: memref<256x256xf16>){
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<256x256xf16> -> !xegpu.tensor_desc<16x16xf16>
  xegpu.prefetch_nd %0 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<16x16xf16>
  gpu.return
}
}

// -----
// CHECK-LABEL: gpu.func @prefetch_1d
// CHECK: (%[[ARG0:[0-9a-zA-Z]+]]: memref<256xf16>) {
// CHECK: %[[T0:.*]] = xegpu.create_nd_tdesc %[[ARG0]][%{{.*}}] : memref<256xf16> -> !xegpu.tensor_desc<16xf16>
// CHECK: xegpu.prefetch_nd %[[T0]] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}> : !xegpu.tensor_desc<16xf16>
gpu.module @test {
gpu.func @prefetch_1d(%arg0: memref<256xf16>){
  %c0 = arith.constant 0 : index
  %0 = xegpu.create_nd_tdesc %arg0[%c0] : memref<256xf16> -> !xegpu.tensor_desc<16xf16>
  xegpu.prefetch_nd %0 <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>: !xegpu.tensor_desc<16xf16>
  gpu.return
}
}
