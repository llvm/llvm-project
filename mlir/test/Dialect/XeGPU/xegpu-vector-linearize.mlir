// RUN: mlir-opt %s -split-input-file -xegpu-vector-linearize -canonicalize | FileCheck %s

// CHECK-LABEL: test_vector_insert_2d_idx
// CHECK-SAME: (%[[DEST:.*]]: vector<2x8x4xf32>, %[[SRC:.*]]: vector<4xf32>) -> vector<2x8x4xf32>
// CHECK: %[[ARG_DEST:.*]] = vector.shape_cast %[[DEST]] : vector<2x8x4xf32> to vector<64xf32>
// CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG_DEST]], %[[SRC]]
// CHECK: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 64, 65, 66, 67, 16, 17, 18, 19, 20, 21,
// CHECK-SAME: 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
// CHECK-SAME: 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf32>, vector<4xf32>
// CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<64xf32> to vector<2x8x4xf32>
// CHECK: return %[[RES]] : vector<2x8x4xf32>
func.func @test_vector_insert_2d_idx(%arg0: vector<2x8x4xf32>, %arg1: vector<4xf32>) -> vector<2x8x4xf32> {
  %0 = vector.insert %arg1, %arg0[0, 3]: vector<4xf32> into vector<2x8x4xf32>
  return %0 : vector<2x8x4xf32>
}

// -----
// CHECK-LABEL: test_vector_transpose
// CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x8xf32>) -> vector<8x2xf32>
// CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x8xf32> to vector<16xf32>
// CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
// CHECK: [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15] : vector<16xf32>, vector<16xf32>
// CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<16xf32> to vector<8x2xf32>
// CHECK: return %[[RES]] : vector<8x2xf32>
func.func @test_vector_transpose(%arg: vector<2x8xf32>) -> vector<8x2xf32> {
  %0 = vector.transpose %arg, [1, 0] : vector<2x8xf32> to vector<8x2xf32>
  return %0 : vector<8x2xf32>
}

// -----
// CHECK-LABEL: test_vector_transpose_16x16
// CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
// CHECK-62: vector.shuffle
func.func @test_vector_transpose_16x16(%arg: vector<16x16xf32>) -> vector<16x16xf32> {
  %0 = vector.transpose %arg, [1, 0] : vector<16x16xf32> to vector<16x16xf32>
  return %0 : vector<16x16xf32>
}

// -----

// CHECK-LABEL: func.func @test_vector_store_load_4x4_f16
// CHECK-SAME: (%[[ARG0:.*]]: memref<4x4xf16>)
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[LOAD0:.*]] = vector.load %[[ARG0]][%[[C0]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: %[[LOAD1:.*]] = vector.load %[[ARG0]][%[[C1]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: %[[LOAD2:.*]] = vector.load %[[ARG0]][%[[C2]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: %[[LOAD3:.*]] = vector.load %[[ARG0]][%[[C3]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: vector.store %[[LOAD0]], %[[ARG0]][%[[C0]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: vector.store %[[LOAD1]], %[[ARG0]][%[[C1]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: vector.store %[[LOAD2]], %[[ARG0]][%[[C2]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
// CHECK: vector.store %[[LOAD3]], %[[ARG0]][%[[C3]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
func.func @test_vector_store_load_4x4_f16(%buffer: memref<4x4xf16>) {
  %c0 = arith.constant 0 : index
  %0 = vector.load %buffer[%c0, %c0] : memref<4x4xf16>, vector<4x4xf16>
  vector.store %0, %buffer[%c0, %c0] : memref<4x4xf16>, vector<4x4xf16>
  return
}
// -----
// CHECK-LABEL: func.func @test_linearize_index
// CHECK-SAME: (%[[ARG0:.*]]: vector<2x2xindex>, %[[ARG1:.*]]: vector<2x2xi32>) -> vector<2x2xindex>
// CHECK: %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// CHECK: %[[CAST1:.*]] = vector.shape_cast %[[ARG1]] : vector<2x2xi32> to vector<4xi32>
// CHECK: %[[CAST2:.*]] = vector.shape_cast %[[ARG0]] : vector<2x2xindex> to vector<4xindex>
// CHECK: %[[ADDI:.*]] = arith.addi %[[CAST2]], %[[CST]] : vector<4xindex>
// CHECK: %[[INDEX_CAST1:.*]] = arith.index_cast %[[ADDI]] : vector<4xindex> to vector<4xi32>
// CHECK: %[[MULI:.*]] = arith.muli %[[INDEX_CAST1]], %[[CAST1]] : vector<4xi32>
// CHECK: %[[INDEX_CAST2:.*]] = arith.index_cast %[[MULI]] : vector<4xi32> to vector<4xindex>
// CHECK: %[[RESULT:.*]] = vector.shape_cast %[[INDEX_CAST2]] : vector<4xindex> to vector<2x2xindex>
// CHECK: return %[[RESULT]] : vector<2x2xindex>
func.func @test_linearize_index(%arg0: vector<2x2xindex>, %arg1: vector<2x2xi32>) -> vector<2x2xindex> {
  %0 = arith.constant dense<[[0, 1], [2, 3]]> : vector<2x2xindex>
  // Arith and math ops are handled in generic way, check some of them
  %1 = arith.addi %arg0, %0 :  vector<2x2xindex>
  %2 = arith.index_cast %1 : vector<2x2xindex> to vector<2x2xi32>
  %3 = arith.muli %2, %arg1 : vector<2x2xi32>
  %4 = arith.index_cast %3 : vector<2x2xi32> to vector<2x2xindex>
  return %4 : vector<2x2xindex>
}

// -----
// CHECK-LABEL: func.func @broadcast_stretch_at_start
// CHECK-SAME: (%[[ARG0:.*]]: vector<1x4xf32>) -> vector<3x4xf32>
// CHECK: %[[POISON:.*]] = ub.poison : vector<12xf32>
// CHECK: %[[CAST:.*]] = vector.shape_cast %[[ARG0]] : vector<1x4xf32> to vector<4xf32>
// CHECK: %[[SHUFFLE1:.*]] = vector.shuffle %[[POISON]], %[[CAST]] [12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11] : vector<12xf32>, vector<4xf32>
// CHECK: %[[SHUFFLE2:.*]] = vector.shuffle %[[SHUFFLE1]], %[[CAST]] [0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11] : vector<12xf32>, vector<4xf32>
// CHECK: %[[SHUFFLE3:.*]] = vector.shuffle %[[SHUFFLE2]], %[[CAST]] [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15] : vector<12xf32>, vector<4xf32>
// CHECK: %[[RESULT:.*]] = vector.shape_cast %[[SHUFFLE3]] : vector<12xf32> to vector<3x4xf32>
func.func @broadcast_stretch_at_start(%arg0: vector<1x4xf32>) -> vector<3x4xf32> {
  %0 = vector.broadcast %arg0 : vector<1x4xf32> to vector<3x4xf32>
  return %0 : vector<3x4xf32>
}

// -----
// CHECK-LABEL: func.func @broadcast_stretch_at_end
// CHECK-SAME: (%[[ARG0:.*]]: vector<4x1xf32>) -> vector<4x3xf32>
// CHECK: %[[POISON:.*]] = ub.poison : vector<12xf32>
// CHECK: %[[EXTRACT1:.*]] = vector.extract %[[ARG0]][0, 0] : f32 from vector<4x1xf32>
// CHECK: %[[BROADCAST1:.*]] = vector.broadcast %[[EXTRACT1]] : f32 to vector<3xf32>
// CHECK: vector.shuffle
// CHECK: %[[EXTRACT2:.*]] = vector.extract %[[ARG0]][1, 0] : f32 from vector<4x1xf32>
// CHECK: %[[BROADCAST2:.*]] = vector.broadcast %[[EXTRACT2]] : f32 to vector<3xf32>
// CHECK: vector.shuffle
// CHECK: %[[EXTRACT3:.*]] = vector.extract %[[ARG0]][2, 0] : f32 from vector<4x1xf32>
// CHECK: %[[BROADCAST3:.*]] = vector.broadcast %[[EXTRACT3]] : f32 to vector<3xf32>
// CHECK: vector.shuffle
// CHECK: %[[EXTRACT4:.*]] = vector.extract %[[ARG0]][3, 0] : f32 from vector<4x1xf32>
// CHECK: %[[BROADCAST4:.*]] = vector.broadcast %[[EXTRACT4]] : f32 to vector<3xf32>
// CHECK: vector.shuffle
// CHECK: vector.shape_cast {{.*}} : vector<12xf32> to vector<4x3xf32>
func.func @broadcast_stretch_at_end(%arg0: vector<4x1xf32>) -> vector<4x3xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1xf32> to vector<4x3xf32>
  return %0 : vector<4x3xf32>
}

// -----
// CHECK-LABEL: func.func @broadcast_stretch_in_middle
// CHECK-SAME: (%[[ARG0:.*]]: vector<4x1x2xf32>) -> vector<4x3x2xf32>
// CHECK: ub.poison : vector<6xf32>
// CHECK: ub.poison : vector<24xf32>
// CHECK: %[[CAST:.*]] = vector.shape_cast %[[ARG0]] : vector<4x1x2xf32> to vector<8xf32>
// CHECK-COUNT-20: vector.shuffle
// CHECK: vector.shape_cast {{.*}} : vector<24xf32> to vector<4x3x2xf32>
// CHECK-NOT: vector.broadcast
func.func @broadcast_stretch_in_middle(%arg0: vector<4x1x2xf32>) -> vector<4x3x2xf32> {
  %0 = vector.broadcast %arg0 : vector<4x1x2xf32> to vector<4x3x2xf32>
  return %0 : vector<4x3x2xf32>
}

// CHECK-LABEL: func.func @gather_memref_2d
// CHECK-SAME: (%arg0: memref<?x?xf32>, %arg1: vector<2x3xindex>, %arg2: vector<2x3xi1>, %arg3: vector<2x3xf32>) -> vector<2x3xf32> {

// CHECK: %0 = ub.poison : vector<6xf32>
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %1 = vector.shape_cast %arg3 : vector<2x3xf32> to vector<6xf32>

// First shuffle + if ladder for row 0
// CHECK: %2 = vector.shuffle %1, %1 [0, 1, 2]
// CHECK: %3 = vector.extract %arg2[0, 0]
// CHECK: %4 = vector.extract %arg1[0, 0]
// CHECK: %5 = arith.addi %4, %c1
// CHECK: %6 = scf.if %3 -> (vector<3xf32>) {
// CHECK:   %{{.*}} = vector.load %arg0[%c0, %5] : memref<?x?xf32>, vector<1xf32>
// CHECK:   %{{.*}} = vector.extract {{.*}}[0] : f32
// CHECK:   %{{.*}} = vector.insert {{.*}}, %2 [0] : f32 into vector<3xf32>
// CHECK:   scf.yield {{.*}} : vector<3xf32>
// CHECK: } else {
// CHECK:   scf.yield %2 : vector<3xf32>
// CHECK: }

// CHECK: %7 = vector.extract %arg2[0, 1]
// CHECK: %8 = vector.extract %arg1[0, 1]
// CHECK: %9 = arith.addi %8, %c1
// CHECK: %10 = scf.if %7 -> (vector<3xf32>)

// … (similar checks for the rest of row 0, then row 1)

// CHECK: %15 = vector.shuffle %0, %{{.*}} [6, 7, 8, 3, 4, 5]
// CHECK: %16 = vector.shuffle %1, %1 [3, 4, 5]

// Row 1 if ladder checks
// CHECK: %17 = vector.extract %arg2[1, 0]
// CHECK: %18 = vector.extract %arg1[1, 0]
// CHECK: %19 = arith.addi %18, %c1
// CHECK: %20 = scf.if %17 -> (vector<3xf32>)

// … (similar checks for remaining row 1 inserts)

// Final reshuffle and cast
// CHECK: %29 = vector.shuffle %15, %{{.*}} [0, 1, 2, 6, 7, 8]
// CHECK: %30 = vector.shape_cast %29 : vector<6xf32> to vector<2x3xf32>
// CHECK: return %30 : vector<2x3xf32>
func.func @gather_memref_2d(%base: memref<?x?xf32>, %v: vector<2x3xindex>, %mask: vector<2x3xi1>, %pass_thru: vector<2x3xf32>) -> vector<2x3xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = vector.gather %base[%c0, %c1][%v], %mask, %pass_thru : memref<?x?xf32>, vector<2x3xindex>, vector<2x3xi1>, vector<2x3xf32> into vector<2x3xf32>
  return %0 : vector<2x3xf32>
}

// -----
// Check for vector linearization in XeGPU dialect.
// The vector<64xf16> loaded from memory is linearized into 4 vector<8xf16> using vector.shuffle ops.
// The pattern is similar to the one used in test_vector_transpose_16x16 above.
gpu.module @test_kernel {
  // CHECK-LABEL: gpu.func @test_kernel
  gpu.func @test_kernel(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>, %arg2: memref<32x32xf32>) kernel {
    %c24 = arith.constant 24 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] : memref<32x32xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
    %1 = xegpu.load_nd %0  : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<64xf16>
    // CHECK: %[[V1:.*]] = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<64xf16>, vector<64xf16>
    %2 = vector.shape_cast %1 : vector<64xf16> to vector<2x32x1xf16>
    %3 = vector.extract %2[0] : vector<32x1xf16> from vector<2x32x1xf16>
    // CHECK: %[[V2:.*]] = vector.shuffle %[[V1]], %[[V1]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<32xf16>, vector<32xf16>
    %4 = vector.extract_strided_slice %3 {offsets = [0], sizes = [8], strides = [1]} : vector<32x1xf16> to vector<8x1xf16>
    // CHECK: %[[V3:.*]] = vector.shuffle %[[V1]], %[[V1]] [8, 9, 10, 11, 12, 13, 14, 15] : vector<32xf16>, vector<32xf16>
    %5 = vector.extract_strided_slice %3 {offsets = [8], sizes = [8], strides = [1]} : vector<32x1xf16> to vector<8x1xf16>
    // CHECK: %[[V4:.*]] = vector.shuffle %[[V1]], %[[V1]] [16, 17, 18, 19, 20, 21, 22, 23] : vector<32xf16>, vector<32xf16>
    %6 = vector.extract_strided_slice %3 {offsets = [16], sizes = [8], strides = [1]} : vector<32x1xf16> to vector<8x1xf16>
    %7 = xegpu.create_nd_tdesc %arg1[%c0, %c0] : memref<32x32xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
    %8 = xegpu.load_nd %7 <{packed}> : !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<64xf16>
    // CHECK: %[[V5:.*]] = vector.shuffle %{{.*}}, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<64xf16>, vector<64xf16>
    %9 = vector.shape_cast %8 : vector<64xf16> to vector<2x32x1xf16>
    %10 = vector.extract %9[0] : vector<32x1xf16> from vector<2x32x1xf16>
    // CHECK: %[[V6:.*]] = vector.shuffle %[[V5]], %[[V5]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xf16>, vector<32xf16>
    %11 = vector.extract_strided_slice %10 {offsets = [0], sizes = [16], strides = [1]} : vector<32x1xf16> to vector<16x1xf16>
    %12 = vector.extract %9[1] : vector<32x1xf16> from vector<2x32x1xf16>
    // CHECK: %[[V7:.*]] = vector.shuffle %{{.*}}, %{{.*}} [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf16>, vector<64xf16>
    // CHECK: %[[V8:.*]] = vector.shuffle %[[V7]], %[[V7]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xf16>, vector<32xf16>
    %13 = vector.extract_strided_slice %12 {offsets = [0], sizes = [16], strides = [1]} : vector<32x1xf16> to vector<16x1xf16>
    // CHECK: %[[V9:.*]] = vector.shuffle %[[V1]], %[[V1]] [24, 25, 26, 27, 28, 29, 30, 31] : vector<32xf16>, vector<32xf16>
    %14 = vector.extract_strided_slice %3 {offsets = [24], sizes = [8], strides = [1]} : vector<32x1xf16> to vector<8x1xf16>
    %15 = vector.extract %2[1] : vector<32x1xf16> from vector<2x32x1xf16>
    // CHECK: %[[V10:.*]] = vector.shuffle %{{.*}}, %{{.*}} [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf16>, vector<64xf16>
    // CHECK: %[[V11:.*]] = vector.shuffle %[[V10]], %[[V10]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<32xf16>, vector<32xf16>
    %16 = vector.extract_strided_slice %15 {offsets = [0], sizes = [8], strides = [1]} : vector<32x1xf16> to vector<8x1xf16>
    // CHECK: %[[V12:.*]] = vector.shuffle %[[V10]], %[[V10]] [8, 9, 10, 11, 12, 13, 14, 15] : vector<32xf16>, vector<32xf16>
    %17 = vector.extract_strided_slice %15 {offsets = [8], sizes = [8], strides = [1]} : vector<32x1xf16> to vector<8x1xf16>
    // CHECK: %[[V13:.*]] = vector.shuffle %[[V10]], %[[V10]] [16, 17, 18, 19, 20, 21, 22, 23] : vector<32xf16>, vector<32xf16>
    %18 = vector.extract_strided_slice %15 {offsets = [16], sizes = [8], strides = [1]} : vector<32x1xf16> to vector<8x1xf16>
    // CHECK: %[[V14:.*]] = vector.shuffle %[[V5]], %[[V5]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xf16>, vector<32xf16>
    %19 = vector.extract_strided_slice %10 {offsets = [16], sizes = [16], strides = [1]} : vector<32x1xf16> to vector<16x1xf16>
    // CHECK: %[[V15:.*]] = vector.shuffle %[[V7]], %[[V7]] [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xf16>, vector<32xf16>
    %20 = vector.extract_strided_slice %12 {offsets = [16], sizes = [16], strides = [1]} : vector<32x1xf16> to vector<16x1xf16>
    // CHECK: %[[V16:.*]] = vector.shuffle %[[V10]], %[[V10]] [24, 25, 26, 27, 28, 29, 30, 31] : vector<32xf16>, vector<32xf16>
    %21 = vector.extract_strided_slice %15 {offsets = [24], sizes = [8], strides = [1]} : vector<32x1xf16> to vector<8x1xf16>
    // CHECK-NOT: vector.shape_cast
    // CHECK-NOT: vector.extract
    // CHECK-NOT: vector.extract_strided_slice
    %22 = vector.shape_cast %4 : vector<8x1xf16> to vector<8xf16>
    %23 = vector.shape_cast %11 : vector<16x1xf16> to vector<16xf16>
    %24 = xegpu.dpas %22, %23 : vector<8xf16>, vector<16xf16> -> vector<8xf32>
    %25 = vector.shape_cast %13 : vector<16x1xf16> to vector<16xf16>
    %26 = xegpu.dpas %22, %25 : vector<8xf16>, vector<16xf16> -> vector<8xf32>
    %27 = vector.shape_cast %5 : vector<8x1xf16> to vector<8xf16>
    %28 = xegpu.dpas %27, %23 : vector<8xf16>, vector<16xf16> -> vector<8xf32>
    %29 = xegpu.dpas %27, %25 : vector<8xf16>, vector<16xf16> -> vector<8xf32>
    %30 = vector.shape_cast %6 : vector<8x1xf16> to vector<8xf16>
    %31 = xegpu.dpas %30, %23 : vector<8xf16>, vector<16xf16> -> vector<8xf32>
    %32 = xegpu.dpas %30, %25 : vector<8xf16>, vector<16xf16> -> vector<8xf32>
    %33 = vector.shape_cast %14 : vector<8x1xf16> to vector<8xf16>
    %34 = xegpu.dpas %33, %23 : vector<8xf16>, vector<16xf16> -> vector<8xf32>
    %35 = xegpu.dpas %33, %25 : vector<8xf16>, vector<16xf16> -> vector<8xf32>
    %36 = vector.shape_cast %16 : vector<8x1xf16> to vector<8xf16>
    %37 = vector.shape_cast %19 : vector<16x1xf16> to vector<16xf16>
    %38 = xegpu.dpas %36, %37, %24 : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
    %39 = vector.shape_cast %20 : vector<16x1xf16> to vector<16xf16>
    %40 = xegpu.dpas %36, %39, %26 : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
    %41 = vector.shape_cast %17 : vector<8x1xf16> to vector<8xf16>
    %42 = xegpu.dpas %41, %37, %28 : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
    %43 = xegpu.dpas %41, %39, %29 : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
    %44 = vector.shape_cast %18 : vector<8x1xf16> to vector<8xf16>
    %45 = xegpu.dpas %44, %37, %31 : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
    %46 = xegpu.dpas %44, %39, %32 : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
    %47 = vector.shape_cast %21 : vector<8x1xf16> to vector<8xf16>
    %48 = xegpu.dpas %47, %37, %34 : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
    %49 = xegpu.dpas %47, %39, %35 : vector<8xf16>, vector<16xf16>, vector<8xf32> -> vector<8xf32>
    %50 = xegpu.create_nd_tdesc %arg2[%c0, %c0] : memref<32x32xf32> -> !xegpu.tensor_desc<8x16xf32>
    xegpu.store_nd %38, %50  : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
    %51 = xegpu.create_nd_tdesc %arg2[%c0, %c16] : memref<32x32xf32> -> !xegpu.tensor_desc<8x16xf32>
    xegpu.store_nd %40, %51  : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
    %52 = xegpu.create_nd_tdesc %arg2[%c8, %c0] : memref<32x32xf32> -> !xegpu.tensor_desc<8x16xf32>
    xegpu.store_nd %42, %52  : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
    %53 = xegpu.create_nd_tdesc %arg2[%c8, %c16] : memref<32x32xf32> -> !xegpu.tensor_desc<8x16xf32>
    xegpu.store_nd %43, %53  : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
    %54 = xegpu.create_nd_tdesc %arg2[%c16, %c0] : memref<32x32xf32> -> !xegpu.tensor_desc<8x16xf32>
    xegpu.store_nd %45, %54  : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
    %55 = xegpu.create_nd_tdesc %arg2[%c16, %c16] : memref<32x32xf32> -> !xegpu.tensor_desc<8x16xf32>
    xegpu.store_nd %46, %55  : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
    %56 = xegpu.create_nd_tdesc %arg2[%c24, %c0] : memref<32x32xf32> -> !xegpu.tensor_desc<8x16xf32>
    xegpu.store_nd %48, %56  : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
    %57 = xegpu.create_nd_tdesc %arg2[%c24, %c16] : memref<32x32xf32> -> !xegpu.tensor_desc<8x16xf32>
    xegpu.store_nd %49, %57  : vector<8xf32>, !xegpu.tensor_desc<8x16xf32>
    gpu.return
  }
}


