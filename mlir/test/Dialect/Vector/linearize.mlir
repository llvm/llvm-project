// RUN: mlir-opt %s -split-input-file -test-vector-linearize -verify-diagnostics | FileCheck %s 

// CHECK-LABEL: test_linearize
// CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x2xf32>)
func.func @test_linearize(%arg0: vector<2x2xf32>) -> vector<2x2xf32> {

  // CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x2xf32> to vector<4xf32>
  // CHECK: %[[CST:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
  // CHECK: %[[RES:.*]] = vector.shape_cast %[[CST]] : vector<4xf32> to vector<2x2xf32>
  %0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>

  // CHECK: %{{.*}} =  math.sin %[[ARG]] : vector<4xf32>
  %1 = math.sin %arg0 : vector<2x2xf32>

  // CHECK: %{{.*}} = arith.addf %[[ARG]], %[[CST]] : vector<4xf32>
  %2 = arith.addf %arg0, %0 :  vector<2x2xf32>

  // CHECK: return %[[RES]] : vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// -----

// CHECK-LABEL: test_linearize_poison
func.func @test_linearize_poison() -> vector<2x2xf32> {

  // CHECK: %[[POISON:.*]] = ub.poison : vector<4xf32>
  // CHECK: %[[RES:.*]] = vector.shape_cast %[[POISON]] : vector<4xf32> to vector<2x2xf32>
  %0 = ub.poison : vector<2x2xf32>

  // CHECK: return %[[RES]] : vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// -----

// CHECK-LABEL: test_partial_linearize
// CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x2xf32>, %[[ORIG_ARG2:.*]]: vector<4x4xf32>)
func.func @test_partial_linearize(%arg0: vector<2x2xf32>, %arg1: vector<4x4xf32>) -> vector<2x2xf32> {

  // CHECK-DAG: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x2xf32> to vector<4xf32>
  // CHECK-DAG: %[[ARG2:.*]] = vector.shape_cast %[[ORIG_ARG2]] : vector<4x4xf32> to vector<16xf32>
  // CHECK: %[[CST:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
  // CHECK: %[[RES:.*]] = vector.shape_cast %[[CST]] : vector<4xf32> to vector<2x2xf32>
  %0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>

  // CHECK: %[[C2:.*]] = arith.constant dense<{{.*}}> : vector<16xf32>
  %5 = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0,3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 5.0, 6.0]]> : vector<4x4xf32>

  // Arith and math ops are handled in generic way, check some of them
  // CHECK: %[[SIN:.*]] =  math.sin %[[ARG]] : vector<4xf32>
  %1 = math.sin %arg0 : vector<2x2xf32>

  // CHECK: %[[SIN1:.*]] =  math.sin %[[ARG2]] : vector<16xf32>
  %6 = math.sin %arg1 : vector<4x4xf32>

  // CHECK: %{{.*}} = arith.addf %[[ARG]], %[[CST]] : vector<4xf32>
  %2 = arith.addf %arg0, %0 :  vector<2x2xf32>

  // CHECK: %[[ADD2:.*]] = arith.addf %[[ARG2]], %[[C2]] : vector<16xf32>
  %7 = arith.addf %arg1, %5 : vector<4x4xf32>

  // CHECK: return %[[RES]] : vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// -----

// vectorizable operation (arith.mulf) with tensor result types.

// CHECK-LABEL: test_tensor_no_linearize
func.func @test_tensor_no_linearize(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {

    // CHECK: %[[MULF:.*]] = arith.mulf %arg0, %arg1 : tensor<2x2xf32>
    %0 = arith.mulf %arg0, %arg1 : tensor<2x2xf32>

    return %0, %arg0 : tensor<2x2xf32>, tensor<2x2xf32>
}

// -----

// CHECK-LABEL:   func.func @test_scalable_linearize(
// CHECK-SAME:    %[[ARG_0:.*]]: vector<2x[2]xf32>) -> vector<2x[2]xf32> {
func.func @test_scalable_linearize(%arg0: vector<2x[2]xf32>) -> vector<2x[2]xf32> {

  // CHECK:  %[[SC:.*]] = vector.shape_cast %[[ARG_0]] : vector<2x[2]xf32> to vector<[4]xf32>
  // CHECK:  %[[CST:.*]] = arith.constant dense<3.000000e+00> : vector<[4]xf32>
  %0 = arith.constant dense<[[3., 3.], [3., 3.]]> : vector<2x[2]xf32>

  // CHECK: %[[SIN:.*]] = math.sin %[[SC]] : vector<[4]xf32>
  %1 = math.sin %arg0 : vector<2x[2]xf32>

  // CHECK: %[[ADDF:.*]] = arith.addf %[[SIN]], %[[CST]] : vector<[4]xf32>
  %2 = arith.addf %0, %1 : vector<2x[2]xf32>

  // CHECK: %[[RES:.*]] = vector.shape_cast %[[ADDF]] : vector<[4]xf32> to vector<2x[2]xf32>
  // CHECK: return %[[RES]] : vector<2x[2]xf32>
  return %2 : vector<2x[2]xf32>
}

// -----

// CHECK-LABEL:   func.func @test_scalable_no_linearize(
// CHECK-SAME:     %[[VAL_0:.*]]: vector<[2]x[2]xf32>) -> vector<[2]x[2]xf32> {
func.func @test_scalable_no_linearize(%arg0: vector<[2]x[2]xf32>) -> vector<[2]x[2]xf32> {

  // CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : vector<[2]x[2]xf32>
  %0 = arith.constant dense<[[2., 2.], [2., 2.]]> : vector<[2]x[2]xf32>

  // CHECK: %[[SIN:.*]] = math.sin %[[VAL_0]] : vector<[2]x[2]xf32>
  %1 = math.sin %arg0 : vector<[2]x[2]xf32>

  // CHECK: %[[RES:.*]] = arith.addf %[[CST]], %[[SIN]] : vector<[2]x[2]xf32>
  %2 = arith.addf %0, %1 : vector<[2]x[2]xf32>

  // CHECK: return %[[RES]] : vector<[2]x[2]xf32>
  return %2 : vector<[2]x[2]xf32>
}

// -----

// CHECK-LABEL: func.func @test_0d_vector
func.func @test_0d_vector() -> vector<f32> {

  // CHECK: %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<f32>
  %0 = arith.constant dense<0.0> : vector<f32>

  // CHECK: return %[[CST]]
  return %0 : vector<f32>
}

// -----

// CHECK-LABEL: test_extract_strided_slice_1
// CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<4x8xf32>) -> vector<2x2xf32> {
func.func @test_extract_strided_slice_1(%arg0 : vector<4x8xf32>) -> vector<2x2xf32> {

  // CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<4x8xf32> to vector<32xf32>
  // CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
  // CHECK-SAME: [4, 5, 12, 13] : vector<32xf32>, vector<32xf32>
  // CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<4xf32> to vector<2x2xf32>
  // CHECK: return %[[RES]] : vector<2x2xf32
  %0 = vector.extract_strided_slice %arg0 { sizes = [2, 2], strides = [1, 1], offsets = [0, 4]}
     : vector<4x8xf32> to vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// -----

// CHECK-LABEL:   func.func @test_extract_strided_slice_1_scalable(
// CHECK-SAME:    %[[VAL_0:.*]]: vector<4x[8]xf32>) -> vector<2x[8]xf32> {
func.func @test_extract_strided_slice_1_scalable(%arg0: vector<4x[8]xf32>) -> vector<2x[8]xf32> {

  // CHECK-NOT: vector.shuffle
  // CHECK-NOT: vector.shape_cast
  // CHECK: %[[RES:.*]] = vector.extract_strided_slice %[[VAL_0]] {offsets = [1, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x[8]xf32> to vector<2x[8]xf32>
  %0 = vector.extract_strided_slice %arg0 { sizes = [2, 8], strides = [1, 1], offsets = [1, 0] } : vector<4x[8]xf32> to vector<2x[8]xf32>

  // CHECK: return %[[RES]] : vector<2x[8]xf32>
  return %0 : vector<2x[8]xf32>
}

// -----

// CHECK-LABEL: test_extract_strided_slice_2
// CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x8x2xf32>) -> vector<1x4x2xf32> {
func.func @test_extract_strided_slice_2(%arg0 : vector<2x8x2xf32>) -> vector<1x4x2xf32> {

  // CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x8x2xf32> to vector<32xf32>
  // CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
  // CHECK-SAME: [20, 21, 22, 23, 24, 25, 26, 27] : vector<32xf32>, vector<32xf32>
  // CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<8xf32> to vector<1x4x2xf32>
  // CHECK: return %[[RES]] : vector<1x4x2xf32>
  %0 = vector.extract_strided_slice %arg0 { offsets = [1, 2], strides = [1, 1], sizes = [1, 4] }
    : vector<2x8x2xf32> to vector<1x4x2xf32>
  return %0 : vector<1x4x2xf32>
}

// -----

// CHECK-LABEL: test_vector_shuffle
// CHECK-SAME: (%[[ORIG_ARG0:.*]]: vector<4x2xf32>, %[[ORIG_ARG1:.*]]: vector<4x2xf32>) -> vector<8x2xf32> {
func.func @test_vector_shuffle(%arg0: vector<4x2xf32>, %arg1: vector<4x2xf32>) -> vector<8x2xf32> {

  // CHECK-DAG: %[[ARG0:.*]] = vector.shape_cast %[[ORIG_ARG0]] : vector<4x2xf32> to vector<8xf32>
  // CHECK-DAG: %[[ARG1:.*]] = vector.shape_cast %[[ORIG_ARG1]] : vector<4x2xf32> to vector<8xf32>
  // CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG0]], %[[ARG1]]
  // CHECK-SAME: [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<16xf32> to vector<8x2xf32>
  // CHECK: return %[[RES]] : vector<8x2xf32>
  %0 = vector.shuffle %arg0, %arg1 [0, 4, 1, 5, 2, 6, 3, 7] : vector<4x2xf32>, vector<4x2xf32>
  return %0 : vector<8x2xf32>
}

// -----

// CHECK-LABEL: test_vector_extract
// CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x8x2xf32>) -> vector<8x2xf32> {
func.func @test_vector_extract(%arg0: vector<2x8x2xf32>) -> vector<8x2xf32> {

  // CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x8x2xf32> to vector<32xf32>
  // CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
  // CHECK-SAME: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xf32>, vector<32xf32>
  // CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<16xf32> to vector<8x2xf32>
  // CHECK: return %[[RES]] : vector<8x2xf32>
  %0 = vector.extract %arg0[1]: vector<8x2xf32> from vector<2x8x2xf32>
  return %0 : vector<8x2xf32>
}

// -----

// CHECK-LABEL:   func.func @test_vector_extract_scalable(
// CHECK-SAME:    %[[VAL_0:.*]]: vector<2x8x[2]xf32>) -> vector<8x[2]xf32> {
func.func @test_vector_extract_scalable(%arg0: vector<2x8x[2]xf32>) -> vector<8x[2]xf32> {

  // CHECK-NOT: vector.shuffle
  // CHECK-NOT: vector.shape_cast
  // CHECK: %[[RES:.*]] = vector.extract %[[VAL_0]][1] : vector<8x[2]xf32> from vector<2x8x[2]xf32>
  %0 = vector.extract %arg0[1]: vector<8x[2]xf32> from vector<2x8x[2]xf32>

  // CHECK: return %[[RES]] : vector<8x[2]xf32>
  return %0 : vector<8x[2]xf32>
}

// -----

// CHECK-LABEL: test_vector_insert
// CHECK-SAME: (%[[DEST:.*]]: vector<2x8x4xf32>, %[[SRC:.*]]: vector<8x4xf32>) -> vector<2x8x4xf32> {
func.func @test_vector_insert(%arg0: vector<2x8x4xf32>, %arg1: vector<8x4xf32>) -> vector<2x8x4xf32> {

  // CHECK-DAG: %[[ARG_SRC:.*]] = vector.shape_cast %[[SRC]] : vector<8x4xf32> to vector<32xf32>
  // CHECK-DAG: %[[ARG_DEST:.*]] = vector.shape_cast %[[DEST]] : vector<2x8x4xf32> to vector<64xf32>
  // CHECK: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG_DEST]], %[[ARG_SRC]]
  // CHECK-SAME: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
  // CHECK-SAME: 88, 89, 90, 91, 92, 93, 94, 95, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
  // CHECK-SAME: 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf32>, vector<32xf32>
  // CHECK: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<64xf32> to vector<2x8x4xf32>
  // CHECK: return %[[RES]] : vector<2x8x4xf32>
  %0 = vector.insert %arg1, %arg0[0]: vector<8x4xf32> into vector<2x8x4xf32>
  return %0 : vector<2x8x4xf32>
}

// -----

// CHECK-LABEL:   func.func @test_vector_insert_scalable(
// CHECK-SAME:    %[[VAL_0:.*]]: vector<2x8x[4]xf32>, %[[VAL_1:.*]]: vector<8x[4]xf32>) -> vector<2x8x[4]xf32> {
func.func @test_vector_insert_scalable(%arg0: vector<2x8x[4]xf32>, %arg1: vector<8x[4]xf32>) -> vector<2x8x[4]xf32> {

  // CHECK-NOT: vector.shuffle
  // CHECK-NOT: vector.shape_cast
  // CHECK: %[[RES:.*]] = vector.insert %[[VAL_1]], %[[VAL_0]] [0] : vector<8x[4]xf32> into vector<2x8x[4]xf32>

  %0 = vector.insert %arg1, %arg0[0]: vector<8x[4]xf32> into vector<2x8x[4]xf32>
  // CHECK: return %[[RES]] : vector<2x8x[4]xf32>
  return %0 : vector<2x8x[4]xf32>
}

// -----

// CHECK-LABEL: test_vector_extract_scalar
func.func @test_vector_extract_scalar(%idx : index) {
  %cst = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>

  // CHECK-NOT: vector.shuffle
  // CHECK:     vector.extract
  // CHECK-NOT: vector.shuffle
  %0 = vector.extract %cst[%idx] : i32 from vector<4xi32>
  return
}

// -----

// CHECK-LABEL: test_vector_bitcast
// CHECK-SAME: %[[ARG_0:.*]]: vector<4x4xf32>
func.func @test_vector_bitcast(%arg0: vector<4x4xf32>) -> vector<4x8xf16> {

  // CHECK: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<4x4xf32> to vector<16xf32>
  // CHECK: %[[BITCAST:.*]] = vector.bitcast %[[DOWNCAST]] : vector<16xf32> to vector<32xf16>
  // CHECK: %[[UPCAST:.*]] = vector.shape_cast %[[BITCAST]] : vector<32xf16> to vector<4x8xf16>
  %1 = vector.bitcast %arg0 : vector<4x4xf32> to vector<4x8xf16>
  return %1 : vector<4x8xf16>
}

// -----

// CHECK-LABEL: test_vector_bitcast
// CHECK-SAME: %[[ARG_0:.*]]: vector<4x2xf32>
func.func @test_vector_bitcast(%arg0: vector<4x2xf32>) -> vector<4x4xf16> {

  // CHECK: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<4x2xf32> to vector<8xf32>
  // CHECK: %[[BITCAST:.*]] = vector.bitcast %[[DOWNCAST]] : vector<8xf32> to vector<16xf16>
  // CHECK: %[[UPCAST:.*]] = vector.shape_cast %[[BITCAST]] : vector<16xf16> to vector<4x4xf16>
  %1 = vector.bitcast %arg0 : vector<4x2xf32> to vector<4x4xf16>
  return %1 : vector<4x4xf16>
}

// -----

// CHECK-LABEL: test_vector_bitcast
// CHECK-SAME: %[[ARG_0:.*]]: vector<4x[2]xf32>
func.func @test_vector_bitcast(%arg0: vector<4x[2]xf32>) -> vector<4x[4]xf16> {

  // CHECK: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<4x[2]xf32> to vector<[8]xf32>
  // CHECK: %[[BITCAST:.*]] = vector.bitcast %[[DOWNCAST]] : vector<[8]xf32> to vector<[16]xf16>
  // CHECK: %[[UPCAST:.*]] = vector.shape_cast %[[BITCAST]] : vector<[16]xf16> to vector<4x[4]xf16>
  %1 = vector.bitcast %arg0 : vector<4x[2]xf32> to vector<4x[4]xf16>
  return %1 : vector<4x[4]xf16>
}

// -----

// CHECK-LABEL: test_vector_bitcast
// CHECK-SAME: %[[ARG_0:.*]]: vector<[4]x2xf32>
func.func @test_vector_bitcast(%arg0: vector<[4]x2xf32>) -> vector<[4]x4xf16> {

  // CHECK: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<[4]x2xf32> to vector<[8]xf32>
  // CHECK: %[[BITCAST:.*]] = vector.bitcast %[[DOWNCAST]] : vector<[8]xf32> to vector<[16]xf16>
  // CHECK: %[[UPCAST:.*]] = vector.shape_cast %[[BITCAST]] : vector<[16]xf16> to vector<[4]x4xf16>
  %1 = vector.bitcast %arg0 : vector<[4]x2xf32> to vector<[4]x4xf16>
  return %1 : vector<[4]x4xf16>
}

// -----

// CHECK-LABEL: linearize_vector_splat
// CHECK-SAME: (%[[ARG:.*]]: i32) -> vector<4x2xi32>
func.func @linearize_vector_splat(%arg0: i32) -> vector<4x2xi32> {

  // CHECK: %[[SPLAT:.*]] = vector.splat %[[ARG]] : vector<8xi32>
  // CHECK: %[[CAST:.*]] = vector.shape_cast %[[SPLAT]] : vector<8xi32> to vector<4x2xi32>
  // CHECK: return %[[CAST]] : vector<4x2xi32>
  %0 = vector.splat %arg0 : vector<4x2xi32>
  return %0 : vector<4x2xi32>
}

// -----

// CHECK-LABEL: linearize_scalable_vector_splat
// CHECK-SAME: (%[[ARG:.*]]: i32) -> vector<4x[2]xi32>
func.func @linearize_scalable_vector_splat(%arg0: i32) -> vector<4x[2]xi32> {

  // CHECK: %[[SPLAT:.*]] = vector.splat %[[ARG]] : vector<[8]xi32>
  // CHECK: %[[CAST:.*]] = vector.shape_cast %[[SPLAT]] : vector<[8]xi32> to vector<4x[2]xi32>
  // CHECK: return %[[CAST]] : vector<4x[2]xi32>
  %0 = vector.splat %arg0 : vector<4x[2]xi32>
  return %0 : vector<4x[2]xi32>
}
