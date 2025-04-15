// RUN: mlir-opt %s -split-input-file -test-vector-linearize -verify-diagnostics | FileCheck %s --check-prefixes=ALL,DEFAULT
// RUN: mlir-opt %s -split-input-file -test-vector-linearize=target-vector-bitwidth=128  -verify-diagnostics | FileCheck %s --check-prefixes=ALL,BW-128
// RUN: mlir-opt %s -split-input-file -test-vector-linearize=target-vector-bitwidth=0 | FileCheck %s --check-prefixes=ALL,BW-0

// ALL-LABEL: test_linearize
// ALL-SAME: (%[[ORIG_ARG:.*]]: vector<2x2xf32>)
func.func @test_linearize(%arg0: vector<2x2xf32>) -> vector<2x2xf32> {
  // DEFAULT: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x2xf32> to vector<4xf32>
  // DEFAULT: %[[CST:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
  // DEFAULT: %[[RES:.*]] = vector.shape_cast %[[CST]] : vector<4xf32> to vector<2x2xf32>

  // BW-128: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x2xf32> to vector<4xf32>
  // BW-128: %[[CST:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
  // BW-128: %[[RES:.*]] = vector.shape_cast %[[CST]] : vector<4xf32> to vector<2x2xf32>

  // BW-0: %[[RES:.*]] = arith.constant dense<{{.*}}> : vector<2x2xf32>
  %0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>

  // DEFAULT: %{{.*}} =  math.sin %[[ARG]] : vector<4xf32>
  // BW-128: %{{.*}} =  math.sin %[[ARG]] : vector<4xf32>
  // BW-0: %{{.*}} =  math.sin %{{.*}} : vector<2x2xf32>
  %1 = math.sin %arg0 : vector<2x2xf32>

  // DEFAULT: %{{.*}} = arith.addf %[[ARG]], %[[CST]] : vector<4xf32>
  // BW-128: %{{.*}} = arith.addf %[[ARG]], %[[CST]] : vector<4xf32>
  // BW-0: %{{.*}} = arith.addf %{{.*}} : vector<2x2xf32>
  %2 = arith.addf %arg0, %0 :  vector<2x2xf32>

  // ALL: return %[[RES]] : vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// -----

// ALL-LABEL: test_linearize_poison
func.func @test_linearize_poison() -> vector<2x2xf32> {
  // DEFAULT: %[[POISON:.*]] = ub.poison : vector<4xf32>
  // DEFAULT: %[[RES:.*]] = vector.shape_cast %[[POISON]] : vector<4xf32> to vector<2x2xf32>

  // BW-128: %[[POISON:.*]] = ub.poison : vector<4xf32>
  // BW-128: %[[RES:.*]] = vector.shape_cast %[[POISON]] : vector<4xf32> to vector<2x2xf32>

  // BW-0: %[[RES:.*]] = ub.poison : vector<2x2xf32>
  %0 = ub.poison : vector<2x2xf32>
  // ALL: return %[[RES]] : vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// -----

// ALL-LABEL: test_partial_linearize
// ALL-SAME: (%[[ORIG_ARG:.*]]: vector<2x2xf32>, %[[ORIG_ARG2:.*]]: vector<4x4xf32>)
func.func @test_partial_linearize(%arg0: vector<2x2xf32>, %arg1: vector<4x4xf32>) -> vector<2x2xf32> {
  // DEFAULT-DAG: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x2xf32> to vector<4xf32>
  // DEFAULT-DAG: %[[ARG2:.*]] = vector.shape_cast %[[ORIG_ARG2]] : vector<4x4xf32> to vector<16xf32>
  // DEFAULT: %[[CST:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
  // DEFAULT: %[[RES:.*]] = vector.shape_cast %[[CST]] : vector<4xf32> to vector<2x2xf32>

  // BW-128: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x2xf32> to vector<4xf32>
  // BW-128: %[[CST:.*]] = arith.constant dense<{{.*}}> : vector<4xf32>
  // BW-128: %[[RES:.*]] = vector.shape_cast %[[CST]] : vector<4xf32> to vector<2x2xf32>

  // BW-0: %[[RES:.*]] = arith.constant dense<{{.*}}> : vector<2x2xf32>
  %0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>

  // DEFAULT: %[[C2:.*]] = arith.constant dense<{{.*}}> : vector<16xf32>
  // BW-128: %[[C2:.*]] = arith.constant dense<{{.*}}> : vector<4x4xf32>
  // BW-0: %[[C2:.*]] = arith.constant dense<{{.*}}> : vector<4x4xf32>
  %5 = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0,3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 5.0, 6.0]]> : vector<4x4xf32>

  // Arith and math ops are handled in generic way, check some of them
  // DEFAULT: %[[SIN:.*]] =  math.sin %[[ARG]] : vector<4xf32>
  // BW-128: %[[SIN:.*]] =  math.sin %[[ARG]] : vector<4xf32>
  // BW-0: %[[SIN:.*]] =  math.sin %[[ORIG_ARG]] : vector<2x2xf32>
  %1 = math.sin %arg0 : vector<2x2xf32>

  // DEFAULT: %[[SIN1:.*]] =  math.sin %[[ARG2]] : vector<16xf32>
  // BW-128: %[[SIN1:.*]] =  math.sin %[[ORIG_ARG2]] : vector<4x4xf32>
  // BW-0: %[[SIN1:.*]] =  math.sin %[[ORIG_ARG2]] : vector<4x4xf32>
  %6 = math.sin %arg1 : vector<4x4xf32>

  // DEFAULT: %{{.*}} = arith.addf %[[ARG]], %[[CST]] : vector<4xf32>
  // BW-128: %{{.*}} = arith.addf %[[ARG]], %[[CST]] : vector<4xf32>
  // BW-0: %{{.*}} = arith.addf %{{.*}} : vector<2x2xf32>
  %2 = arith.addf %arg0, %0 :  vector<2x2xf32>

  // DEFAULT: %[[ADD2:.*]] = arith.addf %[[ARG2]], %[[C2]] : vector<16xf32>
  // BW-128: %[[ADD2:.*]] = arith.addf %[[ORIG_ARG2]], %[[C2]] : vector<4x4xf32>
  // BW-0: %[[ADD2:.*]] = arith.addf %[[ORIG_ARG2]], %[[C2]] : vector<4x4xf32>
  %7 = arith.addf %arg1, %5 : vector<4x4xf32>

  // ALL: return %[[RES]] : vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// -----

// ALL-LABEL: test_index_no_linearize
func.func @test_index_no_linearize(%arg0: vector<2x2xindex>, %arg1: vector<2x2xindex>) -> vector<2x2xindex> {
    // ALL: %[[ADD:.*]] = arith.addi {{.*}} : vector<2x2xindex>
    %0 = arith.addi %arg0, %arg1 : vector<2x2xindex>
    return %0 : vector<2x2xindex>
}

// -----

// vectorizable operation (arith.mulf) with tensor result types.

// ALL-LABEL: test_tensor_no_linearize
func.func @test_tensor_no_linearize(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    // ALL: %[[MULF:.*]] = arith.mulf %arg0, %arg1 : tensor<2x2xf32>
    %0 = arith.mulf %arg0, %arg1 : tensor<2x2xf32>

    return %0, %arg0 : tensor<2x2xf32>, tensor<2x2xf32>
}

// -----

// ALL-LABEL:   func.func @test_scalable_linearize(
// ALL-SAME:    %[[ARG_0:.*]]: vector<2x[2]xf32>) -> vector<2x[2]xf32> {
func.func @test_scalable_linearize(%arg0: vector<2x[2]xf32>) -> vector<2x[2]xf32> {
  // DEFAULT:  %[[SC:.*]] = vector.shape_cast %[[ARG_0]] : vector<2x[2]xf32> to vector<[4]xf32>
  // DEFAULT:  %[[CST:.*]] = arith.constant dense<3.000000e+00> : vector<[4]xf32>
  // BW-128:  %[[SC:.*]] = vector.shape_cast %[[ARG_0]] : vector<2x[2]xf32> to vector<[4]xf32>
  // BW-128:  %[[CST:.*]] = arith.constant dense<3.000000e+00> : vector<[4]xf32>
  // BW-0:  %[[CST:.*]] = arith.constant dense<3.000000e+00> : vector<2x[2]xf32>
  %0 = arith.constant dense<[[3., 3.], [3., 3.]]> : vector<2x[2]xf32>

  // DEFAULT: %[[SIN:.*]] = math.sin %[[SC]] : vector<[4]xf32>
  // BW-128: %[[SIN:.*]] = math.sin %[[SC]] : vector<[4]xf32>
  // BW-0: %[[SIN:.*]] = math.sin %[[ARG_0]] : vector<2x[2]xf32>
  %1 = math.sin %arg0 : vector<2x[2]xf32>

  // DEFAULT: %[[ADDF:.*]] = arith.addf %[[SIN]], %[[CST]] : vector<[4]xf32>
  // BW-128: %[[ADDF:.*]] = arith.addf %[[SIN]], %[[CST]] : vector<[4]xf32>
  // BW-0: %[[RES:.*]] = arith.addf %[[CST]], %[[SIN]] : vector<2x[2]xf32>
  %2 = arith.addf %0, %1 : vector<2x[2]xf32>

  // DEFAULT: %[[RES:.*]] = vector.shape_cast %[[ADDF]] : vector<[4]xf32> to vector<2x[2]xf32>
  // BW-128: %[[RES:.*]] = vector.shape_cast %[[ADDF]] : vector<[4]xf32> to vector<2x[2]xf32>
  // ALL: return %[[RES]] : vector<2x[2]xf32>
  return %2 : vector<2x[2]xf32>
}

// -----

// ALL-LABEL:   func.func @test_scalable_no_linearize(
// ALL-SAME:     %[[VAL_0:.*]]: vector<[2]x[2]xf32>) -> vector<[2]x[2]xf32> {
func.func @test_scalable_no_linearize(%arg0: vector<[2]x[2]xf32>) -> vector<[2]x[2]xf32> {
  // ALL: %[[CST:.*]] = arith.constant dense<2.000000e+00> : vector<[2]x[2]xf32>
  %0 = arith.constant dense<[[2., 2.], [2., 2.]]> : vector<[2]x[2]xf32>

  // ALL: %[[SIN:.*]] = math.sin %[[VAL_0]] : vector<[2]x[2]xf32>
  %1 = math.sin %arg0 : vector<[2]x[2]xf32>

  // ALL: %[[RES:.*]] = arith.addf %[[CST]], %[[SIN]] : vector<[2]x[2]xf32>
  %2 = arith.addf %0, %1 : vector<[2]x[2]xf32>

  // ALL: return %[[RES]] : vector<[2]x[2]xf32>
  return %2 : vector<[2]x[2]xf32>
}

// -----

// ALL-LABEL: func.func @test_0d_vector
func.func @test_0d_vector() -> vector<f32> {
  // ALL: %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<f32>
  %0 = arith.constant dense<0.0> : vector<f32>
  // ALL: return %[[CST]]
  return %0 : vector<f32>
}

// -----
// ALL-LABEL: test_extract_strided_slice_1
// ALL-SAME: (%[[ORIG_ARG:.*]]: vector<4x8xf32>) -> vector<2x2xf32> {
func.func @test_extract_strided_slice_1(%arg0 : vector<4x8xf32>) -> vector<2x2xf32> {
  // DEFAULT: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<4x8xf32> to vector<32xf32>
  // DEFAULT: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
  // DEFAULT-SAME: [4, 5, 12, 13] : vector<32xf32>, vector<32xf32>
  // DEFAULT: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<4xf32> to vector<2x2xf32>
  // DEFAULT: return %[[RES]] : vector<2x2xf32

  // BW-128: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<4x8xf32> to vector<32xf32>
  // BW-128: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
  // BW-128-SAME: [4, 5, 12, 13] : vector<32xf32>, vector<32xf32>
  // BW-128: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<4xf32> to vector<2x2xf32>
  // BW-128: return %[[RES]] : vector<2x2xf32>

  // BW-0: %[[RES:.*]] = vector.extract_strided_slice %[[ARG:.*]] {offsets = [0, 4], sizes = [2, 2], strides = [1, 1]} : vector<4x8xf32> to vector<2x2xf32>
  // BW-0: return %[[RES]] : vector<2x2xf32>
  %0 = vector.extract_strided_slice %arg0 { sizes = [2, 2], strides = [1, 1], offsets = [0, 4]}
     : vector<4x8xf32> to vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// ALL-LABEL:   func.func @test_extract_strided_slice_1_scalable(
// ALL-SAME:    %[[VAL_0:.*]]: vector<4x[8]xf32>) -> vector<2x[8]xf32> {
func.func @test_extract_strided_slice_1_scalable(%arg0: vector<4x[8]xf32>) -> vector<2x[8]xf32> {
  // ALL-NOT: vector.shuffle
  // ALL-NOT: vector.shape_cast
  // ALL: %[[RES:.*]] = vector.extract_strided_slice %[[VAL_0]] {offsets = [1, 0], sizes = [2, 8], strides = [1, 1]} : vector<4x[8]xf32> to vector<2x[8]xf32>
  %0 = vector.extract_strided_slice %arg0 { sizes = [2, 8], strides = [1, 1], offsets = [1, 0] } : vector<4x[8]xf32> to vector<2x[8]xf32>
  // ALL: return %[[RES]] : vector<2x[8]xf32>
  return %0 : vector<2x[8]xf32>
}

// -----
// ALL-LABEL: test_extract_strided_slice_2
// ALL-SAME: (%[[ORIG_ARG:.*]]: vector<2x8x2xf32>) -> vector<1x4x2xf32> {
func.func @test_extract_strided_slice_2(%arg0 : vector<2x8x2xf32>) -> vector<1x4x2xf32> {
  // DEFAULT: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x8x2xf32> to vector<32xf32>
  // DEFAULT: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
  // DEFAULT-SAME: [20, 21, 22, 23, 24, 25, 26, 27] : vector<32xf32>, vector<32xf32>
  // DEFAULT: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<8xf32> to vector<1x4x2xf32>
  // DEFAULT: return %[[RES]] : vector<1x4x2xf32>

  // BW-128: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x8x2xf32> to vector<32xf32>
  // BW-128: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
  // BW-128-SAME: [20, 21, 22, 23, 24, 25, 26, 27] : vector<32xf32>, vector<32xf32>
  // BW-128: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<8xf32> to vector<1x4x2xf32>
  // BW-128: return %[[RES]] : vector<1x4x2xf32>

  // BW-0: %[[RES:.*]] = vector.extract_strided_slice %[[ORIG_ARG]] {offsets = [1, 2], sizes = [1, 4], strides = [1, 1]} : vector<2x8x2xf32> to vector<1x4x2xf32>
  // BW-0: return %[[RES]] : vector<1x4x2xf32>
  %0 = vector.extract_strided_slice %arg0 { offsets = [1, 2], strides = [1, 1], sizes = [1, 4] }
    : vector<2x8x2xf32> to vector<1x4x2xf32>
  return %0 : vector<1x4x2xf32>
}

// -----
// ALL-LABEL: test_vector_shuffle
// ALL-SAME: (%[[ORIG_ARG0:.*]]: vector<4x2xf32>, %[[ORIG_ARG1:.*]]: vector<4x2xf32>) -> vector<8x2xf32> {
func.func @test_vector_shuffle(%arg0: vector<4x2xf32>, %arg1: vector<4x2xf32>) -> vector<8x2xf32> {
  // DEFAULT-DAG: %[[ARG0:.*]] = vector.shape_cast %[[ORIG_ARG0]] : vector<4x2xf32> to vector<8xf32>
  // DEFAULT-DAG: %[[ARG1:.*]] = vector.shape_cast %[[ORIG_ARG1]] : vector<4x2xf32> to vector<8xf32>
  // DEFAULT: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG0]], %[[ARG1]]
  // DEFAULT-SAME: [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
  // DEFAULT: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<16xf32> to vector<8x2xf32>
  // DEFAULT: return %[[RES]] : vector<8x2xf32>

  // BW-128-DAG: %[[ARG0:.*]] = vector.shape_cast %[[ORIG_ARG0]] : vector<4x2xf32> to vector<8xf32>
  // BW-128-DAG: %[[ARG1:.*]] = vector.shape_cast %[[ORIG_ARG1]] : vector<4x2xf32> to vector<8xf32>
  // BW-128: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG0]], %[[ARG1]]
  // BW-128-SAME: [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
  // BW-128: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<16xf32> to vector<8x2xf32>
  // BW-128: return %[[RES]] : vector<8x2xf32>

  // BW-0: %[[RES:.*]] = vector.shuffle %[[ORIG_ARG0]], %[[ORIG_ARG1]] [0, 4, 1, 5, 2, 6, 3, 7] : vector<4x2xf32>, vector<4x2xf32>
  // BW-0: return %[[RES]] : vector<8x2xf32>
  %0 = vector.shuffle %arg0, %arg1 [0, 4, 1, 5, 2, 6, 3, 7] : vector<4x2xf32>, vector<4x2xf32>
  return %0 : vector<8x2xf32>
}

// -----
// ALL-LABEL: test_vector_extract
// ALL-SAME: (%[[ORIG_ARG:.*]]: vector<2x8x2xf32>) -> vector<8x2xf32> {
func.func @test_vector_extract(%arg0: vector<2x8x2xf32>) -> vector<8x2xf32> {
  // DEFAULT: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x8x2xf32> to vector<32xf32>
  // DEFAULT: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
  // DEFAULT-SAME: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xf32>, vector<32xf32>
  // DEFAULT: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<16xf32> to vector<8x2xf32>
  // DEFAULT: return %[[RES]] : vector<8x2xf32>

  // BW-128: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x8x2xf32> to vector<32xf32>
  // BW-128: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG]], %[[ARG]]
  // BW-128-SAME: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xf32>, vector<32xf32>
  // BW-128: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<16xf32> to vector<8x2xf32>
  // BW-128: return %[[RES]] : vector<8x2xf32>

  // BW-0: %[[RES:.*]] = vector.extract %[[ORIG_ARG]][1] : vector<8x2xf32> from vector<2x8x2xf32>
  // BW-0: return %[[RES]] : vector<8x2xf32>
  %0 = vector.extract %arg0[1]: vector<8x2xf32> from vector<2x8x2xf32>
  return %0 : vector<8x2xf32>
}

// ALL-LABEL:   func.func @test_vector_extract_scalable(
// ALL-SAME:    %[[VAL_0:.*]]: vector<2x8x[2]xf32>) -> vector<8x[2]xf32> {
func.func @test_vector_extract_scalable(%arg0: vector<2x8x[2]xf32>) -> vector<8x[2]xf32> {
  // ALL-NOT: vector.shuffle
  // ALL-NOT: vector.shape_cast
  // ALL: %[[RES:.*]] = vector.extract %[[VAL_0]][1] : vector<8x[2]xf32> from vector<2x8x[2]xf32>
  %0 = vector.extract %arg0[1]: vector<8x[2]xf32> from vector<2x8x[2]xf32>
  // ALL: return %[[RES]] : vector<8x[2]xf32>
  return %0 : vector<8x[2]xf32>
}
// -----
// ALL-LABEL: test_vector_insert
// ALL-SAME: (%[[DEST:.*]]: vector<2x8x4xf32>, %[[SRC:.*]]: vector<8x4xf32>) -> vector<2x8x4xf32> {
func.func @test_vector_insert(%arg0: vector<2x8x4xf32>, %arg1: vector<8x4xf32>) -> vector<2x8x4xf32> {
  // DEFAULT-DAG: %[[ARG_SRC:.*]] = vector.shape_cast %[[SRC]] : vector<8x4xf32> to vector<32xf32>
  // DEFAULT-DAG: %[[ARG_DEST:.*]] = vector.shape_cast %[[DEST]] : vector<2x8x4xf32> to vector<64xf32>
  // DEFAULT: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG_DEST]], %[[ARG_SRC]]
  // DEFAULT-SAME: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
  // DEFAULT-SAME: 88, 89, 90, 91, 92, 93, 94, 95, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
  // DEFAULT-SAME: 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf32>, vector<32xf32>
  // DEFAULT: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<64xf32> to vector<2x8x4xf32>
  // DEFAULT: return %[[RES]] : vector<2x8x4xf32>

  // BW-128-DAG: %[[ARG_SRC:.*]] = vector.shape_cast %[[SRC]] : vector<8x4xf32> to vector<32xf32>
  // BW-128-DAG: %[[ARG_DEST:.*]] = vector.shape_cast %[[DEST]] : vector<2x8x4xf32> to vector<64xf32>
  // BW-128: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG_DEST]], %[[ARG_SRC]]
  // BW-128-SAME: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
  // BW-128-SAME: 88, 89, 90, 91, 92, 93, 94, 95, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
  // BW-128-SAME: 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<64xf32>, vector<32xf32>
  // BW-128: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<64xf32> to vector<2x8x4xf32>
  // BW-128: return %[[RES]] : vector<2x8x4xf32>

  // BW-0: %[[RES:.*]] = vector.insert %[[SRC]], %[[DEST]] [0] : vector<8x4xf32> into vector<2x8x4xf32>
  // BW-0: return %[[RES]] : vector<2x8x4xf32>

  %0 = vector.insert %arg1, %arg0[0]: vector<8x4xf32> into vector<2x8x4xf32>
  return %0 : vector<2x8x4xf32>
}

// ALL-LABEL:   func.func @test_vector_insert_scalable(
// ALL-SAME:    %[[VAL_0:.*]]: vector<2x8x[4]xf32>, %[[VAL_1:.*]]: vector<8x[4]xf32>) -> vector<2x8x[4]xf32> {
func.func @test_vector_insert_scalable(%arg0: vector<2x8x[4]xf32>, %arg1: vector<8x[4]xf32>) -> vector<2x8x[4]xf32> {
  // ALL-NOT: vector.shuffle
  // ALL-NOT: vector.shape_cast
  // ALL: %[[RES:.*]] = vector.insert %[[VAL_1]], %[[VAL_0]] [0] : vector<8x[4]xf32> into vector<2x8x[4]xf32>
  %0 = vector.insert %arg1, %arg0[0]: vector<8x[4]xf32> into vector<2x8x[4]xf32>
  // ALL: return %[[RES]] : vector<2x8x[4]xf32>
  return %0 : vector<2x8x[4]xf32>
}

// -----

// ALL-LABEL: test_vector_extract_scalar
func.func @test_vector_extract_scalar(%idx : index) {
  %cst = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
  // ALL-NOT: vector.shuffle
  // ALL:     vector.extract
  // ALL-NOT: vector.shuffle
  %0 = vector.extract %cst[%idx] : i32 from vector<4xi32>
  return
}

// -----

// ALL-LABEL: test_vector_bitcast
// ALL-SAME: %[[ARG_0:.*]]: vector<4x4xf32>
func.func @test_vector_bitcast(%arg0: vector<4x4xf32>) -> vector<4x8xf16> {
  // DEFAULT: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<4x4xf32> to vector<16xf32>
  // DEFAULT: %[[BITCAST:.*]] = vector.bitcast %[[DOWNCAST]] : vector<16xf32> to vector<32xf16>
  // DEFAULT: %[[UPCAST:.*]] = vector.shape_cast %[[BITCAST]] : vector<32xf16> to vector<4x8xf16>

  // BW-128: %[[UPCAST:.*]] = vector.bitcast %[[ARG_0]] : vector<4x4xf32> to vector<4x8xf16>
  // BW-0: %[[BITCAST:.*]] = vector.bitcast %[[ARG_0]] : vector<4x4xf32> to vector<4x8xf16>
  %1 = vector.bitcast %arg0 : vector<4x4xf32> to vector<4x8xf16>
  return %1 : vector<4x8xf16>
}

// -----

// ALL-LABEL: test_vector_bitcast
// ALL-SAME: %[[ARG_0:.*]]: vector<4x2xf32>
func.func @test_vector_bitcast(%arg0: vector<4x2xf32>) -> vector<4x4xf16> {
  // DEFAULT: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<4x2xf32> to vector<8xf32>
  // DEFAULT: %[[BITCAST:.*]] = vector.bitcast %[[DOWNCAST]] : vector<8xf32> to vector<16xf16>
  // DEFAULT: %[[UPCAST:.*]] = vector.shape_cast %[[BITCAST]] : vector<16xf16> to vector<4x4xf16>
  // BW-128: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<4x2xf32> to vector<8xf32>
  // BW-128: %[[BITCAST:.*]] = vector.bitcast %[[DOWNCAST]] : vector<8xf32> to vector<16xf16>
  // BW-128: %[[UPCAST:.*]] = vector.shape_cast %[[BITCAST]] : vector<16xf16> to vector<4x4xf16>

  // BW-0: %[[BITCAST:.*]] = vector.bitcast %[[ARG_0]] : vector<4x2xf32> to vector<4x4xf16>
  %1 = vector.bitcast %arg0 : vector<4x2xf32> to vector<4x4xf16>
  return %1 : vector<4x4xf16>
}

// -----

// ALL-LABEL: test_vector_bitcast
// ALL-SAME: %[[ARG_0:.*]]: vector<4x[2]xf32>
func.func @test_vector_bitcast(%arg0: vector<4x[2]xf32>) -> vector<4x[4]xf16> {
  // DEFAULT: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<4x[2]xf32> to vector<[8]xf32>
  // DEFAULT: %[[BITCAST:.*]] = vector.bitcast %[[DOWNCAST]] : vector<[8]xf32> to vector<[16]xf16>
  // DEFAULT: %[[UPCAST:.*]] = vector.shape_cast %[[BITCAST]] : vector<[16]xf16> to vector<4x[4]xf16>
  // BW-128: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<4x[2]xf32> to vector<[8]xf32>
  // BW-128: %[[BITCAST:.*]] = vector.bitcast %[[DOWNCAST]] : vector<[8]xf32> to vector<[16]xf16>
  // BW-128: %[[UPCAST:.*]] = vector.shape_cast %[[BITCAST]] : vector<[16]xf16> to vector<4x[4]xf16>

  // BW-0: %[[BITCAST:.*]] = vector.bitcast %[[ARG_0]] : vector<4x[2]xf32> to vector<4x[4]xf16>
  %1 = vector.bitcast %arg0 : vector<4x[2]xf32> to vector<4x[4]xf16>
  return %1 : vector<4x[4]xf16>
}

// -----
// ALL-LABEL: test_vector_bitcast
// ALL-SAME: %[[ARG_0:.*]]: vector<[4]x2xf32>
func.func @test_vector_bitcast(%arg0: vector<[4]x2xf32>) -> vector<[4]x4xf16> {
  // DEFAULT: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<[4]x2xf32> to vector<[8]xf32>
  // DEFAULT: %[[BITCAST:.*]] = vector.bitcast %[[DOWNCAST]] : vector<[8]xf32> to vector<[16]xf16>
  // DEFAULT: %[[UPCAST:.*]] = vector.shape_cast %[[BITCAST]] : vector<[16]xf16> to vector<[4]x4xf16>
  // BW-128: %[[DOWNCAST:.*]] = vector.shape_cast %[[ARG_0]] : vector<[4]x2xf32> to vector<[8]xf32>
  // BW-128: %[[BITCAST:.*]] = vector.bitcast %[[DOWNCAST]] : vector<[8]xf32> to vector<[16]xf16>
  // BW-128: %[[UPCAST:.*]] = vector.shape_cast %[[BITCAST]] : vector<[16]xf16> to vector<[4]x4xf16>

  // BW-0: %[[BITCAST:.*]] = vector.bitcast %[[ARG_0]] : vector<[4]x2xf32> to vector<[4]x4xf16>
  %1 = vector.bitcast %arg0 : vector<[4]x2xf32> to vector<[4]x4xf16>
  return %1 : vector<[4]x4xf16>
}

// -----
// ALL-LABEL: test_vector_load
// ALL-SAME: (%[[ARG_0:.*]]: memref<4x4xf16>)
func.func @test_vector_load(%arg0: memref<4x4xf16>) -> vector<4x4xf16> {
  // DEFAULT: %[[C1:.*]] = arith.constant 1 : index
  // BW-128: %[[C1:.*]] = arith.constant 1 : index
  // DEFAULT: %[[C2:.*]] = arith.constant 2 : index
  // BW-128: %[[C2:.*]] = arith.constant 2 : index
  // DEFAULT: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<16xf16>
  // BW-128: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<16xf16>
  // DEFAULT: %[[LOAD0:.*]] = vector.load %[[ARG_0]][%[[C1]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // BW-128: %[[LOAD0:.*]] = vector.load %[[ARG_0]][%[[C1]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // DEFAULT: %[[SHUFFLE0:.*]] = vector.shuffle %[[CST]], %[[LOAD0]] [16, 17, 18, 19, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<16xf16>, vector<4xf16>
  // BW-128: %[[SHUFFLE0:.*]] = vector.shuffle %[[CST]], %[[LOAD0]] [16, 17, 18, 19, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<16xf16>, vector<4xf16>
  // DEFAULT: %[[C1_0:.*]] = arith.constant 1 : index
  // BW-128: %[[C1_0:.*]] = arith.constant 1 : index
  // DEFAULT: %[[ADD0:.*]] = arith.addi %[[C1]], %[[C1_0]] : index
  // BW-128: %[[ADD0:.*]] = arith.addi %[[C1]], %[[C1_0]] : index
  // DEFAULT: %[[LOAD1:.*]] = vector.load %[[ARG_0]][%[[ADD0]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // BW-128: %[[LOAD1:.*]] = vector.load %[[ARG_0]][%[[ADD0]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // DEFAULT: %[[SHUFFLE1:.*]] = vector.shuffle %[[SHUFFLE0]], %[[LOAD1]] [0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 12, 13, 14, 15] : vector<16xf16>, vector<4xf16>
  // BW-128: %[[SHUFFLE1:.*]] = vector.shuffle %[[SHUFFLE0]], %[[LOAD1]] [0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 12, 13, 14, 15] : vector<16xf16>, vector<4xf16>
  // DEFAULT: %[[C2_1:.*]] = arith.constant 2 : index
  // BW-128: %[[C2_1:.*]] = arith.constant 2 : index
  // DEFAULT: %[[ADD1:.*]] = arith.addi %[[C1]], %[[C2_1]] : index
  // BW-128: %[[ADD1:.*]] = arith.addi %[[C1]], %[[C2_1]] : index
  // DEFAULT: %[[LOAD2:.*]] = vector.load %[[ARG_0]][%[[ADD1]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // BW-128: %[[LOAD2:.*]] = vector.load %[[ARG_0]][%[[ADD1]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // DEFAULT: %[[SHUFFLE2:.*]] = vector.shuffle %[[SHUFFLE1]], %[[LOAD2]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15] : vector<16xf16>, vector<4xf16>
  // BW-128: %[[SHUFFLE2:.*]] = vector.shuffle %[[SHUFFLE1]], %[[LOAD2]] [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15] : vector<16xf16>, vector<4xf16>
  // DEFAULT: %[[C3:.*]] = arith.constant 3 : index
  // BW-128: %[[C3:.*]] = arith.constant 3 : index
  // DEFAULT: %[[ADD2:.*]] = arith.addi %[[C1]], %[[C3]] : index
  // BW-128: %[[ADD2:.*]] = arith.addi %[[C1]], %[[C3]] : index
  // DEFAULT: %[[LOAD3:.*]] = vector.load %[[ARG_0]][%[[ADD2]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // BW-128: %[[LOAD3:.*]] = vector.load %[[ARG_0]][%[[ADD2]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // DEFAULT: %[[SHUFFLE3:.*]] = vector.shuffle %[[SHUFFLE2]], %[[LOAD3]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19] : vector<16xf16>, vector<4xf16>
  // BW-128: %[[SHUFFLE3:.*]] = vector.shuffle %[[SHUFFLE2]], %[[LOAD3]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19] : vector<16xf16>, vector<4xf16>
  // DEFAULT: %[[CAST:.*]] = vector.shape_cast %[[SHUFFLE3]] : vector<16xf16> to vector<4x4xf16>
  // BW-128: %[[CAST:.*]] = vector.shape_cast %[[SHUFFLE3]] : vector<16xf16> to vector<4x4xf16>
  // DEFAULT: return %[[CAST]] : vector<4x4xf16>
  // BW-128: return %[[CAST]] : vector<4x4xf16>

  // BW-0: %[[C1:.*]] = arith.constant 1 : index
  // BW-0: %[[C2:.*]] = arith.constant 2 : index
  // BW-0: %[[LOAD:.*]] = vector.load %[[ARG_0]][%[[C1]], %[[C2]]] : memref<4x4xf16>, vector<4x4xf16>
  // BW-0: return %[[LOAD]] : vector<4x4xf16>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = vector.load %arg0[%c1, %c2] : memref<4x4xf16>, vector<4x4xf16>
  return %0 : vector<4x4xf16>
}

// -----
// ALL-LABEL: test_vector_store
// ALL-SAME: (%[[ARG_0:.*]]: memref<4x4xf16>, %[[ARG_1:.*]]: vector<4x4xf16>) {
func.func @test_vector_store(%arg0: memref<4x4xf16>, %arg1: vector<4x4xf16>) {
  // DEFAULT: %[[CAST0:.*]] = vector.shape_cast %[[ARG_1]] : vector<4x4xf16> to vector<16xf16>
  // BW-128: %[[CAST0:.*]] = vector.shape_cast %[[ARG_1]] : vector<4x4xf16> to vector<16xf16>
  // DEFAULT: %[[C1:.*]] = arith.constant 1 : index
  // BW-128: %[[C1:.*]] = arith.constant 1 : index
  // DEFAULT: %[[C2:.*]] = arith.constant 2 : index
  // BW-128: %[[C2:.*]] = arith.constant 2 : index
  // DEFAULT: %[[CAST1:.*]] = vector.shape_cast %[[CAST0]] : vector<16xf16> to vector<4x4xf16>
  // BW-128: %[[CAST1:.*]] = vector.shape_cast %[[CAST0]] : vector<16xf16> to vector<4x4xf16>
  // DEFAULT: %[[CAST2:.*]] = vector.shape_cast %[[CAST1]] : vector<4x4xf16> to vector<16xf16>
  // BW-128: %[[CAST2:.*]] = vector.shape_cast %[[CAST1]] : vector<4x4xf16> to vector<16xf16>
  // DEFAULT: %[[SHUFFLE0:.*]] = vector.shuffle %[[CAST2]], %[[CAST2]] [0, 1, 2, 3] : vector<16xf16>, vector<16xf16>
  // BW-128: %[[SHUFFLE0:.*]] = vector.shuffle %[[CAST2]], %[[CAST2]] [0, 1, 2, 3] : vector<16xf16>, vector<16xf16>
  // DEFAULT: vector.store %[[SHUFFLE0]], %[[ARG_0]][%[[C1]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // BW-128: vector.store %[[SHUFFLE0]], %[[ARG_0]][%[[C1]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // DEFAULT: %[[SHUFFLE1:.*]] = vector.shuffle %[[CAST2]], %[[CAST2]] [4, 5, 6, 7] : vector<16xf16>, vector<16xf16>
  // BW-128: %[[SHUFFLE1:.*]] = vector.shuffle %[[CAST2]], %[[CAST2]] [4, 5, 6, 7] : vector<16xf16>, vector<16xf16>
  // DEFAULT: %[[C1_0:.*]] = arith.constant 1 : index
  // BW-128: %[[C1_0:.*]] = arith.constant 1 : index
  // DEFAULT: %[[ADD0:.*]] = arith.addi %[[C1]], %[[C1_0]] : index
  // BW-128: %[[ADD0:.*]] = arith.addi %[[C1]], %[[C1_0]] : index
  // DEFAULT: vector.store %[[SHUFFLE1]], %[[ARG_0]][%[[ADD0]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // BW-128: vector.store %[[SHUFFLE1]], %[[ARG_0]][%[[ADD0]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // DEFAULT: %[[SHUFFLE2:.*]] = vector.shuffle %[[CAST2]], %[[CAST2]] [8, 9, 10, 11] : vector<16xf16>, vector<16xf16>
  // BW-128: %[[SHUFFLE2:.*]] = vector.shuffle %[[CAST2]], %[[CAST2]] [8, 9, 10, 11] : vector<16xf16>, vector<16xf16>
  // DEFAULT: %[[C2_1:.*]] = arith.constant 2 : index
  // BW-128: %[[C2_1:.*]] = arith.constant 2 : index
  // DEFAULT: %[[ADD1:.*]] = arith.addi %[[C1]], %[[C2_1]] : index
  // BW-128: %[[ADD1:.*]] = arith.addi %[[C1]], %[[C2_1]] : index
  // DEFAULT: vector.store %[[SHUFFLE2]], %[[ARG_0]][%[[ADD1]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // BW-128: vector.store %[[SHUFFLE2]], %[[ARG_0]][%[[ADD1]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // DEFAULT: %[[SHUFFLE3:.*]] = vector.shuffle %[[CAST2]], %[[CAST2]] [12, 13, 14, 15] : vector<16xf16>, vector<16xf16>
  // BW-128: %[[SHUFFLE3:.*]] = vector.shuffle %[[CAST2]], %[[CAST2]] [12, 13, 14, 15] : vector<16xf16>, vector<16xf16>
  // DEFAULT: %[[C3:.*]] = arith.constant 3 : index
  // BW-128: %[[C3:.*]] = arith.constant 3 : index
  // DEFAULT: %[[ADD2:.*]] = arith.addi %[[C1]], %[[C3]] : index
  // BW-128: %[[ADD2:.*]] = arith.addi %[[C1]], %[[C3]] : index
  // DEFAULT: vector.store %[[SHUFFLE3]], %[[ARG_0]][%[[ADD2]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // BW-128: vector.store %[[SHUFFLE3]], %[[ARG_0]][%[[ADD2]], %[[C2]]] : memref<4x4xf16>, vector<4xf16>
  // DEFAULT: return
  // BW-128: return

  // BW-0: %[[C1:.*]] = arith.constant 1 : index
  // BW-0: %[[C2:.*]] = arith.constant 2 : index
  // BW-0: vector.store %[[ARG_1]], %[[ARG_0]][%[[C1]], %[[C2]]] : memref<4x4xf16>, vector<4x4xf16>
  // BW-0: return
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  vector.store %arg1, %arg0[%c1, %c2] : memref<4x4xf16>, vector<4x4xf16>
  return
}

// -----
// ALL-LABEL: test_create_mask
func.func @test_create_mask() -> vector<1x16xi1> {
  // DEFAULT: %[[C0:.*]] = arith.constant 0 : index
  // BW-128: %[[C0:.*]] = arith.constant 0 : index
  // DEFAULT: %[[C20:.*]] = arith.constant 20 : index
  // BW-128: %[[C20:.*]] = arith.constant 20 : index
  // DEFAULT: %[[MASK:.*]] = vector.create_mask %[[C20]] : vector<16xi1>
  // BW-128: %[[MASK:.*]] = vector.create_mask %[[C20]] : vector<16xi1>
  // DEFAULT: %[[CAST:.*]] = vector.shape_cast %[[MASK]] : vector<16xi1> to vector<1x16xi1>
  // BW-128: %[[CAST:.*]] = vector.shape_cast %[[MASK]] : vector<16xi1> to vector<1x16xi1>

  // BW-0: %[[C0:.*]] = arith.constant 0 : index
  // BW-0: %[[C20:.*]] = arith.constant 20 : index
  // BW-0: %[[MASK:.*]] = vector.create_mask %[[C0]], %[[C20]] : vector<1x16xi1>
  %c0 = arith.constant 0 : index
  %c20 = arith.constant 20 : index
  %0 = vector.create_mask %c0, %c20 : vector<1x16xi1>
  return %0 : vector<1x16xi1>
}

// -----
// ALL-LABEL: test_loop
func.func @test_loop() -> vector<2x4xf16> {
  // DEFAULT: %[[C0:.*]] = arith.constant 0 : index
  // BW-128: %[[C0:.*]] = arith.constant 0 : index
  // DEFAULT: %[[C1:.*]] = arith.constant 1 : index
  // BW-128: %[[C1:.*]] = arith.constant 1 : index
  // DEFAULT: %[[C4:.*]] = arith.constant 4 : index
  // BW-128: %[[C4:.*]] = arith.constant 4 : index
  // DEFAULT: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<8xf16>
  // BW-128: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<8xf16>
  // DEFAULT: %[[FOR:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG1:.*]] = %[[CST]]) -> (vector<8xf16>) {
  // BW-128: %[[FOR:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG1:.*]] = %[[CST]]) -> (vector<8xf16>) {
  // DEFAULT: %[[ADD:.*]] = arith.addf %[[ARG1]], %[[CST]] : vector<8xf16>
  // BW-128: %[[ADD:.*]] = arith.addf %[[ARG1]], %[[CST]] : vector<8xf16>
  // DEFAULT: %[[CAST0:.*]] = vector.shape_cast %[[ADD]] : vector<8xf16> to vector<2x4xf16>
  // BW-128: %[[CAST0:.*]] = vector.shape_cast %[[ADD]] : vector<8xf16> to vector<2x4xf16>
  // DEFAULT: %[[CAST1:.*]] = vector.shape_cast %[[CAST0]] : vector<2x4xf16> to vector<8xf16>
  // BW-128: %[[CAST1:.*]] = vector.shape_cast %[[CAST0]] : vector<2x4xf16> to vector<8xf16>
  // DEFAULT: scf.yield %[[CAST1]] : vector<8xf16>
  // BW-128: scf.yield %[[CAST1]] : vector<8xf16>
  // DEFAULT: }
  // BW-128: }
  // DEFAULT: %[[CAST2:.*]] = vector.shape_cast %[[FOR]] : vector<8xf16> to vector<2x4xf16>
  // BW-128: %[[CAST2:.*]] = vector.shape_cast %[[FOR]] : vector<8xf16> to vector<2x4xf16>
  // DEFAULT: return %[[CAST2]] : vector<2x4xf16>
  // BW-128: return %[[CAST2]] : vector<2x4xf16>

  // BW-0: %[[C0:.*]] = arith.constant 0 : index
  // BW-0: %[[C1:.*]] = arith.constant 1 : index
  // BW-0: %[[C4:.*]] = arith.constant 4 : index
  // BW-0: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<2x4xf16>
  // BW-0: %[[FOR:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG1:.*]] = %[[CST]]) -> (vector<2x4xf16>) {
  // BW-0: %[[ADD:.*]] = arith.addf %[[CST]], %[[ARG1]] : vector<2x4xf16>
  // BW-0: scf.yield %[[ADD]] : vector<2x4xf16>
  // BW-0: }
  // BW-0: return %[[FOR]] : vector<2x4xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %1 = arith.constant dense<1.0> : vector<2x4xf16>
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%arg1 = %1) -> (vector<2x4xf16>) {
    %2 = arith.addf %1, %arg1 : vector<2x4xf16>
    scf.yield %2 : vector<2x4xf16>
  }
  return %r : vector<2x4xf16>
}

// -----
// ALL-LABEL: test_vector_insert_2d_idx
// ALL-SAME: (%[[ARG:.*]]: vector<4x8xf16>) -> vector<8x16xf16>
func.func @test_vector_insert_2d_idx(%arg0: vector<4x8xf16>) -> vector<8x16xf16> {
  // DEFAULT: %[[V0:.*]] = vector.shape_cast %[[ARG]] : vector<4x8xf16> to vector<32xf16>
  // DEFAULT: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<128xf16>
  // DEFAULT: %[[V1:.*]] = vector.shuffle %[[V0]], %[[V0]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<32xf16>, vector<32xf16>
  // DEFAULT: %[[V2:.*]] = vector.insert_strided_slice %[[V1]], %[[CST]] {offsets = [0], strides = [1]} : vector<8xf16> into vector<128xf16>
  // DEFAULT: %[[V3:.*]] = vector.shuffle %[[V0]], %[[V0]] [8, 9, 10, 11, 12, 13, 14, 15] : vector<32xf16>, vector<32xf16>
  // DEFAULT: %[[V4:.*]] = vector.insert_strided_slice %[[V3]], %[[V2]] {offsets = [16], strides = [1]} : vector<8xf16> into vector<128xf16>
  // DEFAULT: %[[V5:.*]] = vector.shuffle %[[V0]], %[[V0]] [16, 17, 18, 19, 20, 21, 22, 23] : vector<32xf16>, vector<32xf16>
  // DEFAULT: %[[V6:.*]] = vector.insert_strided_slice %[[V5]], %[[V4]] {offsets = [32], strides = [1]} : vector<8xf16> into vector<128xf16>
  // DEFAULT: %[[V7:.*]] = vector.shuffle %[[V0]], %[[V0]] [24, 25, 26, 27, 28, 29, 30, 31] : vector<32xf16>, vector<32xf16>
  // DEFAULT: %[[V8:.*]] = vector.insert_strided_slice %[[V7]], %[[V6]] {offsets = [48], strides = [1]} : vector<8xf16> into vector<128xf16>
  // DEFAULT: %[[V9:.*]] = vector.shape_cast %[[V8]] : vector<128xf16> to vector<8x16xf16>
  // DEFAULT: return %[[V9]] : vector<8x16xf16>

  // BW-128: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf16>
  // BW-128: %[[V0:.*]] = vector.insert_strided_slice %[[ARG]], %[[CST]] {offsets = [0, 0], strides = [1, 1]} : vector<4x8xf16> into vector<8x16xf16>
  // BW-128: return %[[V0]] : vector<8x16xf16>

  // BW-0: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<8x16xf16>
  // BW-0: %[[V0:.*]] = vector.insert_strided_slice %[[ARG]], %[[CST]] {offsets = [0, 0], strides = [1, 1]} : vector<4x8xf16> into vector<8x16xf16>
  // BW-0: return %[[V0]] : vector<8x16xf16>
  %cst = arith.constant dense <0.0> : vector<8x16xf16>
  %0 = vector.insert_strided_slice %arg0, %cst {offsets = [0, 0], strides = [1, 1]} : vector<4x8xf16> into vector<8x16xf16>
  return %0 : vector<8x16xf16>
}

// -----
// ALL-LABEL: test_if_single_vector
func.func @test_if_single_vector() -> vector<16x1xi32> {
  // DEFAULT: %[[COND:.*]] = arith.constant false
  // DEFAULT: %[[CST:.*]] = arith.constant dense<3> : vector<16xi32>
  // DEFAULT: %[[V0:.*]] = scf.if %[[COND]] -> (vector<16xi32>) {
  // DEFAULT:   %[[CST_THEN:.*]] = arith.constant dense<6> : vector<16xi32>
  // DEFAULT:   %[[V2:.*]] = vector.shape_cast %[[CST_THEN]] : vector<16xi32> to vector<16x1xi32>
  // DEFAULT:   %[[V3:.*]] = vector.shape_cast %[[V2]] : vector<16x1xi32> to vector<16xi32>
  // DEFAULT:   scf.yield %[[V3]] : vector<16xi32>
  // DEFAULT: } else {
  // DEFAULT:   %[[CST_ELSE:.*]] = arith.constant dense<0> : vector<16xi32>
  // DEFAULT:   %[[V4:.*]] = vector.shape_cast %[[CST_ELSE]] : vector<16xi32> to vector<16x1xi32>
  // DEFAULT:   %[[V5:.*]] = vector.shape_cast %[[V4]] : vector<16x1xi32> to vector<16xi32>
  // DEFAULT:   scf.yield %[[V5]] : vector<16xi32>
  // DEFAULT: }
  // DEFAULT: %[[V1:.*]] = vector.shape_cast %[[V0]] : vector<16xi32> to vector<16x1xi32>
  // DEFAULT: return %[[V1]] : vector<16x1xi32>

  // BW-128: %[[COND:.*]] = arith.constant false
  // BW-128: %[[CST:.*]] = arith.constant dense<3> : vector<16xi32>
  // BW-128: %[[V0:.*]] = scf.if %[[COND]] -> (vector<16xi32>) {
  // BW-128:   %[[CST_THEN:.*]] = arith.constant dense<6> : vector<16xi32>
  // BW-128:   %[[V2:.*]] = vector.shape_cast %[[CST_THEN]] : vector<16xi32> to vector<16x1xi32>
  // BW-128:   %[[V3:.*]] = vector.shape_cast %[[V2]] : vector<16x1xi32> to vector<16xi32>
  // BW-128:   scf.yield %[[V3]] : vector<16xi32>
  // BW-128: } else {
  // BW-128:   %[[CST_ELSE:.*]] = arith.constant dense<0> : vector<16xi32>
  // BW-128:   %[[V4:.*]] = vector.shape_cast %[[CST_ELSE]] : vector<16xi32> to vector<16x1xi32>
  // BW-128:   %[[V5:.*]] = vector.shape_cast %[[V4]] : vector<16x1xi32> to vector<16xi32>
  // BW-128:   scf.yield %[[V5]] : vector<16xi32>
  // BW-128: }
  // BW-128: %[[V1:.*]] = vector.shape_cast %[[V0]] : vector<16xi32> to vector<16x1xi32>
  // BW-128: return %[[V1]] : vector<16x1xi32>

  // BW-0: %[[COND:.*]] = arith.constant false
  // BW-0: %[[V:.*]] = arith.constant dense<3> : vector<16x1xi32>
  // BW-0: %[[R:.*]] = scf.if %[[COND]] -> (vector<16x1xi32>) {
  // BW-0:   %[[ADD:.*]] = arith.addi %[[V]], %[[V]] : vector<16x1xi32>
  // BW-0:   scf.yield %[[ADD]] : vector<16x1xi32>
  // BW-0: } else {
  // BW-0:   %[[SUB:.*]] = arith.subi %[[V]], %[[V]] : vector<16x1xi32>
  // BW-0:   scf.yield %[[SUB]] : vector<16x1xi32>
  // BW-0: }
  %cond = arith.constant 0 : i1
  %v = arith.constant dense<3> : vector<16x1xi32>
  %r = scf.if %cond -> (vector<16x1xi32>) {
    %add = arith.addi %v, %v : vector<16x1xi32>
    scf.yield %add : vector<16x1xi32>
  } else {
    %sub = arith.subi %v, %v : vector<16x1xi32>
    scf.yield %sub : vector<16x1xi32>
  }
  return %r : vector<16x1xi32>
}

// -----
// ALL-LABEL: test_while
func.func @test_while() -> vector<2x4xf32> {
  // DEFAULT: %[[CST:.*]] = arith.constant dense<1.000000e+00> : vector<8xf32>
  // DEFAULT: %[[V0:.*]] = scf.while (%[[ARG0:.*]] = %[[CST]]) : (vector<8xf32>) -> vector<8xf32> {
  // DEFAULT:   %[[V2:.*]] = vector.shape_cast %[[ARG0]] : vector<8xf32> to vector<2x4xf32>
  // DEFAULT:   %[[C0:.*]] = arith.constant 0 : i32
  // DEFAULT:   %[[COND:.*]] = arith.cmpi slt, %[[C0]], %[[C0]] : i32
  // DEFAULT:   %[[V4:.*]] = vector.shape_cast %[[V2]] : vector<2x4xf32> to vector<8xf32>
  // DEFAULT:   scf.condition(%[[COND]]) %[[V4]] : vector<8xf32>
  // DEFAULT: } do {
  // DEFAULT: ^bb0(%[[ARG1:.*]]: vector<8xf32>):
  // DEFAULT:   %[[V2:.*]] = arith.addf %[[ARG1]], %[[ARG1]] : vector<8xf32>
  // DEFAULT:   %[[V3:.*]] = vector.shape_cast %[[V2]] : vector<8xf32> to vector<2x4xf32>
  // DEFAULT:   %[[V4:.*]] = vector.shape_cast %[[V3]] : vector<2x4xf32> to vector<8xf32>
  // DEFAULT:   scf.yield %[[V4]] : vector<8xf32>
  // DEFAULT: }
  // DEFAULT: %[[V1:.*]] = vector.shape_cast %[[V0]] : vector<8xf32> to vector<2x4xf32>
  // DEFAULT: return %[[V1]] : vector<2x4xf32>

  // BW-128: %[[V:.*]] = arith.constant dense<1.000000e+00> : vector<2x4xf32>
  // BW-128: %[[RESULT:.*]] = scf.while (%[[ARG0:.*]] = %[[V]]) : (vector<2x4xf32>) -> vector<2x4xf32> {
  // BW-128:   %[[C0:.*]] = arith.constant 0 : i32
  // BW-128:   %[[COND:.*]] = arith.cmpi slt, %[[C0]], %[[C0]] : i32
  // BW-128:   scf.condition(%[[COND]]) %[[ARG0]] : vector<2x4xf32>
  // BW-128: } do {
  // BW-128: ^bb0(%[[ARG1:.*]]: vector<2x4xf32>):
  // BW-128:   %[[ADD:.*]] = arith.addf %[[ARG1]], %[[ARG1]] : vector<2x4xf32>
  // BW-128:   scf.yield %[[ADD]] : vector<2x4xf32>
  // BW-128: }
  // BW-128: return %[[RESULT]] : vector<2x4xf32>

  // BW-0: %[[V:.*]] = arith.constant dense<1.000000e+00> : vector<2x4xf32>
  // BW-0: %[[RESULT:.*]] = scf.while (%[[ARG0:.*]] = %[[V]]) : (vector<2x4xf32>) -> vector<2x4xf32> {
  // BW-0:   %[[C0:.*]] = arith.constant 0 : i32
  // BW-0:   %[[COND:.*]] = arith.cmpi slt, %[[C0]], %[[C0]] : i32
  // BW-0:   scf.condition(%[[COND]]) %[[ARG0]] : vector<2x4xf32>
  // BW-0: } do {
  // BW-0: ^bb0(%[[ARG1:.*]]: vector<2x4xf32>):
  // BW-0:   %[[ADD:.*]] = arith.addf %[[ARG1]], %[[ARG1]] : vector<2x4xf32>
  // BW-0:   scf.yield %[[ADD]] : vector<2x4xf32>
  // BW-0: }
  // BW-0: return %[[RESULT]] : vector<2x4xf32>
  %v = arith.constant dense<1.0> : vector<2x4xf32>
  %result = scf.while (%arg0 = %v) : (vector<2x4xf32>) -> vector<2x4xf32> {
    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi slt, %c0, %c0 : i32
    scf.condition(%cond) %arg0 : vector<2x4xf32>
  } do {
  ^bb0(%arg1: vector<2x4xf32>):
    %add = arith.addf %arg1, %arg1 : vector<2x4xf32>
    scf.yield %add : vector<2x4xf32>
  }
  return %result : vector<2x4xf32>
}

// -----
// ALL-LABEL: test_vector_splat
// ALL-SAME: (%[[ARG:.*]]: i32) -> vector<4x2xi32>
func.func @test_vector_splat(%arg0: i32) -> vector<4x2xi32> {
  // DEFAULT: %[[SPLAT:.*]] = vector.splat %[[ARG]] : vector<8xi32>
  // DEFAULT: %[[CAST:.*]] = vector.shape_cast %[[SPLAT]] : vector<8xi32> to vector<4x2xi32>
  // DEFAULT: return %[[CAST]] : vector<4x2xi32>
  // BW-128: %[[SPLAT:.*]] = vector.splat %[[ARG]] : vector<8xi32>
  // BW-128: %[[CAST:.*]] = vector.shape_cast %[[SPLAT]] : vector<8xi32> to vector<4x2xi32>
  // BW-128: return %[[CAST]] : vector<4x2xi32>

  // BW-0: %[[SPLAT:.*]] = vector.splat %[[ARG]] : vector<4x2xi32>
  // BW-0: return %[[SPLAT]] : vector<4x2xi32>
  %0 = vector.splat %arg0 : vector<4x2xi32>
  return %0 : vector<4x2xi32>
}
