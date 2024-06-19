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
