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
  // DEFAULT: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x2xf32> to vector<4xf32>
  // DEFAULT: %[[ARG2:.*]] = vector.shape_cast %[[ORIG_ARG2]] : vector<4x4xf32> to vector<16xf32>
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
