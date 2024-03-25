// RUN: mlir-opt %s -split-input-file -test-vector-linearize | FileCheck %s
// RUN: mlir-opt %s -split-input-file -test-vector-linearize=target-vector-bitwidth=128 | FileCheck %s --check-prefix=CHECK128
// RUN: mlir-opt %s -split-input-file -test-vector-linearize=target-vector-bitwidth=0 | FileCheck %s --check-prefix=CHECK0

// CHECK-LABEL: test_linearize
// CHECK128-LABEL: test_linearize
// CHECK0-LABEL: test_linearize
//  CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x2xf32>)
//  CHECK128-SAME: (%[[ORIG_ARG:.*]]: vector<2x2xf32>)
//       CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x2xf32> to vector<4xf32>
//       CHECK128: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x2xf32> to vector<4xf32>
func.func @test_linearize(%arg0: vector<2x2xf32>) -> vector<2x2xf32> {
//       CHECK: %[[C1:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
//       CHECK128: %[[C1:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
//       CHECK0: %[[C1:.*]] = arith.constant dense<{{.*}}> : vector<2x2xf32>

  %0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[C1]] : vector<4xf32> to vector<2x2xf32>
//       CHECK128: %[[RES:.*]] = vector.shape_cast %[[C1]] : vector<4xf32> to vector<2x2xf32>
// Arith and math ops are handled in generic way, check some of them
//       CHECK: %{{.*}} =  math.sin %[[ARG]] : vector<4xf32>
//       CHECK128: %{{.*}} =  math.sin %[[ARG]] : vector<4xf32>
//       CHECK0: %{{.*}} =  math.sin %{{.*}} : vector<2x2xf32>
  %1 = math.sin %arg0 : vector<2x2xf32>
//       CHECK: %{{.*}} = arith.addf %[[ARG]], %[[C1]] : vector<4xf32>
//       CHECK128: %{{.*}} = arith.addf %[[ARG]], %[[C1]] : vector<4xf32>
//       CHECK0: %{{.*}} = arith.addf %{{.*}} : vector<2x2xf32>

  %2 = arith.addf %arg0, %0 :  vector<2x2xf32>

//       CHECK: return %[[RES]] : vector<2x2xf32>
//       CHECK128: return %[[RES]] : vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// CHECK-LABEL: test_partial_linearize
// CHECK128-LABEL: test_partial_linearize
// CHECK0-LABEL: test_partial_linearize
//  CHECK-SAME: (%[[ORIG_ARG:.*]]: vector<2x2xf32>, %[[ORIG_ARG2:.*]]: vector<4x4xf32>)
//  CHECK128-SAME: (%[[ORIG_ARG:.*]]: vector<2x2xf32>, %[[ORIG_ARG2:.*]]: vector<4x4xf32>)
//  CHECK0-SAME: (%[[ORIG_ARG:.*]]: vector<2x2xf32>, %[[ORIG_ARG2:.*]]: vector<4x4xf32>)
//       CHECK: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x2xf32> to vector<4xf32>
//       CHECK128: %[[ARG:.*]] = vector.shape_cast %[[ORIG_ARG]] : vector<2x2xf32> to vector<4xf32>
//       CHECK: %[[ARG2:.*]] = vector.shape_cast %[[ORIG_ARG2]] : vector<4x4xf32> to vector<16xf32>
func.func @test_partial_linearize(%arg0: vector<2x2xf32>, %arg1: vector<4x4xf32>) -> vector<2x2xf32> {
//       CHECK: %[[C1:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
//       CHECK128: %[[C1:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf32>
//       CHECK0: %[[C1:.*]] = arith.constant dense<{{.*}}> : vector<2x2xf32>

  %0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>
//       CHECK: %[[RES:.*]] = vector.shape_cast %[[C1]] : vector<4xf32> to vector<2x2xf32>
//       CHECK128: %[[RES:.*]] = vector.shape_cast %[[C1]] : vector<4xf32> to vector<2x2xf32>

  // CHECK: %[[C2:.*]] = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 1.000000e+00, 2.000000e+00, 5.000000e+00, 6.000000e+00]> : vector<16xf32>
  // CHECK128: %[[C2:.*]] = arith.constant dense<{{.*}}> : vector<4x4xf32>
  // CHECK0: %[[C2:.*]] = arith.constant dense<{{.*}}> : vector<4x4xf32>
  %5 = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0,3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 5.0, 6.0]]> : vector<4x4xf32>
// Arith and math ops are handled in generic way, check some of them
//       CHECK: %[[SIN:.*]] =  math.sin %[[ARG]] : vector<4xf32>
//       CHECK128: %[[SIN:.*]] =  math.sin %[[ARG]] : vector<4xf32>
//       CHECK0: %[[SIN:.*]] =  math.sin %[[ORIG_ARG]] : vector<2x2xf32>
  %1 = math.sin %arg0 : vector<2x2xf32>

  //     CHECK: %[[SIN1:.*]] =  math.sin %[[ARG2]] : vector<16xf32>
//       CHECK128: %[[SIN1:.*]] =  math.sin %[[ORIG_ARG2]] : vector<4x4xf32>
//       CHECK0: %[[SIN1:.*]] =  math.sin %[[ORIG_ARG2]] : vector<4x4xf32>
  %6 = math.sin %arg1 : vector<4x4xf32>
//       CHECK: %{{.*}} = arith.addf %[[ARG]], %[[C1]] : vector<4xf32>
//       CHECK128: %{{.*}} = arith.addf %[[ARG]], %[[C1]] : vector<4xf32>
//       CHECK0: %{{.*}} = arith.addf %{{.*}} : vector<2x2xf32>

  %2 = arith.addf %arg0, %0 :  vector<2x2xf32>

  // CHECK: %[[ADD2:.*]] = arith.addf %[[ARG2]], %[[C2]] : vector<16xf32>
  // CHECK128: %[[ADD2:.*]] = arith.addf %[[ORIG_ARG2]], %[[C2]] : vector<4x4xf32>
  // CHECK0: %[[ADD2:.*]] = arith.addf %[[ORIG_ARG2]], %[[C2]] : vector<4x4xf32>
  %7 = arith.addf %arg1, %5 : vector<4x4xf32>
//       CHECK: return %[[RES]] : vector<2x2xf32>
//       CHECK128: return %[[RES]] : vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// CHECK-LABEL: test_index_no_linearize
// CHECK128-LABEL: test_index_no_linearize
// CHECK0-LABEL: test_index_no_linearize
func.func @test_index_no_linearize(%arg0: vector<2x2xindex>, %arg1: vector<2x2xindex>) -> vector<2x2xindex> {
    // CHECK: %[[ADD:.*]] = arith.addi {{.*}} : vector<2x2xindex>
    // CHECK128: %[[ADD:.*]] = arith.addi {{.*}} : vector<2x2xindex>
    // CHECK0: %[[ADD:.*]] = arith.addi {{.*}} : vector<2x2xindex>
    %0 = arith.addi %arg0, %arg1 : vector<2x2xindex>
    return %0 : vector<2x2xindex>
}

// -----

// vectorizable operation (arith.mulf) with tensor result types.

func.func @nonvec_result(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    // CHECK: %[[MULF:.*]] = arith.mulf %arg0, %arg1 : tensor<2x2xf32>
    // CHECK128: %[[MULF:.*]] = arith.mulf %arg0, %arg1 : tensor<2x2xf32>
    // CHECK0: %[[MULF:.*]] = arith.mulf %arg0, %arg1 : tensor<2x2xf32>
    %0 = arith.mulf %arg0, %arg1 : tensor<2x2xf32>

    return %0, %arg0 : tensor<2x2xf32>, tensor<2x2xf32>
}
