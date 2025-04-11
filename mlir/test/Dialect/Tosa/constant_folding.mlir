// RUN: mlir-opt --test-constant-fold %s | FileCheck %s

// CHECK-LABEL: func @test_const
func.func @test_const(%arg0 : index) -> tensor<4xi32> {
  // CHECK: tosa.const
  %0 = "tosa.const"() {values = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: func @test_const_i64
func.func @test_const_i64(%arg0 : index) -> tensor<4xi64> {
  // CHECK: tosa.const
  %0 = "tosa.const"() {values = dense<[3, 0, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  return %0 : tensor<4xi64>
}

// CHECK-LABEL: func @try_fold_equal_with_unranked_tensor
func.func @try_fold_equal_with_unranked_tensor(%arg0: tensor<4xi32>, %arg1: tensor<1xi32>) {
  // CHECK: tosa.equal
  // CHECK-NEXT: return
  %0 = tosa.equal %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi1>
  return
}

// -----

// CHECK-LABEL: test_1d_slice
func.func @test_1d_slice() -> tensor<6xi32> {
  // CHECK: %[[VAL_0:.+]] = "tosa.const"() <{values = dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>}> : () -> tensor<6xi32>
  // CHECK: return %[[VAL_0]] : tensor<6xi32>
  %0 = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>}> : () -> tensor<10xi32>
  %1 = tosa.const_shape {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2 = tosa.const_shape {values = dense<6> : tensor<1xindex>} : () -> !tosa.shape<1>
  %3 = tosa.slice %0, %1, %2 : (tensor<10xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<6xi32>
  return %3 : tensor<6xi32>
}

// -----

// CHECK-LABEL: test_1d_slice_non_const_input
func.func @test_1d_slice_non_const_input(%arg0 : tensor<10xi32>) -> tensor<6xi32> {
  // check that slice is not folded for non-constant input1
  // CHECK: tosa.slice
  %1 = tosa.const_shape {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2 = tosa.const_shape {values = dense<6> : tensor<1xindex>} : () -> !tosa.shape<1>
  %3 = tosa.slice %arg0, %1, %2 : (tensor<10xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<6xi32>
  return %3 : tensor<6xi32>
}

// -----

// CHECK-LABEL: test_1d_slice_rank_2_input
func.func @test_1d_slice_rank_2_input(%arg0 : tensor<1x10xi32>) -> tensor<1x6xi32> {
  // check that slice is not folded for input1 rank > 1
  // CHECK: tosa.slice
  %0 = "tosa.const"() <{values = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]> : tensor<1x10xi32>}> : () -> tensor<1x10xi32>
  %1 = tosa.const_shape {values = dense<[0, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %2 = tosa.const_shape {values = dense<[1, 6]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %3 = tosa.slice %arg0, %1, %2 : (tensor<1x10xi32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x6xi32>
  return %3 : tensor<1x6xi32>
}

// -----

// CHECK-LABEL: test_1d_slice_more_than_6
func.func @test_1d_slice_more_than_6() -> tensor<7xi32> {
  // check that slice is not folded because output has more than 6 elements
  // CHECK: tosa.slice
  %0 = "tosa.const"() <{values = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]> : tensor<10xi32>}> : () -> tensor<10xi32>
  %1 = tosa.const_shape {values = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2 = tosa.const_shape {values = dense<7> : tensor<1xindex>} : () -> !tosa.shape<1>
  %3 = tosa.slice %0, %1, %2 : (tensor<10xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<7xi32>
  return %3 : tensor<7xi32>
}
