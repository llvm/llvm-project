// RUN: mlir-opt -test-linalg-fuse-fill-op-with-reduce-op -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func private @test_reduce_sum_kernel(
// CHECK-SAME:                                              %[[VAL_0:.*]]: tensor<147456xi8>) -> tensor<i8> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<i8>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%[[VAL_0]] : tensor<147456xi8>) outs(%[[VAL_3]] : tensor<i8>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: i8, %[[VAL_6:.*]]: i8):
// CHECK:             %[[VAL_7:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_7]], %[[VAL_1]] : index
// CHECK:             %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_2]], %[[VAL_6]] : i8
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_5]], %[[VAL_9]] : i8
// CHECK:             linalg.yield %[[VAL_10]] : i8
// CHECK:           } -> tensor<i8>
// CHECK:           return %[[VAL_11:.*]] : tensor<i8>
// CHECK:         }

func.func private @test_reduce_sum_kernel(%arg0: tensor<147456xi8>) -> (tensor<i8>) {
  %1 = tensor.empty() : tensor<i8>
  %c0_i8 = arith.constant 0 : i8
  %2 = linalg.fill ins(%c0_i8 : i8) outs(%1 : tensor<i8>) -> tensor<i8>
  %reduced = linalg.reduce ins(%arg0 : tensor<147456xi8>) outs(%2 : tensor<i8>) dimensions = [0]
    (%in: i8, %init: i8) {
      %3 = arith.addi %in, %init : i8
      linalg.yield %3 : i8
    }
  return %reduced : tensor<i8>
}

// -----

func.func private @test_missing_fill(%arg0: tensor<147456xi8>) -> (tensor<i8>) {
  %1 = tensor.empty() : tensor<i8>
  // CHECK: linalg.reduce
  %reduced = linalg.reduce ins(%arg0 : tensor<147456xi8>) outs(%1 : tensor<i8>) dimensions = [0]
    (%in: i8, %init: i8) {
      %3 = arith.addi %in, %init : i8
      linalg.yield %3 : i8
    }
  return %reduced : tensor<i8>
}

// -----

// CHECK-LABEL:   func.func private @test_reduce_multiply_kernel(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: tensor<147456xi8>) -> tensor<i8> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i8
// CHECK:           %[[VAL_3:.*]] = tensor.empty() : tensor<i8>
// CHECK:           %[[VAL_4:.*]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%[[VAL_0]] : tensor<147456xi8>) outs(%[[VAL_3]] : tensor<i8>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: i8, %[[VAL_6:.*]]: i8):
// CHECK:             %[[VAL_7:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_7]], %[[VAL_1]] : index
// CHECK:             %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_2]], %[[VAL_6]] : i8
// CHECK:             %[[VAL_10:.*]] = arith.muli %[[VAL_5]], %[[VAL_9]] : i8
// CHECK:             linalg.yield %[[VAL_10]] : i8
// CHECK:           } -> tensor<i8>
// CHECK:           return %[[VAL_11:.*]] : tensor<i8>
// CHECK:         }

func.func private @test_reduce_multiply_kernel(%arg0: tensor<147456xi8>) -> (tensor<i8>) {
  %1 = tensor.empty() : tensor<i8>
  %c1_i8 = arith.constant 1 : i8
  %2 = linalg.fill ins(%c1_i8 : i8) outs(%1 : tensor<i8>) -> tensor<i8>
  %reduced = linalg.reduce ins(%arg0 : tensor<147456xi8>) outs(%2 : tensor<i8>) dimensions = [0]
    (%in: i8, %init: i8) {
      %3 = arith.muli %in, %init : i8
      linalg.yield %3 : i8
    }
  return %reduced : tensor<i8>
}

// -----

func.func private @test_reduce_sum_on_multiple_dims(%arg0: tensor<2x147456xi8>) -> (tensor<i8>) {
  %1 = tensor.empty() : tensor<i8>
  %c0_i8 = arith.constant 0 : i8
  // CHECK: linalg.fill
  %2 = linalg.fill ins(%c0_i8 : i8) outs(%1 : tensor<i8>) -> tensor<i8>
  // CHECK: linalg.reduce
  %reduced = linalg.reduce ins(%arg0 : tensor<2x147456xi8>) outs(%2 : tensor<i8>) dimensions = [0, 1]
    (%in: i8, %init: i8) {
      %3 = arith.addi %in, %init : i8
      linalg.yield %3 : i8
    }
  return %reduced : tensor<i8>
}
