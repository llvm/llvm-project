// RUN: mlir-opt %s --linalg-generalize-named-ops --sparsification --cse --canonicalize | FileCheck %s

#COO_2D = #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ], posWidth = 32, crdWidth = 32 }>
#COO_3D = #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton-nu", "singleton" ], posWidth = 32, crdWidth = 32 }>

// CHECK-LABEL:   func.func @sparse_reshape_fused(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<5x6xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<6x2x3xf32,
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 5 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = tensor.empty() : tensor<5x6xf32>
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index}
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 0 : index}
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 1 : index}
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 2 : index}
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.values %[[VAL_1]]
// CHECK-DAG:       %[[VAL_12:.*]] = bufferization.to_memref %[[VAL_6]] : memref<5x6xf32>
// CHECK:           scf.for %[[VAL_13:.*]] = %[[VAL_4]] to %[[VAL_2]] step %[[VAL_5]] {
// CHECK:             %[[VAL_14:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_4]]] : memref<?xi32>
// CHECK:             %[[VAL_15:.*]] = arith.extui %[[VAL_14]] : i32 to i64
// CHECK:             %[[VAL_16:.*]] = arith.index_cast %[[VAL_15]] : i64 to index
// CHECK:             %[[VAL_17:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_5]]] : memref<?xi32>
// CHECK:             %[[VAL_18:.*]] = arith.extui %[[VAL_17]] : i32 to i64
// CHECK:             %[[VAL_19:.*]] = arith.index_cast %[[VAL_18]] : i64 to index
// CHECK:             scf.for %[[VAL_20:.*]] = %[[VAL_16]] to %[[VAL_19]] step %[[VAL_5]] {
// CHECK:               %[[VAL_21:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_20]]] : memref<?xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_22:.*]] = arith.extui %[[VAL_21]] : i32 to i64
// CHECK:               %[[VAL_23:.*]] = arith.index_cast %[[VAL_22]] : i64 to index
// CHECK:               %[[VAL_24:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_13]], %[[VAL_23]]] : tensor<5x6xf32>
// CHECK:               %[[VAL_25:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_20]]] : memref<?xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_26:.*]] = arith.extui %[[VAL_25]] : i32 to i64
// CHECK:               %[[VAL_27:.*]] = arith.index_cast %[[VAL_26]] : i64 to index
// CHECK:               %[[VAL_28:.*]] = arith.muli %[[VAL_27]], %[[VAL_3]] : index
// CHECK:               %[[VAL_29:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_20]]] : memref<?xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_30:.*]] = arith.extui %[[VAL_29]] : i32 to i64
// CHECK:               %[[VAL_31:.*]] = arith.index_cast %[[VAL_30]] : i64 to index
// CHECK:               %[[VAL_32:.*]] = arith.addi %[[VAL_28]], %[[VAL_31]] : index
// CHECK:               %[[VAL_33:.*]] = tensor.extract %[[VAL_6]]{{\[}}%[[VAL_13]], %[[VAL_32]]] : tensor<5x6xf32>
// CHECK:               %[[VAL_34:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_20]]] : memref<?xf32>
// CHECK:               %[[VAL_35:.*]] = arith.mulf %[[VAL_24]], %[[VAL_34]] : f32
// CHECK:               %[[VAL_36:.*]] = arith.addf %[[VAL_33]], %[[VAL_35]] : f32
// CHECK:               memref.store %[[VAL_36]], %[[VAL_12]]{{\[}}%[[VAL_13]], %[[VAL_32]]] : memref<5x6xf32>
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:           %[[VAL_37:.*]] = bufferization.to_tensor %[[VAL_12]] : memref<5x6xf32>
// CHECK:           %[[VAL_38:.*]] = tensor.expand_shape %[[VAL_37]] {{\[\[}}0], [1, 2]] : tensor<5x6xf32> into tensor<5x2x3xf32>
// CHECK:           %[[VAL_39:.*]] = tensor.cast %[[VAL_38]] : tensor<5x2x3xf32> to tensor<?x?x?xf32>
// CHECK:           return %[[VAL_39]] : tensor<?x?x?xf32>
// CHECK:         }
func.func @sparse_reshape_fused(%arg0: tensor<5x6xf32>, %arg1: tensor<6x2x3xf32, #COO_3D>) -> tensor<?x?x?xf32> {
  %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2]] : tensor<6x2x3xf32, #COO_3D> into tensor<6x6xf32, #COO_2D>
  %0 = tensor.empty() : tensor<5x6xf32>
  %2 = linalg.matmul ins(%arg0, %collapsed : tensor<5x6xf32>, tensor<6x6xf32, #COO_2D>) outs(%0 : tensor<5x6xf32>) -> tensor<5x6xf32>
  %expanded = tensor.expand_shape %2 [[0], [1, 2]] : tensor<5x6xf32> into tensor<5x2x3xf32>
  %ret1 = tensor.cast %expanded : tensor<5x2x3xf32> to tensor<?x?x?xf32>
  return %ret1 : tensor<?x?x?xf32>
}
