//
// TODO: this test case is temporarily disabled as we are improving zero-cost sparse tensor reshaping.
// XFAIL: *
//
// RUN: mlir-opt %s --linalg-generalize-named-ops --sparsification --cse --canonicalize | FileCheck %s

#COO_2D = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton), posWidth = 32, crdWidth = 32 }>
#COO_3D = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : compressed(nonunique), d1 : singleton(nonunique), d2 : singleton), posWidth = 32, crdWidth = 32 }>


// CHECK-LABEL:   func.func @sparse_reshape_fused(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<5x6xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<6x2x3xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed_nu", "singleton_nu", "singleton" ], posWidth = 32, crdWidth = 32 }>>) -> tensor<?x?x?xf32> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 5 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_7:.*]] = tensor.empty() : tensor<5x6xf32>
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index}
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 0 : index}
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 1 : index}
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 2 : index}
// CHECK-DAG:       %[[VAL_12:.*]] = sparse_tensor.values %[[VAL_1]]
// CHECK-DAG:       %[[VAL_13:.*]] = bufferization.to_memref %[[VAL_7]] : memref<5x6xf32>
// CHECK:           scf.for %[[VAL_14:.*]] = %[[VAL_5]] to %[[VAL_3]] step %[[VAL_6]] {
// CHECK:             %[[VAL_15:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_5]]] : memref<?xi32>
// CHECK:             %[[VAL_16:.*]] = arith.extui %[[VAL_15]] : i32 to i64
// CHECK:             %[[VAL_17:.*]] = arith.index_cast %[[VAL_16]] : i64 to index
// CHECK:             %[[VAL_18:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_6]]] : memref<?xi32>
// CHECK:             %[[VAL_19:.*]] = arith.extui %[[VAL_18]] : i32 to i64
// CHECK:             %[[VAL_20:.*]] = arith.index_cast %[[VAL_19]] : i64 to index
// CHECK:             %[[VAL_21:.*]] = scf.while (%[[VAL_22:.*]] = %[[VAL_17]]) : (index) -> index {
// CHECK:               %[[VAL_23:.*]] = arith.cmpi ult, %[[VAL_22]], %[[VAL_20]] : index
// CHECK:               scf.condition(%[[VAL_23]]) %[[VAL_22]] : index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_24:.*]]: index):
// CHECK:               %[[VAL_25:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_24]]] : memref<?xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_26:.*]] = arith.extui %[[VAL_25]] : i32 to i64
// CHECK:               %[[VAL_27:.*]] = arith.index_cast %[[VAL_26]] : i64 to index
// CHECK:               %[[VAL_28:.*]] = scf.while (%[[VAL_29:.*]] = %[[VAL_24]]) : (index) -> index {
// CHECK:                 %[[VAL_30:.*]] = arith.cmpi ult, %[[VAL_29]], %[[VAL_20]] : index
// CHECK:                 %[[VAL_31:.*]] = scf.if %[[VAL_30]] -> (i1) {
// CHECK:                   %[[VAL_32:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_29]]] : memref<?xi32, strided<[?], offset: ?>>
// CHECK:                   %[[VAL_33:.*]] = arith.extui %[[VAL_32]] : i32 to i64
// CHECK:                   %[[VAL_34:.*]] = arith.index_cast %[[VAL_33]] : i64 to index
// CHECK:                   %[[VAL_35:.*]] = arith.cmpi eq, %[[VAL_34]], %[[VAL_27]] : index
// CHECK:                   scf.yield %[[VAL_35]] : i1
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_2]] : i1
// CHECK:                 }
// CHECK:                 scf.condition(%[[VAL_36:.*]]) %[[VAL_29]] : index
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_37:.*]]: index):
// CHECK:                 %[[VAL_38:.*]] = arith.addi %[[VAL_37]], %[[VAL_6]] : index
// CHECK:                 scf.yield %[[VAL_38]] : index
// CHECK:               }
// CHECK:               %[[VAL_39:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_14]], %[[VAL_27]]] : tensor<5x6xf32>
// CHECK:               scf.for %[[VAL_40:.*]] = %[[VAL_24]] to %[[VAL_41:.*]] step %[[VAL_6]] {
// CHECK:                 %[[VAL_42:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_40]]] : memref<?xi32, strided<[?], offset: ?>>
// CHECK:                 %[[VAL_43:.*]] = arith.extui %[[VAL_42]] : i32 to i64
// CHECK:                 %[[VAL_44:.*]] = arith.index_cast %[[VAL_43]] : i64 to index
// CHECK:                 %[[VAL_45:.*]] = arith.muli %[[VAL_44]], %[[VAL_4]] : index
// CHECK:                 %[[VAL_46:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_40]]] : memref<?xi32, strided<[?], offset: ?>>
// CHECK:                 %[[VAL_47:.*]] = arith.extui %[[VAL_46]] : i32 to i64
// CHECK:                 %[[VAL_48:.*]] = arith.index_cast %[[VAL_47]] : i64 to index
// CHECK:                 %[[VAL_49:.*]] = arith.addi %[[VAL_45]], %[[VAL_48]] : index
// CHECK:                 %[[VAL_50:.*]] = tensor.extract %[[VAL_7]]{{\[}}%[[VAL_14]], %[[VAL_49]]] : tensor<5x6xf32>
// CHECK:                 %[[VAL_51:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_40]]] : memref<?xf32>
// CHECK:                 %[[VAL_52:.*]] = arith.mulf %[[VAL_39]], %[[VAL_51]] : f32
// CHECK:                 %[[VAL_53:.*]] = arith.addf %[[VAL_50]], %[[VAL_52]] : f32
// CHECK:                 memref.store %[[VAL_53]], %[[VAL_13]]{{\[}}%[[VAL_14]], %[[VAL_49]]] : memref<5x6xf32>
// CHECK:               } {"Emitted from" = "linalg.generic"}
// CHECK:               scf.yield %[[VAL_54:.*]] : index
// CHECK:             } attributes {"Emitted from" = "linalg.generic"}
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:           %[[VAL_55:.*]] = bufferization.to_tensor %[[VAL_13]] : memref<5x6xf32>
// CHECK:           %[[VAL_56:.*]] = tensor.expand_shape %[[VAL_55]] {{\[\[}}0], [1, 2]] : tensor<5x6xf32> into tensor<5x2x3xf32>
// CHECK:           %[[VAL_57:.*]] = tensor.cast %[[VAL_56]] : tensor<5x2x3xf32> to tensor<?x?x?xf32>
// CHECK:           return %[[VAL_57]] : tensor<?x?x?xf32>
// CHECK:         }
func.func @sparse_reshape_fused(%arg0: tensor<5x6xf32>, %arg1: tensor<6x2x3xf32, #COO_3D>) -> tensor<?x?x?xf32> {
  %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2]] : tensor<6x2x3xf32, #COO_3D> into tensor<6x6xf32, #COO_2D>
  %0 = tensor.empty() : tensor<5x6xf32>
  %2 = linalg.matmul ins(%arg0, %collapsed : tensor<5x6xf32>, tensor<6x6xf32, #COO_2D>) outs(%0 : tensor<5x6xf32>) -> tensor<5x6xf32>
  %expanded = tensor.expand_shape %2 [[0], [1, 2]] : tensor<5x6xf32> into tensor<5x2x3xf32>
  %ret1 = tensor.cast %expanded : tensor<5x2x3xf32> to tensor<?x?x?xf32>
  return %ret1 : tensor<?x?x?xf32>
}
