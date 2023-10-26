// RUN: mlir-opt %s --linalg-generalize-named-ops --sparsification --cse --canonicalize | FileCheck %s

#COO_2D = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton), posWidth = 32, crdWidth = 32 }>
#COO_3D = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : compressed(nonunique), d1 : singleton(nonunique), d2 : singleton), posWidth = 32, crdWidth = 32 }>

// CHECK-LABEL:   func.func @sparse_reshape_fused(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<5x6xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<6x2x3xf32, #sparse_tensor.encoding<{{{.*}}}>>) -> tensor<?x?x?xf32> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 5 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = tensor.collapse_shape %[[VAL_1]]
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<5x6xf32>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.positions %[[VAL_6]] {level = 0 : index}
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_6]] {level = 0 : index}
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.coordinates %[[VAL_6]] {level = 1 : index}
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.values %[[VAL_6]]
// CHECK:           %[[VAL_12:.*]] = bufferization.to_memref %[[VAL_7]] : memref<5x6xf32>
// CHECK:           scf.for %[[VAL_13:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK:             %[[VAL_14:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_4]]] : memref<?xi32>
// CHECK:             %[[VAL_15:.*]] = arith.extui %[[VAL_14]] : i32 to i64
// CHECK:             %[[VAL_16:.*]] = arith.index_cast %[[VAL_15]] : i64 to index
// CHECK:             %[[VAL_17:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_5]]] : memref<?xi32>
// CHECK:             %[[VAL_18:.*]] = arith.extui %[[VAL_17]] : i32 to i64
// CHECK:             %[[VAL_19:.*]] = arith.index_cast %[[VAL_18]] : i64 to index
// CHECK:             %[[VAL_20:.*]] = scf.while (%[[VAL_21:.*]] = %[[VAL_16]]) : (index) -> index {
// CHECK:               %[[VAL_22:.*]] = arith.cmpi ult, %[[VAL_21]], %[[VAL_19]] : index
// CHECK:               scf.condition(%[[VAL_22]]) %[[VAL_21]] : index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_23:.*]]: index):
// CHECK:               %[[VAL_24:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_23]]] : memref<?xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_25:.*]] = arith.extui %[[VAL_24]] : i32 to i64
// CHECK:               %[[VAL_26:.*]] = arith.index_cast %[[VAL_25]] : i64 to index
// CHECK:               %[[VAL_27:.*]] = scf.while (%[[VAL_28:.*]] = %[[VAL_23]]) : (index) -> index {
// CHECK:                 %[[VAL_29:.*]] = arith.cmpi ult, %[[VAL_28]], %[[VAL_19]] : index
// CHECK:                 %[[VAL_30:.*]] = scf.if %[[VAL_29]] -> (i1) {
// CHECK:                   %[[VAL_31:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_28]]] : memref<?xi32, strided<[?], offset: ?>>
// CHECK:                   %[[VAL_32:.*]] = arith.extui %[[VAL_31]] : i32 to i64
// CHECK:                   %[[VAL_33:.*]] = arith.index_cast %[[VAL_32]] : i64 to index
// CHECK:                   %[[VAL_34:.*]] = arith.cmpi eq, %[[VAL_33]], %[[VAL_26]] : index
// CHECK:                   scf.yield %[[VAL_34]] : i1
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_2]] : i1
// CHECK:                 }
// CHECK:                 scf.condition(%[[VAL_30]]) %[[VAL_28]] : index
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_35:.*]]: index):
// CHECK:                 %[[VAL_36:.*]] = arith.addi %[[VAL_35]], %[[VAL_5]] : index
// CHECK:                 scf.yield %[[VAL_36]] : index
// CHECK:               }
// CHECK:               %[[VAL_37:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_13]], %[[VAL_26]]] : tensor<5x6xf32>
// CHECK:               scf.for %[[VAL_38:.*]] = %[[VAL_23]] to %[[VAL_27]] step %[[VAL_5]] {
// CHECK:                 %[[VAL_39:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_38]]] : memref<?xi32, strided<[?], offset: ?>>
// CHECK:                 %[[VAL_40:.*]] = arith.extui %[[VAL_39]] : i32 to i64
// CHECK:                 %[[VAL_41:.*]] = arith.index_cast %[[VAL_40]] : i64 to index
// CHECK:                 %[[VAL_42:.*]] = tensor.extract %[[VAL_7]]{{\[}}%[[VAL_13]], %[[VAL_41]]] : tensor<5x6xf32>
// CHECK:                 %[[VAL_43:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_38]]] : memref<?xf32>
// CHECK:                 %[[VAL_44:.*]] = arith.mulf %[[VAL_37]], %[[VAL_43]] : f32
// CHECK:                 %[[VAL_45:.*]] = arith.addf %[[VAL_42]], %[[VAL_44]] : f32
// CHECK:                 memref.store %[[VAL_45]], %[[VAL_12]]{{\[}}%[[VAL_13]], %[[VAL_41]]] : memref<5x6xf32>
// CHECK:               }
// CHECK:               scf.yield %[[VAL_27]] : index
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_46:.*]] = bufferization.to_tensor %[[VAL_12]] : memref<5x6xf32>
// CHECK:           %[[VAL_47:.*]] = tensor.expand_shape %[[VAL_46]] {{\[\[}}0], [1, 2]] : tensor<5x6xf32> into tensor<5x2x3xf32>
// CHECK:           %[[VAL_48:.*]] = tensor.cast %[[VAL_47]] : tensor<5x2x3xf32> to tensor<?x?x?xf32>
// CHECK:           return %[[VAL_48]] : tensor<?x?x?xf32>
// CHECK:         }
func.func @sparse_reshape_fused(%arg0: tensor<5x6xf32>, %arg1: tensor<6x2x3xf32, #COO_3D>) -> tensor<?x?x?xf32> {
  %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2]] : tensor<6x2x3xf32, #COO_3D> into tensor<6x6xf32, #COO_2D>
  %0 = tensor.empty() : tensor<5x6xf32>
  %2 = linalg.matmul ins(%arg0, %collapsed : tensor<5x6xf32>, tensor<6x6xf32, #COO_2D>) outs(%0 : tensor<5x6xf32>) -> tensor<5x6xf32>
  %expanded = tensor.expand_shape %2 [[0], [1, 2]] : tensor<5x6xf32> into tensor<5x2x3xf32>
  %ret1 = tensor.cast %expanded : tensor<5x2x3xf32> to tensor<?x?x?xf32>
  return %ret1 : tensor<?x?x?xf32>
}
