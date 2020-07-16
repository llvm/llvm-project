// RUN: mlir-opt -convert-shape-to-scf -split-input-file %s | FileCheck %s

// CHECK-LABEL: @shape_reduce
// CHECK-SAME:  (%[[SHAPE:.*]]: tensor<?xindex>) -> index
func @shape_reduce(%shape : tensor<?xindex>) -> index {
  %init = constant 1 : index
  %num_elements = shape.reduce(%shape, %init) : tensor<?xindex> -> index {
    ^bb0(%index : index, %extent : index, %acc: index):
      %new_acc = muli %acc, %extent : index
      shape.yield %new_acc : index
  }
  return %num_elements : index
}
// CHECK-NEXT: %[[INIT:.*]] = constant 1 : index
// CHECK-NEXT: %[[C0:.*]] = constant 0 : index
// CHECK-NEXT: %[[C1:.*]] = constant 1 : index
// CHECK-NEXT: %[[RANK:.*]] = dim %[[SHAPE]], %[[C0]] : tensor<?xindex>
// CHECK-NEXT: %[[RESULT:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[RANK]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[INIT]]) -> (index)
// CHECK-NEXT:   %[[EXTENT:.*]] = extract_element %[[SHAPE]][%[[I]]]
// CHECK-NEXT:   %[[NEW_ACC:.*]] = muli %[[ACC]], %[[EXTENT]] : index
// CHECK-NEXT:   scf.yield %[[NEW_ACC]] : index
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]] : index

// -----

// Lower `shape_of` for unranked tensors.
// CHECK-LABEL: @shape_of_unranked
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>)
func @shape_of_unranked(%arg : tensor<*xf32>) {
  // CHECK-DAG: %[[RANK:.*]] = rank %[[ARG]] : tensor<*xf32>
  // CHECK-DAG: %[[SHAPE_MEM:.*]] = alloca(%[[RANK]]) : memref<?xi64>
  // CHECK-DAG: %[[C0:.*]] = constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = constant 1 : index
  // CHECK:     scf.for %[[I:.*]] = %[[C0]] to %[[RANK]] step %[[C1]] {
  // CHECK-DAG:   %[[DIM:.]] = dim %[[ARG]], %[[I]] : tensor<*xf32>
  // CHECK-DAG:   %[[DIM_INT:.*]] = index_cast %[[DIM]] : index to i64
  // CHECK-DAG:   store %[[DIM_INT]], %[[SHAPE_MEM]][%[[I]]] : memref<?xi64>
  // CHECK:     }
  // CHECK-DAG: %[[SHAPE_INT:.*]] = tensor_load %[[SHAPE_MEM]] : memref<?xi64>
  // CHECK-DAG: %[[SHAPE:.*]] = index_cast %[[SHAPE_INT]] : tensor<?xi64> to tensor<?xindex>
  %shape = shape.shape_of %arg : tensor<*xf32>
  return
}

