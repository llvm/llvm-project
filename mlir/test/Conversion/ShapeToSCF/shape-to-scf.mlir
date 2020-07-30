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

// Don't lower `shape_of` for result type of `shape.shape`.
// CHECK-LABEL: @shape_of
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>)
func @shape_of(%arg : tensor<*xf32>) {
  // CHECK: shape.shape
  %shape = shape.shape_of %arg : tensor<*xf32> -> !shape.shape
  return
}

// -----

// Lower `shape_of` for unranked tensors.
// CHECK-LABEL: @shape_of_unranked
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>)
func @shape_of_unranked(%arg : tensor<*xf32>) {
  // CHECK: %[[RANK:.*]] = rank %[[ARG]] : tensor<*xf32>
  // CHECK: %[[SHAPE_MEM:.*]] = alloca(%[[RANK]]) : memref<?xindex>
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[RANK]] step %[[C1]] {
  // CHECK:   %[[DIM:.]] = dim %[[ARG]], %[[I]] : tensor<*xf32>
  // CHECK:   store %[[DIM]], %[[SHAPE_MEM]][%[[I]]] : memref<?xindex>
  // CHECK: }
  // CHECK: %[[SHAPE:.*]] = tensor_load %[[SHAPE_MEM]] : memref<?xindex>
  %shape = shape.shape_of %arg : tensor<*xf32> -> tensor<?xindex>
  return
}

// -----

// CHECK-LABEL:  @shape_eq
// CHECK-SAME:   (%[[A:.*]]: tensor<?xindex>, %[[B:.*]]: tensor<?xindex>) -> i1
func @shape_eq(%a : tensor<?xindex>, %b : tensor<?xindex>) -> i1 {
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[RANK_A:.*]] = dim %[[A]], %[[C0]] : tensor<?xindex>
  // CHECK: %[[RANK_B:.*]] = dim %[[B]], %[[C0]] : tensor<?xindex>
  // CHECK: %[[RANK_EQ:.*]] = cmpi "eq", %[[RANK_A]], %[[RANK_B]]
  // CHECK: %[[SHAPE_EQ:.*]] = scf.if %[[RANK_EQ]] -> (i1) {
  // CHECK:   %[[C1:.*]] = constant 1 : index
  // CHECK:   %[[INIT:.*]] = constant true
  // CHECK:   %[[SHAPE_EQ_INNER:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[RANK_A]] step %[[C1]] iter_args(%[[CONJ:.*]] = %[[INIT]]) -> (i1) {
  // CHECK:     %[[EXTENT_A:.*]] = extract_element %[[A]][%[[I]]] : tensor<?xindex>
  // CHECK:     %[[EXTENT_B:.*]] = extract_element %[[B]][%[[I]]] : tensor<?xindex>
  // CHECK:     %[[EXTENT_EQ:.*]] = cmpi "eq", %[[EXTENT_A]], %[[EXTENT_B]]
  // CHECK:     %[[CONJ_NEXT:.*]] = and %[[CONJ]], %[[EXTENT_EQ]]
  // CHECK:     scf.yield %[[CONJ_NEXT]] : i1
  // CHECK:   }
  // CHECK:   scf.yield %[[SHAPE_EQ_INNER]] : i1
  // CHECK: } else {
  // CHECK:   %[[SHAPE_EQ_INNER:.*]] = constant false
  // CHECK:   scf.yield %[[SHAPE_EQ_INNER]] : i1
  // CHECK: }
  // CHECK: return %[[SHAPE_EQ]] : i1
  %result = shape.shape_eq %a, %b : tensor<?xindex>, tensor<?xindex>
  return %result : i1
}
