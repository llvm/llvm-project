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

// -----

// Don't lower `shape.broadcast` if a `shape.shape` type is involved.
// CHECK-LABEL: @broadcast
func @broadcast(%a : tensor<?xindex>, %b : !shape.shape) -> !shape.shape {
  // CHECK: shape.broadcast
  %c = shape.broadcast %a, %b : tensor<?xindex>, !shape.shape -> !shape.shape
  return %c : !shape.shape
}

// -----

// CHECK-LABEL: @broadcast
// CHECK-SAME:  (%[[LHS:.*]]: tensor<?xindex>, %[[RHS:.*]]: tensor<?xindex>)
func @broadcast(%a : tensor<?xindex>, %b : tensor<?xindex>) {
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: %[[LHS_RANK:.*]] = dim %[[LHS]], %[[C0]] : tensor<?xindex>
  // CHECK: %[[RHS_RANK:.*]] = dim %[[RHS]], %[[C0]] : tensor<?xindex>
  // CHECK: %[[LHS_SMALLER:.*]] = cmpi "ule", %[[LHS_RANK]], %[[RHS_RANK]]
  // CHECK: %[[ARG:.*]]:4 = scf.if %[[LHS_SMALLER]] -> (index, tensor<?xindex>, index, tensor<?xindex>) {
  // CHECK:   scf.yield %[[LHS_RANK]], %[[LHS]], %[[RHS_RANK]], %[[RHS]] : index, tensor<?xindex>, index, tensor<?xindex>
  // CHECK: } else {
  // CHECK:   scf.yield %[[RHS_RANK]], %[[RHS]], %[[LHS_RANK]], %[[LHS]] : index, tensor<?xindex>, index, tensor<?xindex>
  // CHECK: }
  // CHECK: %[[MEM:.*]] = alloca(%[[ARG]]#2) : memref<?xindex>
  // CHECK: %[[RANK_DIFF:.*]] = subi %[[ARG]]#2, %[[ARG]]#0 : index
  // CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[RANK_DIFF]] step %[[C1]] {
  // CHECK:   %[[EXTENT:.*]] = extract_element %[[ARG]]#3[%[[IV]]] : tensor<?xindex>
  // CHECK:   store %[[EXTENT]], %[[MEM]][%[[IV]]] : memref<?xindex>
  // CHECK: }
  // CHECK: scf.for %[[IV:.*]] = %[[RANK_DIFF]] to %[[ARG]]#2 step %[[C1]] {
  // CHECK:   %[[GREATER_OPERAND_EXTENT:.*]] = extract_element %[[ARG]]#3[%[[IV]]] : tensor<?xindex>
  // CHECK:   %[[GREATER_OPERAND_EXTENT_IS_ONE:.*]] = cmpi "eq", %[[GREATER_OPERAND_EXTENT]], %[[C1]] : index
  // CHECK:   %[[EXTENT:.*]] = scf.if %[[GREATER_OPERAND_EXTENT_IS_ONE]] -> (index) {
  // CHECK:     %[[IV_SHIFTED:.*]] = subi %[[IV]], %[[RANK_DIFF]] : index
  // CHECK:     %[[SMALLER_OPERAND_EXTENT:.*]] = extract_element %[[ARG]]#1[%[[IV_SHIFTED]]] : tensor<?xindex>
  // CHECK:     scf.yield %[[SMALLER_OPERAND_EXTENT]] : index
  // CHECK:   } else {
  // CHECK:     scf.yield %[[GREATER_OPERAND_EXTENT]] : index
  // CHECK:   }
  // CHECK:   store %[[EXTENT]], %[[MEM]][%[[IV]]] : memref<?xindex>
  // CHECK: }
  // CHECK: %[[BROADCASTED:.*]] = tensor_load %[[MEM]] : memref<?xindex>
  %0 = shape.broadcast %a, %b
      : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
  return
}

