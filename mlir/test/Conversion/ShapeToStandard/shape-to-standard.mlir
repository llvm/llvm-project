// RUN: mlir-opt --split-input-file --convert-shape-to-std --verify-diagnostics %s | FileCheck %s

// Lower binary ops.
// CHECK-LABEL: @binary_ops
// CHECK-SAME: (%[[LHS:.*]]: index, %[[RHS:.*]]: index)
func @binary_ops(%lhs : index, %rhs : index) {
  // CHECK: addi %[[LHS]], %[[RHS]] : index
  %sum = shape.add %lhs, %rhs : index, index -> index
  // CHECK: muli %[[LHS]], %[[RHS]] : index
  %product = shape.mul %lhs, %rhs : index, index -> index
  return
}

// -----

// Don't lower binary ops when they operate on `shape.size`.
// CHECK-LABEL: @binary_ops_on_size
// CHECK-SAME: (%[[LHS:.*]]: !shape.size, %[[RHS:.*]]: !shape.size)
func @binary_ops_on_size(%lhs : !shape.size, %rhs : !shape.size) {
  // CHECK: shape.add %[[LHS]], %[[RHS]] : !shape.size, !shape.size -> !shape.size
  // CHECK: shape.mul %[[LHS]], %[[RHS]] : !shape.size, !shape.size -> !shape.size
  %sum = shape.add %lhs, %rhs : !shape.size, !shape.size -> !shape.size
  %prod = shape.mul %lhs, %rhs : !shape.size, !shape.size -> !shape.size
  return
}

// -----

// Convert `rank` to `dim` of the first dimension.
// CHECK-LABEL: @rank
// CHECK-SAME: (%[[SHAPE:.*]]: tensor<?xindex>) -> index
func @rank(%shape : tensor<?xindex>) -> index {
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[RESULT:.*]] = dim %[[SHAPE]], %[[C0]]
  // CHECK: return %[[RESULT]] : index
  %rank = shape.rank %shape : tensor<?xindex> -> index
  return %rank : index
}

// -----

// Don't lower `get_extent` if it is of type `shape.size`.
// CHECK-LABEL: @get_extent
func @get_extent(%shape : tensor<?xindex>, %idx : !shape.size) -> !shape.size {
  // CHECK: shape.get_extent
  %result = shape.get_extent %shape, %idx
      : tensor<?xindex>, !shape.size -> !shape.size
  return %result : !shape.size
}

// -----

// Don't lower `rank` if type is not error-free.
// CHECK-LABEL: @rank
func @rank(%shape : !shape.shape) {
  // CHECK: shape.rank
  %rank = shape.rank %shape : !shape.shape -> !shape.size
  return
}

// -----

// Express `get_extent` as `std.dim` when it relies directly on the outcome of a
// `shape_of` operation.
// CHECK-LABEL: @get_extent_shape_of
// CHECK-SAME:  (%[[ARG:.*]]: tensor<2x3xf32>, %[[IDX:.*]]: index) -> index
func @get_extent_shape_of(%arg : tensor<2x3xf32>, %idx : index) -> index {
  // CHECK: %[[RESULT:.*]] = dim %[[ARG]], %[[IDX]] : tensor<2x3xf32>
  // CHECK: return %[[RESULT]] : index
  %shape = shape.shape_of %arg : tensor<2x3xf32> -> tensor<?xindex>
  %result = shape.get_extent %shape, %idx : tensor<?xindex>, index -> index
  return %result : index
}

// -----

// Express `get_extent` as `std.extract_element`.
// CHECK-LABEL: @get_extent_from_extent_tensor
// CHECK-SAME: (%[[EXTENTS:.*]]: tensor<?xindex>, %[[IDX:.*]]: index) -> index
func @get_extent_from_extent_tensor(%extents : tensor<?xindex>, %idx : index)
    -> index {
  // CHECK: %[[RESULT:.*]] = extract_element %[[EXTENTS]][%[[IDX]]] : tensor<?xindex>
  // CHECK: return %[[RESULT]] : index
  %result = shape.get_extent %extents, %idx : tensor<?xindex>, index -> index
  return %result : index
}

// -----

// Lower `const_shape` to `tensor_from_elements`.
// CHECK-LABEL: @const_shape
// CHECK-SAME: () -> tensor<?xindex>
func @const_shape() -> tensor<?xindex> {
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK: %[[C3:.*]] = constant 3 : index
  // CHECK: %[[TENSOR3:.*]] = tensor_from_elements %[[C1]], %[[C2]], %[[C3]]
  // CHECK: %[[RESULT:.*]] = tensor_cast %[[TENSOR3]] : tensor<3xindex> to tensor<?xindex>
  // CHECK: return %[[RESULT]] : tensor<?xindex>
  %shape = shape.const_shape [1, 2, 3] : tensor<?xindex>
  return %shape : tensor<?xindex>
}

// -----

// Lower `const_shape` in the case of rank 0.
// CHECK-LABEL: func @const_shape_zero_elements
// CHECK-SAME: () -> tensor<?xindex>
func @const_shape_zero_elements() -> tensor<?xindex> {
  // CHECK: %[[TENSOR:.*]] = tensor_from_elements : tensor<0xindex>
  // CHECK: %[[RESULT:.*]] = tensor_cast %[[TENSOR]] : tensor<0xindex> to tensor<?xindex>
  // CHECK: return %[[RESULT]] : tensor<?xindex>
  %shape = shape.const_shape [] : tensor<?xindex>
  return %shape : tensor<?xindex>
}

// -----

// Lower `any` to its first operand.
// CHECK-LABEL: @any_of_three
// CHECK-SAME:  (%[[A:.*]]: tensor<?xindex>, %[[B:.*]]: tensor<?xindex>, %[[C:.*]]: tensor<?xindex>) -> tensor<?xindex>
func @any_of_three(%a : tensor<?xindex>,
                   %b : tensor<?xindex>,
                   %c : tensor<?xindex>) -> tensor<?xindex> {
  // CHECK: return %[[A]] : tensor<?xindex>
  %result = "shape.any"(%a, %b, %c) : (tensor<?xindex>, tensor<?xindex>, tensor<?xindex>) -> tensor<?xindex>
  return %result : tensor<?xindex>
}

// -----

// Lower `any` to its first operand.
// CHECK-LABEL: @any_of_one
// CHECK-SAME:  (%[[A:.*]]: tensor<?xindex>) -> tensor<?xindex>
func @any_of_one(%a : tensor<?xindex>) -> tensor<?xindex> {
  // CHECK: return %[[A]] : tensor<?xindex>
  %result = "shape.any"(%a) : (tensor<?xindex>) -> tensor<?xindex>
  return %result : tensor<?xindex>
}

// -----

// Lower 'const_size` to `std.constant`
// CHECK-LABEL: @const_size
func @const_size() -> index {
  // CHECK: %[[RES:.*]] = constant 42 : index
  %size = shape.const_size 42
  %result = shape.size_to_index %size : !shape.size
  // CHECK: return %[[RES]]
  return %result : index
}

// -----

// Lower `to_extent_tensor` to `std.tensor_cast`
// Fold to_extent_tensor when already on tensor.
// CHECK-LABEL: @to_extent_tensor
// CHECK-SAME: (%[[ARG:.*]]: tensor<?xindex>
func @to_extent_tensor(%arg: tensor<?xindex>) -> tensor<3xindex> {
  // CHECK-NOT: to_extent_tensor
  // CHECK: %[[RES:.*]] = tensor_cast %[[ARG]] : tensor<?xindex> to tensor<3xindex
  %casted = shape.to_extent_tensor %arg : tensor<?xindex> -> tensor<3xindex>
  // CHECK: return %[[RES]]
  return %casted : tensor<3xindex>
}

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
  // CHECK: %[[SHAPE:.*]] = dynamic_tensor_from_elements %[[RANK]] {
  // CHECK: ^bb0(%[[I:.*]]: index):
  // CHECK:   %[[EXTENT:.*]] = dim %[[ARG]], %[[I]] : tensor<*xf32>
  // CHECK:   yield %[[EXTENT]] : index
  // CHECK: } : tensor<?xindex>
  %shape = shape.shape_of %arg : tensor<*xf32> -> tensor<?xindex>
  return
}

// -----

// Don't lower `shape_of` with `shape.shape` type.
// CHECK-LABEL: @shape_of
// CHECK-SAME: (%[[ARG:.*]]: tensor<1x2x3xf32>)
func @shape_of_stat(%arg : tensor<1x2x3xf32>) {
  // CHECK: shape.shape_of %[[ARG]] : tensor<1x2x3xf32> -> !shape.shape
  %shape = shape.shape_of %arg : tensor<1x2x3xf32> -> !shape.shape
  return
}

// -----

// Lower `shape_of` for statically shaped tensor.
// CHECK-LABEL: @shape_of_stat
// CHECK-SAME: (%[[ARG:.*]]: tensor<1x2x3xf32>)
func @shape_of_stat(%arg : tensor<1x2x3xf32>) {
  // CHECK-DAG: %[[C1:.*]] = constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = constant 2 : index
  // CHECK-DAG: %[[C3:.*]] = constant 3 : index
  // CHECK-DAG: %[[SHAPE_UNCASTED:.*]] = tensor_from_elements %[[C1]], %[[C2]], %[[C3]] : tensor<3xindex>
  %shape = shape.shape_of %arg : tensor<1x2x3xf32> -> tensor<?xindex>
  return
}

// -----

// Lower `shape_of` for 0-D tensor.
// CHECK-LABEL: @shape_of_zero_d
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func @shape_of_zero_d(%arg : tensor<f32>) {
  // CHECK-DAG: %[[SHAPE_UNCASTED:.*]] = tensor_from_elements : tensor<0xindex>
  %shape = shape.shape_of %arg : tensor<f32> -> tensor<?xindex>
  return
}

// -----

// Lower `shape_of` for dynamically shaped tensor.
// CHECK-LABEL: @shape_of_dyn
// CHECK-SAME: (%[[ARG:.*]]: tensor<1x5x?xf32>)
func @shape_of_dyn(%arg : tensor<1x5x?xf32>) {
  // CHECK-DAG: %[[C1:.*]] = constant 1 : index
  // CHECK-DAG: %[[C5:.*]] = constant 5 : index
  // CHECK-DAG: %[[C2:.*]] = constant 2 : index
  // CHECK-DAG: %[[DYN_DIM:.*]] = dim %[[ARG]], %[[C2]] : tensor<1x5x?xf32>
  // CHECK-DAG: %[[SHAPE_UNCASTED:.*]] = tensor_from_elements %[[C1]], %[[C5]], %[[DYN_DIM]] : tensor<3xindex>
  %shape = shape.shape_of %arg : tensor<1x5x?xf32> -> tensor<?xindex>
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

// CHECK-LABEL: @broadcast_unknown_extents
// CHECK-SAME:  (%[[LHS:.*]]: tensor<?xindex>, %[[RHS:.*]]: tensor<?xindex>)
func @broadcast_unknown_extents(%a : tensor<?xindex>, %b : tensor<?xindex>) {
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: %[[LHS_RANK:.*]] = dim %[[LHS]], %[[C0]] : tensor<?xindex>
  // CHECK: %[[RHS_RANK:.*]] = dim %[[RHS]], %[[C0]] : tensor<?xindex>
  // CHECK: %[[LHS_RANK_ULE:.*]] = cmpi "ule", %[[LHS_RANK]], %[[RHS_RANK]] : index
  // CHECK: %[[LESSER_RANK:.*]] = select %[[LHS_RANK_ULE]], %[[LHS_RANK]], %[[RHS_RANK]] : index
  // CHECK: %[[GREATER_RANK:.*]] = select %[[LHS_RANK_ULE]], %[[RHS_RANK]], %[[LHS_RANK]] : index
  // CHECK: %[[ERASED_LHS:.*]] = tensor_cast %[[LHS]] : tensor<?xindex> to tensor<?xindex>
  // CHECK: %[[ERASED_RHS:.*]] = tensor_cast %[[RHS]] : tensor<?xindex> to tensor<?xindex>
  // CHECK: %[[LESSER_RANK_OPERAND:.*]] = select %[[LHS_RANK_ULE]], %[[ERASED_LHS]], %[[ERASED_RHS]] : tensor<?xindex>
  // CHECK: %[[GREATER_RANK_OPERAND:.*]] = select %[[LHS_RANK_ULE]], %[[ERASED_RHS]], %[[ERASED_LHS]] : tensor<?xindex>
  // CHECK: %[[MEM:.*]] = alloca(%[[GREATER_RANK]]) : memref<?xindex>
  // CHECK: %[[RANK_DIFF:.*]] = subi %[[GREATER_RANK]], %[[LESSER_RANK]] : index
  // CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[RANK_DIFF]] step %[[C1]] {
  // CHECK:   %[[EXTENT:.*]] = extract_element %[[GREATER_RANK_OPERAND]][%[[IV]]] : tensor<?xindex>
  // CHECK:   store %[[EXTENT]], %[[MEM]][%[[IV]]] : memref<?xindex>
  // CHECK: }
  // CHECK: scf.for %[[IV:.*]] = %[[RANK_DIFF]] to %[[GREATER_RANK]] step %[[C1]] {
  // CHECK:   %[[GREATER_RANK_OPERAND_EXTENT:.*]] = extract_element %[[GREATER_RANK_OPERAND]][%[[IV]]] : tensor<?xindex>
  // CHECK:   %[[GREATER_OPERAND_EXTENT_IS_ONE:.*]] = cmpi "eq", %[[GREATER_RANK_OPERAND_EXTENT]], %[[C1]] : index
  // CHECK:   %[[EXTENT:.*]] = scf.if %[[GREATER_OPERAND_EXTENT_IS_ONE]] -> (index) {
  // CHECK:     %[[IV_SHIFTED:.*]] = subi %[[IV]], %[[RANK_DIFF]] : index
  // CHECK:     %[[LESSER_RANK_OPERAND_EXTENT:.*]] = extract_element %[[LESSER_RANK_OPERAND]][%[[IV_SHIFTED]]] : tensor<?xindex>
  // CHECK:     scf.yield %[[LESSER_RANK_OPERAND_EXTENT]] : index
  // CHECK:   } else {
  // CHECK:     scf.yield %[[GREATER_RANK_OPERAND_EXTENT]] : index
  // CHECK:   }
  // CHECK:   store %[[EXTENT]], %[[MEM]][%[[IV]]] : memref<?xindex>
  // CHECK: }
  // CHECK: %[[BROADCASTED:.*]] = tensor_load %[[MEM]] : memref<?xindex>
  %0 = shape.broadcast %a, %b
      : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
  return
}

// -----

// CHECK-LABEL: @broadcast_known_different_extents
// CHECK-SAME:  (%[[LHS:.*]]: tensor<2xindex>, %[[RHS:.*]]: tensor<3xindex>)
func @broadcast_known_different_extents(%a : tensor<2xindex>, %b : tensor<3xindex>) {
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: %[[LHS_RANK:.*]] = dim %[[LHS]], %[[C0]] : tensor<2xindex>
  // CHECK: %[[RHS_RANK:.*]] = dim %[[RHS]], %[[C0]] : tensor<3xindex>
  // CHECK: %[[LHS_RANK_ULE:.*]] = cmpi "ule", %[[LHS_RANK]], %[[RHS_RANK]] : index
  // CHECK: %[[LESSER_RANK:.*]] = select %[[LHS_RANK_ULE]], %[[LHS_RANK]], %[[RHS_RANK]] : index
  // CHECK: %[[GREATER_RANK:.*]] = select %[[LHS_RANK_ULE]], %[[RHS_RANK]], %[[LHS_RANK]] : index
  // CHECK: %[[ERASED_LHS:.*]] = tensor_cast %[[LHS]] : tensor<2xindex> to tensor<?xindex>
  // CHECK: %[[ERASED_RHS:.*]] = tensor_cast %[[RHS]] : tensor<3xindex> to tensor<?xindex>
  // CHECK: %[[LESSER_RANK_OPERAND:.*]] = select %[[LHS_RANK_ULE]], %[[ERASED_LHS]], %[[ERASED_RHS]] : tensor<?xindex>
  // CHECK: %[[GREATER_RANK_OPERAND:.*]] = select %[[LHS_RANK_ULE]], %[[ERASED_RHS]], %[[ERASED_LHS]] : tensor<?xindex>
  // CHECK: %[[MEM:.*]] = alloca(%[[GREATER_RANK]]) : memref<?xindex>
  // CHECK: %[[RANK_DIFF:.*]] = subi %[[GREATER_RANK]], %[[LESSER_RANK]] : index
  // CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[RANK_DIFF]] step %[[C1]] {
  // CHECK:   %[[EXTENT:.*]] = extract_element %[[GREATER_RANK_OPERAND]][%[[IV]]] : tensor<?xindex>
  // CHECK:   store %[[EXTENT]], %[[MEM]][%[[IV]]] : memref<?xindex>
  // CHECK: }
  // CHECK: scf.for %[[IV:.*]] = %[[RANK_DIFF]] to %[[GREATER_RANK]] step %[[C1]] {
  // CHECK:   %[[GREATER_RANK_OPERAND_EXTENT:.*]] = extract_element %[[GREATER_RANK_OPERAND]][%[[IV]]] : tensor<?xindex>
  // CHECK:   %[[GREATER_OPERAND_EXTENT_IS_ONE:.*]] = cmpi "eq", %[[GREATER_RANK_OPERAND_EXTENT]], %[[C1]] : index
  // CHECK:   %[[EXTENT:.*]] = scf.if %[[GREATER_OPERAND_EXTENT_IS_ONE]] -> (index) {
  // CHECK:     %[[IV_SHIFTED:.*]] = subi %[[IV]], %[[RANK_DIFF]] : index
  // CHECK:     %[[LESSER_RANK_OPERAND_EXTENT:.*]] = extract_element %[[LESSER_RANK_OPERAND]][%[[IV_SHIFTED]]] : tensor<?xindex>
  // CHECK:     scf.yield %[[LESSER_RANK_OPERAND_EXTENT]] : index
  // CHECK:   } else {
  // CHECK:     scf.yield %[[GREATER_RANK_OPERAND_EXTENT]] : index
  // CHECK:   }
  // CHECK:   store %[[EXTENT]], %[[MEM]][%[[IV]]] : memref<?xindex>
  // CHECK: }
  // CHECK: %[[BROADCASTED:.*]] = tensor_load %[[MEM]] : memref<?xindex>
  %0 = shape.broadcast %a, %b
      : tensor<2xindex>, tensor<3xindex> -> tensor<?xindex>
  return
}
