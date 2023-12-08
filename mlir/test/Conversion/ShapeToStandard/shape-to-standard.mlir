// RUN: mlir-opt --split-input-file --convert-shape-to-std --verify-diagnostics %s | FileCheck %s

// Lower binary ops.
// CHECK-LABEL: @binary_ops
// CHECK-SAME: (%[[LHS:.*]]: index, %[[RHS:.*]]: index)
func.func @binary_ops(%lhs : index, %rhs : index) {
  // CHECK: arith.addi %[[LHS]], %[[RHS]] : index
  %sum = shape.add %lhs, %rhs : index, index -> index
  // CHECK: arith.muli %[[LHS]], %[[RHS]] : index
  %product = shape.mul %lhs, %rhs : index, index -> index
  return
}

// -----

// Don't lower binary ops when they operate on `shape.size`.
// CHECK-LABEL: @binary_ops_on_size
// CHECK-SAME: (%[[LHS:.*]]: !shape.size, %[[RHS:.*]]: !shape.size)
func.func @binary_ops_on_size(%lhs : !shape.size, %rhs : !shape.size) {
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
func.func @rank(%shape : tensor<?xindex>) -> index {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.*]] = tensor.dim %[[SHAPE]], %[[C0]]
  // CHECK: return %[[RESULT]] : index
  %rank = shape.rank %shape : tensor<?xindex> -> index
  return %rank : index
}

// -----

// Don't lower `get_extent` if it is of type `shape.size`.
// CHECK-LABEL: @get_extent
func.func @get_extent(%shape : tensor<?xindex>, %idx : !shape.size) -> !shape.size {
  // CHECK: shape.get_extent
  %result = shape.get_extent %shape, %idx
      : tensor<?xindex>, !shape.size -> !shape.size
  return %result : !shape.size
}

// -----

// Don't lower `rank` if type is not error-free.
// CHECK-LABEL: @rank
func.func @rank(%shape : !shape.shape) {
  // CHECK: shape.rank
  %rank = shape.rank %shape : !shape.shape -> !shape.size
  return
}

// -----

// Express `shape.dim` as `tensor.dim` when valid.
// CHECK-LABEL: @dim
// CHECK-SAME:  (%[[ARG:.*]]: tensor<2x3xf32>, %[[IDX:.*]]: index) -> index
func.func @dim(%arg : tensor<2x3xf32>, %idx : index) -> index {
  // CHECK: %[[RESULT:.*]] = tensor.dim %[[ARG]], %[[IDX]] : tensor<2x3xf32>
  // CHECK: return %[[RESULT]] : index
  %result = shape.dim %arg, %idx : tensor<2x3xf32>, index -> index
  return %result : index
}

// -----

// Express `get_extent` as `tensor.dim` when it relies directly on the outcome of a
// `shape_of` operation.
// CHECK-LABEL: @get_extent_shape_of
// CHECK-SAME:  (%[[ARG:.*]]: tensor<2x3xf32>, %[[IDX:.*]]: index) -> index
func.func @get_extent_shape_of(%arg : tensor<2x3xf32>, %idx : index) -> index {
  // CHECK: %[[RESULT:.*]] = tensor.dim %[[ARG]], %[[IDX]] : tensor<2x3xf32>
  // CHECK: return %[[RESULT]] : index
  %shape = shape.shape_of %arg : tensor<2x3xf32> -> tensor<?xindex>
  %result = shape.get_extent %shape, %idx : tensor<?xindex>, index -> index
  return %result : index
}

// -----

// Express `get_extent` as `tensor.extract`.
// CHECK-LABEL: @get_extent_from_extent_tensor
// CHECK-SAME: (%[[EXTENTS:.*]]: tensor<?xindex>, %[[IDX:.*]]: index) -> index
func.func @get_extent_from_extent_tensor(%extents : tensor<?xindex>, %idx : index)
    -> index {
  // CHECK: %[[RESULT:.*]] = tensor.extract %[[EXTENTS]][%[[IDX]]] : tensor<?xindex>
  // CHECK: return %[[RESULT]] : index
  %result = shape.get_extent %extents, %idx : tensor<?xindex>, index -> index
  return %result : index
}

// -----

// Lower `const_shape` to `tensor.from_elements`.
// CHECK-LABEL: @const_shape
// CHECK-SAME: () -> tensor<3xindex>
func.func @const_shape() -> tensor<3xindex> {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[TENSOR3:.*]] = tensor.from_elements %[[C1]], %[[C2]], %[[C3]]
  // CHECK: %[[RESULT:.*]] = tensor.cast %[[TENSOR3]] : tensor<3xindex> to tensor<3xindex>
  // CHECK: return %[[RESULT]] : tensor<3xindex>
  %shape = shape.const_shape [1, 2, 3] : tensor<3xindex>
  return %shape : tensor<3xindex>
}

// -----

// Lower `const_shape` in the case of rank 0.
// CHECK-LABEL: func @const_shape_zero_elements
// CHECK-SAME: () -> tensor<0xindex>
func.func @const_shape_zero_elements() -> tensor<0xindex> {
  // CHECK: %[[TENSOR:.*]] = tensor.from_elements : tensor<0xindex>
  // CHECK: %[[RESULT:.*]] = tensor.cast %[[TENSOR]] : tensor<0xindex> to tensor<0xindex>
  // CHECK: return %[[RESULT]] : tensor<0xindex>
  %shape = shape.const_shape [] : tensor<0xindex>
  return %shape : tensor<0xindex>
}

// -----

// Lower `any` to its first operand.
// CHECK-LABEL: @any_of_three
// CHECK-SAME:  (%[[A:.*]]: tensor<?xindex>, %[[B:.*]]: tensor<?xindex>, %[[C:.*]]: tensor<?xindex>) -> tensor<?xindex>
func.func @any_of_three(%a : tensor<?xindex>,
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
func.func @any_of_one(%a : tensor<?xindex>) -> tensor<?xindex> {
  // CHECK: return %[[A]] : tensor<?xindex>
  %result = "shape.any"(%a) : (tensor<?xindex>) -> tensor<?xindex>
  return %result : tensor<?xindex>
}

// -----

// Lower 'const_size` to `arith.constant`
// CHECK-LABEL: @const_size
func.func @const_size() -> index {
  // CHECK: %[[RES:.*]] = arith.constant 42 : index
  %size = shape.const_size 42
  %result = shape.size_to_index %size : !shape.size
  // CHECK: return %[[RES]]
  return %result : index
}

// -----

// Lower `to_extent_tensor` to `tensor.cast`
// Fold to_extent_tensor when already on tensor.
// CHECK-LABEL: @to_extent_tensor
// CHECK-SAME: (%[[ARG:.*]]: tensor<?xindex>
func.func @to_extent_tensor(%arg: tensor<?xindex>) -> tensor<3xindex> {
  // CHECK-NOT: to_extent_tensor
  // CHECK: %[[RES:.*]] = tensor.cast %[[ARG]] : tensor<?xindex> to tensor<3xindex
  %casted = shape.to_extent_tensor %arg : tensor<?xindex> -> tensor<3xindex>
  // CHECK: return %[[RES]]
  return %casted : tensor<3xindex>
}

// CHECK-LABEL: @shape_reduce
// CHECK-SAME:  (%[[SHAPE:.*]]: tensor<?xindex>) -> index
func.func @shape_reduce(%shape : tensor<?xindex>) -> index {
  %init = arith.constant 1 : index
  %num_elements = shape.reduce(%shape, %init) : tensor<?xindex> -> index {
    ^bb0(%index : index, %extent : index, %acc: index):
      %new_acc = arith.muli %acc, %extent : index
      shape.yield %new_acc : index
  }
  return %num_elements : index
}
// CHECK-NEXT: %[[INIT:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[RANK:.*]] = tensor.dim %[[SHAPE]], %[[C0]] : tensor<?xindex>
// CHECK-NEXT: %[[RESULT:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[RANK]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[INIT]]) -> (index)
// CHECK-NEXT:   %[[EXTENT:.*]] = tensor.extract %[[SHAPE]][%[[I]]]
// CHECK-NEXT:   %[[NEW_ACC:.*]] = arith.muli %[[ACC]], %[[EXTENT]] : index
// CHECK-NEXT:   scf.yield %[[NEW_ACC]] : index
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]] : index

// -----

// Don't lower `shape_of` for result type of `shape.shape`.
// CHECK-LABEL: @shape_of
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>)
func.func @shape_of(%arg : tensor<*xf32>) {
  // CHECK: shape.shape
  %shape = shape.shape_of %arg : tensor<*xf32> -> !shape.shape
  return
}

// -----

// Lower `shape_of` for unranked tensors.
// CHECK-LABEL: @shape_of_unranked
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>)
func.func @shape_of_unranked(%arg : tensor<*xf32>) {
  // CHECK: %[[RANK:.*]] = tensor.rank %[[ARG]] : tensor<*xf32>
  // CHECK: %[[SHAPE:.*]] = tensor.generate %[[RANK]] {
  // CHECK: ^bb0(%[[I:.*]]: index):
  // CHECK:   %[[EXTENT:.*]] = tensor.dim %[[ARG]], %[[I]] : tensor<*xf32>
  // CHECK:   yield %[[EXTENT]] : index
  // CHECK: } : tensor<?xindex>
  %shape = shape.shape_of %arg : tensor<*xf32> -> tensor<?xindex>
  return
}

// -----

// Don't lower `shape_of` with `shape.shape` type.
// CHECK-LABEL: @shape_of
// CHECK-SAME: (%[[ARG:.*]]: tensor<1x2x3xf32>)
func.func @shape_of_stat(%arg : tensor<1x2x3xf32>) {
  // CHECK: shape.shape_of %[[ARG]] : tensor<1x2x3xf32> -> !shape.shape
  %shape = shape.shape_of %arg : tensor<1x2x3xf32> -> !shape.shape
  return
}

// -----

// Lower `shape_of` for statically shaped tensor.
// CHECK-LABEL: @shape_of_stat
// CHECK-SAME: (%[[ARG:.*]]: tensor<1x2x3xf32>)
func.func @shape_of_stat(%arg : tensor<1x2x3xf32>) {
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[SHAPE_UNCASTED:.*]] = tensor.from_elements %[[C1]], %[[C2]], %[[C3]] : tensor<3xindex>
  %shape = shape.shape_of %arg : tensor<1x2x3xf32> -> tensor<?xindex>
  return
}

// -----

// Lower `shape_of` for 0-D tensor.
// CHECK-LABEL: @shape_of_zero_d
// CHECK-SAME: (%[[ARG:.*]]: tensor<f32>)
func.func @shape_of_zero_d(%arg : tensor<f32>) {
  // CHECK-DAG: %[[SHAPE_UNCASTED:.*]] = tensor.from_elements : tensor<0xindex>
  %shape = shape.shape_of %arg : tensor<f32> -> tensor<?xindex>
  return
}

// -----

// Lower `shape_of` for dynamically shaped tensor.
// CHECK-LABEL: @shape_of_dyn
// CHECK-SAME: (%[[ARG:.*]]: tensor<1x5x?xf32>)
func.func @shape_of_dyn(%arg : tensor<1x5x?xf32>) {
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[DYN_DIM:.*]] = tensor.dim %[[ARG]], %[[C2]] : tensor<1x5x?xf32>
  // CHECK-DAG: %[[SHAPE_UNCASTED:.*]] = tensor.from_elements %[[C1]], %[[C5]], %[[DYN_DIM]] : tensor<3xindex>
  %shape = shape.shape_of %arg : tensor<1x5x?xf32> -> tensor<?xindex>
  return
}

// -----

// CHECK-LABEL:  @shape_eq
// CHECK-SAME:   (%[[A:.*]]: tensor<?xindex>, %[[B:.*]]: tensor<?xindex>) -> i1
func.func @shape_eq(%a : tensor<?xindex>, %b : tensor<?xindex>) -> i1 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[RANK_A:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?xindex>
  // CHECK: %[[RANK_B:.*]] = tensor.dim %[[B]], %[[C0]] : tensor<?xindex>
  // CHECK: %[[RANK_EQ:.*]] = arith.cmpi eq, %[[RANK_A]], %[[RANK_B]]
  // CHECK: %[[SHAPE_EQ:.*]] = scf.if %[[RANK_EQ]] -> (i1) {
  // CHECK:   %[[C1:.*]] = arith.constant 1 : index
  // CHECK:   %[[INIT:.*]] = arith.constant true
  // CHECK:   %[[SHAPE_EQ_INNER:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[RANK_A]] step %[[C1]] iter_args(%[[CONJ:.*]] = %[[INIT]]) -> (i1) {
  // CHECK:     %[[EXTENT_A:.*]] = tensor.extract %[[A]][%[[I]]] : tensor<?xindex>
  // CHECK:     %[[EXTENT_B:.*]] = tensor.extract %[[B]][%[[I]]] : tensor<?xindex>
  // CHECK:     %[[EXTENT_EQ:.*]] = arith.cmpi eq, %[[EXTENT_A]], %[[EXTENT_B]]
  // CHECK:     %[[CONJ_NEXT:.*]] = arith.andi %[[CONJ]], %[[EXTENT_EQ]]
  // CHECK:     scf.yield %[[CONJ_NEXT]] : i1
  // CHECK:   }
  // CHECK:   scf.yield %[[SHAPE_EQ_INNER]] : i1
  // CHECK: } else {
  // CHECK:   %[[SHAPE_EQ_INNER:.*]] = arith.constant false
  // CHECK:   scf.yield %[[SHAPE_EQ_INNER]] : i1
  // CHECK: }
  // CHECK: return %[[SHAPE_EQ]] : i1
  %result = shape.shape_eq %a, %b : tensor<?xindex>, tensor<?xindex>
  return %result : i1
}

// -----

// CHECK-LABEL:  @shape_eq
// CHECK-SAME:   (%[[A:.*]]: tensor<?xindex>, %[[B:.*]]: tensor<?xindex>, %[[C:.*]]: tensor<?xindex>) -> i1
func.func @shape_eq(%a : tensor<?xindex>, %b : tensor<?xindex>, %c : tensor<?xindex>) -> i1 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[RANK_A:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?xindex>
  // CHECK: %[[RANK_B:.*]] = tensor.dim %[[B]], %[[C0]] : tensor<?xindex>
  // CHECK: %[[RANK_EQ:.*]] = arith.cmpi eq, %[[RANK_A]], %[[RANK_B]]
  // CHECK: %[[SHAPE_EQ:.*]] = scf.if %[[RANK_EQ]] -> (i1) {
  // CHECK:   %[[C1:.*]] = arith.constant 1 : index
  // CHECK:   %[[INIT:.*]] = arith.constant true
  // CHECK:   %[[SHAPE_EQ_INNER:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[RANK_A]] step %[[C1]] iter_args(%[[CONJ:.*]] = %[[INIT]]) -> (i1) {
  // CHECK:     %[[EXTENT_A:.*]] = tensor.extract %[[A]][%[[I]]] : tensor<?xindex>
  // CHECK:     %[[EXTENT_B:.*]] = tensor.extract %[[B]][%[[I]]] : tensor<?xindex>
  // CHECK:     %[[EXTENT_EQ:.*]] = arith.cmpi eq, %[[EXTENT_A]], %[[EXTENT_B]]
  // CHECK:     %[[CONJ_NEXT:.*]] = arith.andi %[[CONJ]], %[[EXTENT_EQ]]
  // CHECK:     scf.yield %[[CONJ_NEXT]] : i1
  // CHECK:   }
  // CHECK:   scf.yield %[[SHAPE_EQ_INNER]] : i1
  // CHECK: } else {
  // CHECK:   %[[SHAPE_EQ_INNER:.*]] = arith.constant false
  // CHECK:   scf.yield %[[SHAPE_EQ_INNER]] : i1
  // CHECK: }
  // CHECK: %[[RANK_C:.*]] = tensor.dim %[[C]], %[[C0]] : tensor<?xindex>
  // CHECK: %[[RANK_EQ:.*]] = arith.cmpi eq, %[[RANK_A]], %[[RANK_C]]
  // CHECK: %[[SHAPE_EQ2:.*]] = scf.if %[[RANK_EQ]] -> (i1) {
  // CHECK:   %[[C1:.*]] = arith.constant 1 : index
  // CHECK:   %[[INIT:.*]] = arith.constant true
  // CHECK:   %[[SHAPE_EQ_INNER:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[RANK_A]] step %[[C1]] iter_args(%[[CONJ:.*]] = %[[INIT]]) -> (i1) {
  // CHECK:     %[[EXTENT_A:.*]] = tensor.extract %[[A]][%[[I]]] : tensor<?xindex>
  // CHECK:     %[[EXTENT_C:.*]] = tensor.extract %[[C]][%[[I]]] : tensor<?xindex>
  // CHECK:     %[[EXTENT_EQ:.*]] = arith.cmpi eq, %[[EXTENT_A]], %[[EXTENT_C]]
  // CHECK:     %[[CONJ_NEXT:.*]] = arith.andi %[[CONJ]], %[[EXTENT_EQ]]
  // CHECK:     scf.yield %[[CONJ_NEXT]] : i1
  // CHECK:   }
  // CHECK:   scf.yield %[[SHAPE_EQ_INNER]] : i1
  // CHECK: } else {
  // CHECK:   %[[SHAPE_EQ_INNER:.*]] = arith.constant false
  // CHECK:   scf.yield %[[SHAPE_EQ_INNER]] : i1
  // CHECK: }
  // CHECK: %[[RESULT:.*]] = arith.andi %[[SHAPE_EQ]], %[[SHAPE_EQ2]] : i1
  // CHECK: return %[[RESULT]] : i1
  %result = shape.shape_eq %a, %b, %c : tensor<?xindex>, tensor<?xindex>, tensor<?xindex>
  return %result : i1
}

// -----

// Don't lower `shape.broadcast` if a `shape.shape` type is involved.
// CHECK-LABEL: @broadcast
func.func @broadcast(%a : tensor<?xindex>, %b : !shape.shape) -> !shape.shape {
  // CHECK: shape.broadcast
  %c = shape.broadcast %a, %b : tensor<?xindex>, !shape.shape -> !shape.shape
  return %c : !shape.shape
}

// -----

func.func @try_is_broadcastable (%a : tensor<2xindex>, %b : tensor<3xindex>, %c : tensor<2xindex>) -> i1 {
  %0 = shape.is_broadcastable %a, %b, %c : tensor<2xindex>, tensor<3xindex>, tensor<2xindex>
  return %0 : i1
}
// CHECK-LABEL: @try_is_broadcastable
// CHECK-SAME:          %[[ARG0:.*]]: tensor<2xindex>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<3xindex>,
// CHECK-SAME:          %[[ARG2:.*]]: tensor<2xindex>)
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[RANK0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<2xindex>
// CHECK:           %[[RANK1:.*]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<3xindex>
// CHECK:           %[[RANK2:.*]] = tensor.dim %[[ARG2]], %[[C0]] : tensor<2xindex>
// CHECK:           %[[CMP0:.*]] = arith.cmpi ugt, %[[RANK1]], %[[RANK0]] : index
// CHECK:           %[[LARGER_DIM:.*]] = arith.select %[[CMP0]], %[[RANK1]], %[[RANK0]] : index
// CHECK:           %[[CMP1:.*]] = arith.cmpi ugt, %[[RANK2]], %[[LARGER_DIM]] : index
// CHECK:           %[[MAX_RANK:.*]] = arith.select %[[CMP1]], %[[RANK2]], %[[LARGER_DIM]] : index
// CHECK:           %[[DIM_DIFF0:.*]] = arith.subi %[[MAX_RANK]], %[[RANK0]] : index
// CHECK:           %[[DIM_DIFF1:.*]] = arith.subi %[[MAX_RANK]], %[[RANK1]] : index
// CHECK:           %[[DIM_DIFF2:.*]] = arith.subi %[[MAX_RANK]], %[[RANK2]] : index
// CHECK:           %[[TRUE:.*]] = arith.constant true
// CHECK:           %[[ALL_RESULT:.*]] = scf.for %[[IDX:.*]] = %[[C0]] to %[[MAX_RANK]] step %[[C1]] iter_args(%[[ALL_SO_FAR:.*]] = %[[TRUE]]) -> (i1) {
// CHECK:             %[[C1_0:.*]] = arith.constant 1 : index
// CHECK:             %[[OUTBOUNDS0:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF0]] : index
// CHECK:             %[[DIM0:.*]] = scf.if %[[OUTBOUNDS0]] -> (index) {
// CHECK:               scf.yield %[[C1_0]] : index
// CHECK:             } else {
// CHECK:               %[[IDX0:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF0]] : index
// CHECK:               %[[EXTRACTED_0:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[IDX0]]] : tensor<2xindex>
// CHECK:               %[[DIM0_IS_1:.*]] = arith.cmpi eq, %[[EXTRACTED_0:.*]], %[[C1_0]] : index
// CHECK:               %[[MAX_DIM0:.*]] = arith.select %[[DIM0_IS_1]], %[[C1_0]], %[[EXTRACTED_0]] : index
// CHECK:             }
// CHECK:             %[[VAL_28:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF1]] : index
// CHECK:             %[[DIM1:.*]] = scf.if %[[VAL_28]] -> (index) {
// CHECK:               scf.yield %[[DIM0]] : index
// CHECK:             } else {
// CHECK:               %[[IDX1:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF1]] : index
// CHECK:               %[[EXTRACTED_1:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[IDX1]]] : tensor<3xindex>
// CHECK:               %[[DIM1_IS_1:.*]] = arith.cmpi eq, %[[EXTRACTED_1:.*]], %[[C1_0]] : index
// CHECK:               %[[MAX_DIM1:.*]] = arith.select %[[DIM1_IS_1]], %[[DIM0]], %[[EXTRACTED_1]] : index
// CHECK:             }
// CHECK:             %[[VAL_36:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF2]] : index
// CHECK:             %[[DIM2:.*]] = scf.if %[[VAL_36]] -> (index) {
// CHECK:               scf.yield %[[DIM1]] : index
// CHECK:             } else {
// CHECK:               %[[IDX2:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF2]] : index
// CHECK:               %[[EXTRACTED_2:.*]] = tensor.extract %[[ARG2]]{{\[}}%[[IDX2]]] : tensor<2xindex>
// CHECK:               %[[DIM2_IS_1:.*]] = arith.cmpi eq, %[[EXTRACTED_2]], %[[C1_0]] : index
// CHECK:               %[[MAX_DIM2:.*]] = arith.select %[[DIM2_IS_1]], %[[DIM1]], %[[EXTRACTED_2]] : index
// CHECK:             }
// CHECK:             %[[OUT_BOUND_0:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF0]] : index
// CHECK:             %[[REDUCTION_0:.*]] = scf.if %[[OUT_BOUND_0]] -> (i1) {
// CHECK:                scf.yield %[[ALL_SO_FAR]] : i1
// CHECK:             } else {
// CHECK:                %[[SHIFTED:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF0]] : index
// CHECK:                %[[EXTRACTED:.*]] = tensor.extract %arg0[%[[SHIFTED]]] : tensor<2xindex>
// CHECK:                %[[EQUALS_1:.*]] = arith.cmpi eq, %[[EXTRACTED]], %c1 : index
// CHECK:                %[[EQUALS_BROADCASTED:.*]] = arith.cmpi eq, %[[EXTRACTED]], %[[DIM2]] : index
// CHECK:                %[[GOOD:.*]] = arith.ori %[[EQUALS_1]], %[[EQUALS_BROADCASTED]] : i1
// CHECK:                %[[AND_REDUCTION:.*]] = arith.andi %[[ALL_SO_FAR]], %[[GOOD]] : i1
// CHECK:                scf.yield %[[AND_REDUCTION]] : i1
// CHECK:             }
// CHECK:             %[[OUT_BOUND_1:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF1]] : index
// CHECK:             %[[SECOND_REDUCTION:.*]] = scf.if %[[OUT_BOUND_1]] -> (i1) {
// CHECK:                scf.yield %[[REDUCTION_0]] : i1
// CHECK:             } else {
// CHECK:                %[[SHIFTED:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF1]] : index
// CHECK:                %[[EXTRACTED:.*]] = tensor.extract %arg1[%[[SHIFTED]]] : tensor<3xindex>
// CHECK:                %[[EQUALS_1:.*]] = arith.cmpi eq, %[[EXTRACTED]], %c1 : index
// CHECK:                %[[EQUALS_BROADCASTED:.*]] = arith.cmpi eq, %[[EXTRACTED]], %[[DIM2]] : index
// CHECK:                %[[GOOD:.*]] = arith.ori %[[EQUALS_1]], %[[EQUALS_BROADCASTED]] : i1
// CHECK:                %[[AND_REDUCTION:.*]] = arith.andi %[[REDUCTION_0]], %[[GOOD]] : i1
// CHECK:                scf.yield %[[AND_REDUCTION]] : i1
// CHECK:             }
// CHECK:             %[[OUT_BOUND_2:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF2]] : index
// CHECK:             %[[FINAL_RESULT:.*]] = scf.if %[[OUT_BOUND_2]] -> (i1) {
// CHECK:                scf.yield %[[SECOND_REDUCTION]] : i1
// CHECK:             } else {
// CHECK:                %[[SHIFTED:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF2]] : index
// CHECK:                %[[EXTRACTED:.*]] = tensor.extract %arg2[%[[SHIFTED]]] : tensor<2xindex>
// CHECK:                %[[EQUALS_1:.*]] = arith.cmpi eq, %[[EXTRACTED:.*]], %c1 : index
// CHECK:                %[[EQUALS_BROADCASTED:.*]] = arith.cmpi eq, %[[EXTRACTED:.*]], %[[DIM2]] : index
// CHECK:                %[[GOOD:.*]] = arith.ori %[[EQUALS_1:.*]], %[[EQUALS_BROADCASTED:.*]] : i1
// CHECK:                %[[AND_REDUCTION:.*]] = arith.andi %[[SECOND_REDUCTION]], %[[GOOD]] : i1
// CHECK:                scf.yield %[[AND_REDUCTION]] : i1
// CHECK:             }
// CHECK:             scf.yield %[[FINAL_RESULT]] : i1

// -----

func.func @broadcast(%a : tensor<2xindex>, %b : tensor<3xindex>, %c : tensor<2xindex>) -> !shape.witness {
  %0 = shape.cstr_broadcastable %a, %b, %c : tensor<2xindex>, tensor<3xindex>, tensor<2xindex>
  return %0 : !shape.witness
}
// CHECK-LABEL:   func @broadcast(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<2xindex>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<3xindex>,
// CHECK-SAME:          %[[ARG2:.*]]: tensor<2xindex>)
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[RANK0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<2xindex>
// CHECK:           %[[RANK1:.*]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<3xindex>
// CHECK:           %[[RANK2:.*]] = tensor.dim %[[ARG2]], %[[C0]] : tensor<2xindex>
// CHECK:           %[[CMP0:.*]] = arith.cmpi ugt, %[[RANK1]], %[[RANK0]] : index
// CHECK:           %[[LARGER_DIM:.*]] = arith.select %[[CMP0]], %[[RANK1]], %[[RANK0]] : index
// CHECK:           %[[CMP1:.*]] = arith.cmpi ugt, %[[RANK2]], %[[LARGER_DIM]] : index
// CHECK:           %[[MAX_RANK:.*]] = arith.select %[[CMP1]], %[[RANK2]], %[[LARGER_DIM]] : index
// CHECK:           %[[DIM_DIFF0:.*]] = arith.subi %[[MAX_RANK]], %[[RANK0]] : index
// CHECK:           %[[DIM_DIFF1:.*]] = arith.subi %[[MAX_RANK]], %[[RANK1]] : index
// CHECK:           %[[DIM_DIFF2:.*]] = arith.subi %[[MAX_RANK]], %[[RANK2]] : index
// CHECK:           %[[TRUE:.*]] = arith.constant true
// CHECK:           %[[ALL_RESULT:.*]] = scf.for %[[IDX:.*]] = %[[C0]] to %[[MAX_RANK]] step %[[C1]] iter_args(%[[ALL_SO_FAR:.*]] = %[[TRUE]]) -> (i1) {
// CHECK:             %[[C1_0:.*]] = arith.constant 1 : index
// CHECK:             %[[OUTBOUNDS0:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF0]] : index
// CHECK:             %[[DIM0:.*]] = scf.if %[[OUTBOUNDS0]] -> (index) {
// CHECK:               scf.yield %[[C1_0]] : index
// CHECK:             } else {
// CHECK:               %[[IDX0:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF0]] : index
// CHECK:               %[[EXTRACTED_0:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[IDX0]]] : tensor<2xindex>
// CHECK:               %[[DIM0_IS_1:.*]] = arith.cmpi eq, %[[EXTRACTED_0:.*]], %[[C1_0]] : index
// CHECK:               %[[MAX_DIM0:.*]] = arith.select %[[DIM0_IS_1]], %[[C1_0]], %[[EXTRACTED_0]] : index
// CHECK:             }
// CHECK:             %[[VAL_28:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF1]] : index
// CHECK:             %[[DIM1:.*]] = scf.if %[[VAL_28]] -> (index) {
// CHECK:               scf.yield %[[DIM0]] : index
// CHECK:             } else {
// CHECK:               %[[IDX1:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF1]] : index
// CHECK:               %[[EXTRACTED_1:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[IDX1]]] : tensor<3xindex>
// CHECK:               %[[DIM1_IS_1:.*]] = arith.cmpi eq, %[[EXTRACTED_1:.*]], %[[C1_0]] : index
// CHECK:               %[[MAX_DIM1:.*]] = arith.select %[[DIM1_IS_1]], %[[DIM0]], %[[EXTRACTED_1]] : index
// CHECK:             }
// CHECK:             %[[VAL_36:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF2]] : index
// CHECK:             %[[DIM2:.*]] = scf.if %[[VAL_36]] -> (index) {
// CHECK:               scf.yield %[[DIM1]] : index
// CHECK:             } else {
// CHECK:               %[[IDX2:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF2]] : index
// CHECK:               %[[EXTRACTED_2:.*]] = tensor.extract %[[ARG2]]{{\[}}%[[IDX2]]] : tensor<2xindex>
// CHECK:               %[[DIM2_IS_1:.*]] = arith.cmpi eq, %[[EXTRACTED_2]], %[[C1_0]] : index
// CHECK:               %[[MAX_DIM2:.*]] = arith.select %[[DIM2_IS_1]], %[[DIM1]], %[[EXTRACTED_2]] : index
// CHECK:             }
// CHECK:             %[[OUT_BOUND_0:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF0]] : index
// CHECK:             %[[REDUCTION_0:.*]] = scf.if %[[OUT_BOUND_0]] -> (i1) {
// CHECK:                scf.yield %[[ALL_SO_FAR]] : i1
// CHECK:             } else {
// CHECK:                %[[SHIFTED:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF0]] : index
// CHECK:                %[[EXTRACTED:.*]] = tensor.extract %arg0[%[[SHIFTED]]] : tensor<2xindex>
// CHECK:                %[[EQUALS_1:.*]] = arith.cmpi eq, %[[EXTRACTED]], %c1 : index
// CHECK:                %[[EQUALS_BROADCASTED:.*]] = arith.cmpi eq, %[[EXTRACTED]], %[[DIM2]] : index
// CHECK:                %[[GOOD:.*]] = arith.ori %[[EQUALS_1]], %[[EQUALS_BROADCASTED]] : i1
// CHECK:                %[[AND_REDUCTION:.*]] = arith.andi %[[ALL_SO_FAR]], %[[GOOD]] : i1
// CHECK:                scf.yield %[[AND_REDUCTION]] : i1
// CHECK:             }
// CHECK:             %[[OUT_BOUND_1:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF1]] : index
// CHECK:             %[[SECOND_REDUCTION:.*]] = scf.if %[[OUT_BOUND_1]] -> (i1) {
// CHECK:                scf.yield %[[REDUCTION_0]] : i1
// CHECK:             } else {
// CHECK:                %[[SHIFTED:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF1]] : index
// CHECK:                %[[EXTRACTED:.*]] = tensor.extract %arg1[%[[SHIFTED]]] : tensor<3xindex>
// CHECK:                %[[EQUALS_1:.*]] = arith.cmpi eq, %[[EXTRACTED]], %c1 : index
// CHECK:                %[[EQUALS_BROADCASTED:.*]] = arith.cmpi eq, %[[EXTRACTED]], %[[DIM2]] : index
// CHECK:                %[[GOOD:.*]] = arith.ori %[[EQUALS_1]], %[[EQUALS_BROADCASTED]] : i1
// CHECK:                %[[AND_REDUCTION:.*]] = arith.andi %[[REDUCTION_0]], %[[GOOD]] : i1
// CHECK:                scf.yield %[[AND_REDUCTION]] : i1
// CHECK:             }
// CHECK:             %[[OUT_BOUND_2:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF2]] : index
// CHECK:             %[[FINAL_RESULT:.*]] = scf.if %[[OUT_BOUND_2]] -> (i1) {
// CHECK:                scf.yield %[[SECOND_REDUCTION]] : i1
// CHECK:             } else {
// CHECK:                %[[SHIFTED:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF2]] : index
// CHECK:                %[[EXTRACTED:.*]] = tensor.extract %arg2[%[[SHIFTED]]] : tensor<2xindex>
// CHECK:                %[[EQUALS_1:.*]] = arith.cmpi eq, %[[EXTRACTED:.*]], %c1 : index
// CHECK:                %[[EQUALS_BROADCASTED:.*]] = arith.cmpi eq, %[[EXTRACTED:.*]], %[[DIM2]] : index
// CHECK:                %[[GOOD:.*]] = arith.ori %[[EQUALS_1:.*]], %[[EQUALS_BROADCASTED:.*]] : i1
// CHECK:                %[[AND_REDUCTION:.*]] = arith.andi %[[SECOND_REDUCTION]], %[[GOOD]] : i1
// CHECK:                scf.yield %[[AND_REDUCTION]] : i1
// CHECK:             }
// CHECK:             scf.yield %[[FINAL_RESULT]] : i1

// CHECK:           %[[RESULT:.*]] = shape.cstr_require %[[ALL_RESULT]], "required broadcastable shapes"
// CHECK:           return %[[RESULT]] : !shape.witness
// CHECK:         }

// -----

func.func @broadcast_3_shapes_different_extents(%a : tensor<2xindex>,
                                           %b : tensor<3xindex>,
                                           %c : tensor<2xindex>) {
// CHECK-LABEL:   func @broadcast_3_shapes_different_extents(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<2xindex>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<3xindex>,
// CHECK-SAME:          %[[ARG2:.*]]: tensor<2xindex>) {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[RANK0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<2xindex>
// CHECK:           %[[RANK1:.*]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<3xindex>
// CHECK:           %[[RANK2:.*]] = tensor.dim %[[ARG2]], %[[C0]] : tensor<2xindex>
// CHECK:           %[[CMP0:.*]] = arith.cmpi ugt, %[[RANK1]], %[[RANK0]] : index
// CHECK:           %[[LARGER_DIM:.*]] = arith.select %[[CMP0]], %[[RANK1]], %[[RANK0]] : index
// CHECK:           %[[CMP1:.*]] = arith.cmpi ugt, %[[RANK2]], %[[LARGER_DIM]] : index
// CHECK:           %[[MAX_RANK:.*]] = arith.select %[[CMP1]], %[[RANK2]], %[[LARGER_DIM]] : index
// CHECK:           %[[DIM_DIFF0:.*]] = arith.subi %[[MAX_RANK]], %[[RANK0]] : index
// CHECK:           %[[DIM_DIFF1:.*]] = arith.subi %[[MAX_RANK]], %[[RANK1]] : index
// CHECK:           %[[DIM_DIFF2:.*]] = arith.subi %[[MAX_RANK]], %[[RANK2]] : index
// CHECK:           %[[RESULT:.*]] = tensor.generate %[[MAX_RANK]]  {
// CHECK:           ^bb0(%[[IDX:.*]]: index):
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[OUTBOUNDS0:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF0]] : index
// CHECK:             %[[DIM0:.*]] = scf.if %[[OUTBOUNDS0]] -> (index) {
// CHECK:               scf.yield %[[C1]] : index
// CHECK:             } else {
// CHECK:               %[[IDX0:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF0]] : index
// CHECK:               %[[EXTRACTED_0:.*]] = tensor.extract %[[ARG0]]{{\[}}%[[IDX0]]] : tensor<2xindex>
// CHECK:               %[[DIM0_IS_1:.*]] = arith.cmpi eq, %[[EXTRACTED_0:.*]], %[[C1]] : index
// CHECK:               %[[MAX_DIM0:.*]] = arith.select %[[DIM0_IS_1]], %[[C1]], %[[EXTRACTED_0]] : index
// CHECK:             }
// CHECK:             %[[VAL_28:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF1]] : index
// CHECK:             %[[DIM1:.*]] = scf.if %[[VAL_28]] -> (index) {
// CHECK:               scf.yield %[[DIM0]] : index
// CHECK:             } else {
// CHECK:               %[[IDX1:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF1]] : index
// CHECK:               %[[EXTRACTED_1:.*]] = tensor.extract %[[ARG1]]{{\[}}%[[IDX1]]] : tensor<3xindex>
// CHECK:               %[[DIM1_IS_1:.*]] = arith.cmpi eq, %[[EXTRACTED_1:.*]], %[[C1]] : index
// CHECK:               %[[MAX_DIM1:.*]] = arith.select %[[DIM1_IS_1]], %[[DIM0]], %[[EXTRACTED_1]] : index
// CHECK:             }
// CHECK:             %[[VAL_36:.*]] = arith.cmpi ult, %[[IDX]], %[[DIM_DIFF2]] : index
// CHECK:             %[[DIM2:.*]] = scf.if %[[VAL_36]] -> (index) {
// CHECK:               scf.yield %[[DIM1]] : index
// CHECK:             } else {
// CHECK:               %[[IDX2:.*]] = arith.subi %[[IDX]], %[[DIM_DIFF2]] : index
// CHECK:               %[[EXTRACTED_2:.*]] = tensor.extract %[[ARG2]]{{\[}}%[[IDX2]]] : tensor<2xindex>
// CHECK:               %[[DIM2_IS_1:.*]] = arith.cmpi eq, %[[EXTRACTED_2:.*]], %[[C1]] : index
// CHECK:               %[[MAX_DIM2:.*]] = arith.select %[[DIM2_IS_1]], %[[DIM1]], %[[EXTRACTED_2]] : index
// CHECK:             }
// CHECK:             tensor.yield %[[DIM2]] : index
// CHECK:           } : tensor<?xindex>
// CHECK:           return
// CHECK:         }
  %0 = shape.broadcast %a, %b, %c
      : tensor<2xindex>, tensor<3xindex>, tensor<2xindex> -> tensor<?xindex>
  return
}

// -----

// CHECK-LABEL: @broadcast_to_known_rank
func.func @broadcast_to_known_rank(%a : tensor<1xindex>, %b : tensor<3xindex>)
    -> tensor<3xindex> {
  // CHECK: %[[RES:.*]] = tensor.cast %{{.*}} : tensor<?xindex> to tensor<3xindex>
  // CHECK: return %[[RES]] : tensor<3xindex>
  %0 = shape.broadcast %a, %b : tensor<1xindex>, tensor<3xindex> -> tensor<3xindex>
  return %0 : tensor<3xindex>
}

// -----

// Lower `split_at`
// CHECK-LABEL: @split_at
// CHECK-SAME: %[[SHAPE:.*]]: tensor<?xindex>, %[[INDEX:.*]]: index
func.func @split_at(%shape: tensor<?xindex>, %index: index) -> (tensor<?xindex>, tensor<?xindex>) {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[RANK:.*]] = tensor.dim %[[SHAPE]], %[[C0]] : tensor<?xindex>
  // CHECK-NEXT: %[[POSINDEX:.*]] = arith.addi %[[INDEX]], %[[RANK]] : index
  // CHECK-NEXT: %[[ISNEG:.*]] = arith.cmpi slt, %[[INDEX]], %[[C0]] : index
  // CHECK-NEXT: %[[SELECT:.*]] = arith.select %[[ISNEG]], %[[POSINDEX]], %[[INDEX]] : index
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[HEAD:.*]] = tensor.extract_slice %[[SHAPE]][%[[C0]]] [%[[SELECT]]] [%[[C1]]] : tensor<?xindex> to tensor<?xindex>
  // CHECK-NEXT: %[[TAIL_SIZE:.*]] = arith.subi %[[RANK]], %[[SELECT]] : index
  // CHECK-NEXT: %[[TAIL:.*]] = tensor.extract_slice %[[SHAPE]][%[[SELECT]]] [%[[TAIL_SIZE]]] [%[[C1]]] : tensor<?xindex> to tensor<?xindex>
  // CHECK-NEXT: return %[[HEAD]], %[[TAIL]] : tensor<?xindex>, tensor<?xindex>
  %head, %tail = "shape.split_at"(%shape, %index) : (tensor<?xindex>, index) -> (tensor<?xindex>, tensor<?xindex>)
  return %head, %tail : tensor<?xindex>, tensor<?xindex>
}
