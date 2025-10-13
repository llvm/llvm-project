// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: create_vector_mask_to_constant_mask
func.func @create_vector_mask_to_constant_mask() -> (vector<4x3xi1>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK: vector.constant_mask [3, 2] : vector<4x3xi1>
  %0 = vector.create_mask %c3, %c2 : vector<4x3xi1>
  return %0 : vector<4x3xi1>
}

// -----

// CHECK-LABEL: create_scalable_vector_mask_to_constant_mask
func.func @create_scalable_vector_mask_to_constant_mask() -> (vector<[8]xi1>) {
  %c-1 = arith.constant -1 : index
  // CHECK: arith.constant dense<false> : vector<[8]xi1>
  %0 = vector.create_mask %c-1 : vector<[8]xi1>
  return %0 : vector<[8]xi1>
}

// -----

// CHECK-LABEL: create_vector_mask_to_constant_mask_truncation
func.func @create_vector_mask_to_constant_mask_truncation() -> (vector<4x3xi1>) {
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  // CHECK: vector.constant_mask [4, 2] : vector<4x3xi1>
  %0 = vector.create_mask %c5, %c2 : vector<4x3xi1>
  return %0 : vector<4x3xi1>
}

// -----

// CHECK-LABEL: create_vector_mask_to_constant_mask_truncation_neg
func.func @create_vector_mask_to_constant_mask_truncation_neg() -> (vector<4x3xi1>) {
  %cneg2 = arith.constant -2 : index
  %c5 = arith.constant 5 : index
  // CHECK: arith.constant dense<false> : vector<4x3xi1>
  %0 = vector.create_mask %c5, %cneg2 : vector<4x3xi1>
  return %0 : vector<4x3xi1>
}

// -----

// CHECK-LABEL: create_vector_mask_to_constant_mask_truncation_zero
func.func @create_vector_mask_to_constant_mask_truncation_zero() -> (vector<4x3xi1>) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  // CHECK: arith.constant dense<false> : vector<4x3xi1>
  %0 = vector.create_mask %c0, %c2 : vector<4x3xi1>
  return %0 : vector<4x3xi1>
}

// -----

// CHECK-LABEL: create_vector_mask_to_constant_mask_scalable_all_true
func.func @create_vector_mask_to_constant_mask_scalable_all_true() -> (vector<8x[16]xi1>) {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %0 = vector.vscale
  %1 = arith.muli %0, %c16 : index
  // CHECK: arith.constant dense<true> : vector<8x[16]xi1>
  %10 = vector.create_mask %c8, %1 : vector<8x[16]xi1>
  return %10 : vector<8x[16]xi1>
}

// -----

// CHECK-LABEL: create_mask_transpose_to_transposed_create_mask
//  CHECK-SAME: %[[DIM0:.*]]: index, %[[DIM1:.*]]: index, %[[DIM2:.*]]: index
func.func @create_mask_transpose_to_transposed_create_mask(
  %dim0: index, %dim1: index, %dim2: index) -> (vector<2x3x4xi1>, vector<4x2x3xi1>) {
  //     CHECK: vector.create_mask %[[DIM0]], %[[DIM1]], %[[DIM2]] : vector<2x3x4xi1>
  //     CHECK: vector.create_mask %[[DIM2]], %[[DIM0]], %[[DIM1]] : vector<4x2x3xi1>
  // CHECK-NOT: vector.transpose
  %0 = vector.create_mask %dim0, %dim1, %dim2 : vector<2x3x4xi1>
  %1 = vector.transpose %0, [2, 0, 1] : vector<2x3x4xi1> to vector<4x2x3xi1>
  return %0, %1 : vector<2x3x4xi1>, vector<4x2x3xi1>
}

// -----

// CHECK-LABEL: extract_from_create_mask
//  CHECK-SAME: %[[DIM0:.*]]: index, %[[DIM1:.*]]: index
func.func @extract_from_create_mask(%dim0: index, %dim1: index) -> vector<[4]x[4]xi1> {
  %c2 = arith.constant 2 : index
  %mask = vector.create_mask %c2, %dim0, %dim1 : vector<4x[4]x[4]xi1>
  // CHECK: vector.create_mask %[[DIM0]], %[[DIM1]] : vector<[4]x[4]xi1>
  // CHECK-NOT: vector.extract
  %extract = vector.extract %mask[1] : vector<[4]x[4]xi1> from vector<4x[4]x[4]xi1>
  return %extract : vector<[4]x[4]xi1>
}

// -----

// CHECK-LABEL: extract_from_create_mask_all_false
func.func @extract_from_create_mask_all_false(%dim0: index, %dim1: index) -> vector<[4]x[4]xi1> {
  %c2 = arith.constant 2 : index
  %mask = vector.create_mask %c2, %dim0, %dim1 : vector<4x[4]x[4]xi1>
  // CHECK: arith.constant dense<false> : vector<[4]x[4]xi1>
  // CHECK-NOT: vector.extract
  %extract = vector.extract %mask[2] : vector<[4]x[4]xi1> from vector<4x[4]x[4]xi1>
  return %extract : vector<[4]x[4]xi1>
}

// -----

// CHECK-LABEL: extract_from_create_mask_leading_scalable
//  CHECK-SAME: %[[DIM0:.*]]: index
func.func @extract_from_create_mask_leading_scalable(%dim0: index) -> vector<8xi1> {
  %c3 = arith.constant 3 : index
  %mask = vector.create_mask %c3, %dim0 : vector<[4]x8xi1>
  // CHECK: vector.create_mask %[[DIM0]] : vector<8xi1>
  // CHECK-NOT: vector.extract
  %extract = vector.extract %mask[1] : vector<8xi1> from vector<[4]x8xi1>
  return %extract : vector<8xi1>
}

// -----

// CHECK-LABEL: extract_from_create_mask_dynamic_position
//  CHECK-SAME: %[[DIM0:.*]]: index, %[[INDEX:.*]]: index
func.func @extract_from_create_mask_dynamic_position(%dim0: index, %index: index) -> vector<6xi1> {
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %mask = vector.create_mask %c3, %c4, %dim0 : vector<4x4x6xi1>
  // CHECK: vector.create_mask %[[DIM0]] : vector<6xi1>
  // CHECK-NOT: vector.extract
  %extract = vector.extract %mask[2, %index] : vector<6xi1> from vector<4x4x6xi1>
  return %extract : vector<6xi1>
}

// -----

// CHECK-LABEL: @extract_scalar_poison
func.func @extract_scalar_poison() -> f32 {
  // CHECK-NEXT: %[[UB:.*]] = ub.poison : f32
  //  CHECK-NOT: vector.extract
  // CHECK-NEXT: return %[[UB]] : f32
  %0 = ub.poison : vector<4x8xf32>
  %1 = vector.extract %0[2, 4] : f32 from vector<4x8xf32>
  return %1 : f32
}

// -----

// CHECK-LABEL: @extract_vector_poison
func.func @extract_vector_poison() -> vector<8xf32> {
  // CHECK-NEXT: %[[UB:.*]] = ub.poison : vector<8xf32>
  //  CHECK-NOT: vector.extract
  // CHECK-NEXT: return %[[UB]] : vector<8xf32>
  %0 = ub.poison : vector<4x8xf32>
  %1 = vector.extract %0[2] : vector<8xf32> from vector<4x8xf32>
  return %1 : vector<8xf32>
}

// -----

// CHECK-LABEL: @extract_scalar_poison_idx
func.func @extract_scalar_poison_idx(%a: vector<4x5xf32>) -> f32 {
  // CHECK-NEXT: %[[UB:.*]] = ub.poison : f32
  //  CHECK-NOT: vector.extract
  // CHECK-NEXT: return %[[UB]] : f32
  %0 = vector.extract %a[-1, 0] : f32 from vector<4x5xf32>
  return %0 : f32
}

// -----

// Similar to the test above, but the index is not a static constant.

// CHECK-LABEL: @extract_scalar_poison_idx_non_cst
func.func @extract_scalar_poison_idx_non_cst(%a: vector<4x5xf32>) -> f32 {
  // CHECK-NEXT: %[[UB:.*]] = ub.poison : f32
  //  CHECK-NOT: vector.extract
  // CHECK-NEXT: return %[[UB]] : f32
  %c_neg_1 = arith.constant -1 : index
  %0 = vector.extract %a[%c_neg_1, 0] : f32 from vector<4x5xf32>
  return %0 : f32
}

// -----

// Similar to test above, but now the index is out-of-bounds.

// CHECK-LABEL: @no_fold_extract_scalar_oob_idx
func.func @no_fold_extract_scalar_oob_idx(%a: vector<4x5xf32>) -> f32 {
  //  CHECK: vector.extract
  %c_neg_2 = arith.constant -2 : index
  %0 = vector.extract %a[%c_neg_2, 0] : f32 from vector<4x5xf32>
  return %0 : f32
}


// -----

// CHECK-LABEL: @extract_vector_poison_idx
func.func @extract_vector_poison_idx(%a: vector<4x5xf32>) -> vector<5xf32> {
  // CHECK-NEXT: %[[UB:.*]] = ub.poison : vector<5xf32>
  //  CHECK-NOT: vector.extract
  // CHECK-NEXT: return %[[UB]] : vector<5xf32>
  %0 = vector.extract %a[-1] : vector<5xf32> from vector<4x5xf32>
  return %0 : vector<5xf32>
}

// -----

// CHECK-LABEL: @extract_multiple_poison_idx
func.func @extract_multiple_poison_idx(%a: vector<4x5x8xf32>)
    -> vector<8xf32> {
  // CHECK-NEXT: %[[UB:.*]] = ub.poison : vector<8xf32>
  //  CHECK-NOT: vector.extract
  // CHECK-NEXT: return %[[UB]] : vector<8xf32>
  %0 = vector.extract %a[-1, -1] : vector<8xf32> from vector<4x5x8xf32>
  return %0 : vector<8xf32>
}

// -----

// CHECK-LABEL: extract_from_create_mask_dynamic_position_all_false
//  CHECK-SAME: %[[DIM0:.*]]: index, %[[INDEX:.*]]: index
func.func @extract_from_create_mask_dynamic_position_all_false(%dim0: index, %index: index) -> vector<6xi1> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %mask = vector.create_mask %c1, %c0, %dim0 : vector<1x4x6xi1>
  // CHECK: arith.constant dense<false> : vector<6xi1>
  // CHECK-NOT: vector.extract
  %extract = vector.extract %mask[0, %index] : vector<6xi1> from vector<1x4x6xi1>
  return %extract : vector<6xi1>
}

// -----

// CHECK-LABEL: extract_from_create_mask_dynamic_position_unknown
//  CHECK-SAME: %[[DIM0:.*]]: index, %[[INDEX:.*]]: index
func.func @extract_from_create_mask_dynamic_position_unknown(%dim0: index, %index: index) -> vector<6xi1> {
  %c2 = arith.constant 2 : index
  %mask = vector.create_mask %c2, %dim0 : vector<4x6xi1>
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-NEXT: %[[MASK:.*]] = vector.create_mask %[[C2]], %[[DIM0]] : vector<4x6xi1>
  // CHECK-NEXT: vector.extract %[[MASK]][%[[INDEX]]] : vector<6xi1> from vector<4x6xi1>
  %extract = vector.extract %mask[%index] : vector<6xi1> from vector<4x6xi1>
  return %extract : vector<6xi1>
}

// -----

// CHECK-LABEL: extract_from_create_mask_mixed_position_unknown
//  CHECK-SAME: %[[DIM0:.*]]: index, %[[INDEX:.*]]: index
func.func @extract_from_create_mask_mixed_position_unknown(%dim0: index, %index0: index) -> vector<4xi1> {
  %c2 = arith.constant 2 : index
  %mask = vector.create_mask %c2, %c2, %dim0 : vector<2x4x4xi1>
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-NEXT: %[[MASK:.*]] = vector.create_mask %[[C2]], %[[C2]], %[[DIM0]] : vector<2x4x4xi1>
  // CHECK-NEXT: vector.extract %[[MASK]][1, %[[INDEX]]] : vector<4xi1> from vector<2x4x4xi1>
  %extract = vector.extract %mask[1, %index0] : vector<4xi1> from vector<2x4x4xi1>
  return %extract : vector<4xi1>
}

// -----

// CHECK-LABEL: extract_from_non_constant_create_mask
//  CHECK-SAME: %[[DIM0:.*]]: index
func.func @extract_from_non_constant_create_mask(%dim0: index) -> vector<[2]xi1> {
  %mask = vector.create_mask %dim0, %dim0 : vector<[2]x[2]xi1>
  // CHECK: %[[MASK:.*]] = vector.create_mask %[[DIM0]], %[[DIM0]] : vector<[2]x[2]xi1>
  // CHECK-NEXT: vector.extract %[[MASK]][0] : vector<[2]xi1> from vector<[2]x[2]xi1>
  %extract = vector.extract %mask[0] : vector<[2]xi1> from vector<[2]x[2]xi1>
  return %extract : vector<[2]xi1>
}

// -----

// CHECK-LABEL: constant_mask_to_true_splat
func.func @constant_mask_to_true_splat() -> vector<2x4xi1> {
  // CHECK: arith.constant dense<true>
  // CHECK-NOT: vector.constant_mask
  %0 = vector.constant_mask [2, 4] : vector<2x4xi1>
  return %0 : vector<2x4xi1>
}

// CHECK-LABEL: constant_mask_to_false_splat
func.func @constant_mask_to_false_splat() -> vector<2x4xi1> {
  // CHECK: arith.constant dense<false>
  // CHECK-NOT: vector.constant_mask
  %0 = vector.constant_mask [0, 0] : vector<2x4xi1>
  return %0 : vector<2x4xi1>
}

// CHECK-LABEL: constant_mask_to_true_splat_0d
func.func @constant_mask_to_true_splat_0d() -> vector<i1> {
  // CHECK: arith.constant dense<true>
  // CHECK-NOT: vector.constant_mask
  %0 = vector.constant_mask [1] : vector<i1>
  return %0 : vector<i1>
}

// CHECK-LABEL: constant_mask_transpose_to_transposed_constant_mask
func.func @constant_mask_transpose_to_transposed_constant_mask() -> (vector<2x3x4xi1>, vector<4x2x3xi1>) {
  //     CHECK: vector.constant_mask [1, 2, 3] : vector<2x3x4xi1>
  //     CHECK: vector.constant_mask [3, 1, 2] : vector<4x2x3xi1>
  // CHECK-NOT: vector.transpose
  %0 = vector.constant_mask [1, 2, 3] : vector<2x3x4xi1>
  %1 = vector.transpose %0, [2, 0, 1] : vector<2x3x4xi1> to vector<4x2x3xi1>
  return %0, %1 : vector<2x3x4xi1>, vector<4x2x3xi1>
}

// -----

func.func @extract_strided_slice_of_constant_mask() -> (vector<2x2xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x2xi1>
  // CHECK: arith.constant dense<true> : vector<2x2xi1>
  return %1 : vector<2x2xi1>
}

// -----

func.func @extract_strided_slice_of_constant_mask() -> (vector<2x2xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [1, 0], sizes = [2, 2], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x2xi1>
  // CHECK: vector.constant_mask [1, 2] : vector<2x2xi1>
  return %1 : vector<2x2xi1>
}

// -----

func.func @extract_strided_slice_of_constant_mask() -> (vector<2x2xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [0, 1], sizes = [2, 2], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x2xi1>
  // CHECK: vector.constant_mask [2, 1] : vector<2x2xi1>
  return %1 : vector<2x2xi1>
}

// -----

func.func @extract_strided_slice_of_constant_mask() -> (vector<2x2xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x2xi1>
  // CHECK: arith.constant dense<false> : vector<2x2xi1>
  return %1 : vector<2x2xi1>
}

// -----

func.func @extract_strided_slice_of_constant_mask() -> (vector<2x1xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [0, 2], sizes = [2, 1], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x1xi1>
  // CHECK: arith.constant dense<false> : vector<2x1xi1>
  return %1 : vector<2x1xi1>
}

// -----

func.func @extract_strided_slice_of_constant_mask() -> (vector<2x1xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [0, 1], sizes = [2, 1], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x1xi1>
  // CHECK: arith.constant dense<true> : vector<2x1xi1>
  return %1 : vector<2x1xi1>
}

// -----

func.func @extract_strided_slice_of_constant_mask() -> (vector<2x1xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [1, 1], sizes = [2, 1], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x1xi1>
  // CHECK: vector.constant_mask [1, 1] : vector<2x1xi1>
  return %1 : vector<2x1xi1>
}

// -----

// CHECK-LABEL: func.func @extract_strided_slice_of_create_mask
// CHECK-SAME: (%[[DIM0:.+]]: index, %[[DIM1:.+]]: index)
func.func @extract_strided_slice_of_create_mask(%dim0: index, %dim1: index) -> (vector<2x2xi1>) {
  %0 = vector.create_mask %dim0, %dim1 : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [2, 1], sizes = [2, 2], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x2xi1>
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[A:.+]] = arith.subi %[[DIM0]], %[[C2]]
  // CHECK-DAG: %[[B:.+]] = arith.subi %[[DIM1]], %[[C1]]
  // CHECK: vector.create_mask %[[A]], %[[B]] : vector<2x2xi1>
  return %1 : vector<2x2xi1>
}

// -----

// CHECK-LABEL: func.func @extract_strided_slice_partial_of_create_mask
// CHECK-SAME: (%[[DIM0:.+]]: index, %[[DIM1:.+]]: index, %[[DIM2:.+]]: index)
func.func @extract_strided_slice_partial_of_create_mask(
  %dim0: index, %dim1: index, %dim2 : index) -> (vector<2x2x8xi1>) {
  %0 = vector.create_mask %dim0, %dim1, %dim2 : vector<4x3x8xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [2, 1], sizes = [2, 2], strides = [1, 1]}
      : vector<4x3x8xi1> to vector<2x2x8xi1>
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[A:.+]] = arith.subi %[[DIM0]], %[[C2]]
  // CHECK-DAG: %[[B:.+]] = arith.subi %[[DIM1]], %[[C1]]
  // CHECK: vector.create_mask %[[A]], %[[B]], %[[DIM2]] : vector<2x2x8xi1>
  return %1 : vector<2x2x8xi1>
}

// -----

// CHECK-LABEL: extract_strided_fold
//  CHECK-SAME: (%[[ARG:.*]]: vector<4x3xi1>)
//  CHECK-NEXT:   return %[[ARG]] : vector<4x3xi1>
func.func @extract_strided_fold(%arg : vector<4x3xi1>) -> (vector<4x3xi1>) {
  %0 = vector.extract_strided_slice %arg
    {offsets = [0, 0], sizes = [4, 3], strides = [1, 1]}
      : vector<4x3xi1> to vector<4x3xi1>
  return %0 : vector<4x3xi1>
}

// -----

// CHECK-LABEL: extract_strided_fold_insert
//  CHECK-SAME: (%[[ARG:.*]]: vector<4x4xf32>
//  CHECK-NEXT:   return %[[ARG]] : vector<4x4xf32>
func.func @extract_strided_fold_insert(%a: vector<4x4xf32>, %b: vector<8x16xf32>)
  -> (vector<4x4xf32>) {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1, 1]}
    : vector<4x4xf32> into vector<8x16xf32>
  %1 = vector.extract_strided_slice %0
    {offsets = [2, 2], sizes = [4, 4], strides = [1, 1]}
      : vector<8x16xf32> to vector<4x4xf32>
  return %1 : vector<4x4xf32>
}

// -----

// Case where the vector inserted is a subset of the vector extracted.
// CHECK-LABEL: extract_strided_fold_insert
//  CHECK-SAME: (%[[ARG0:.*]]: vector<6x4xf32>
//  CHECK-NEXT:   %[[EXT:.*]] = vector.extract_strided_slice %[[ARG0]]
//  CHECK-SAME:     {offsets = [0, 0], sizes = [4, 4], strides = [1, 1]}
//  CHECK-SAME:       : vector<6x4xf32> to vector<4x4xf32>
//  CHECK-NEXT:   return %[[EXT]] : vector<4x4xf32>
func.func @extract_strided_fold_insert(%a: vector<6x4xf32>, %b: vector<8x16xf32>)
  -> (vector<4x4xf32>) {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1, 1]}
    : vector<6x4xf32> into vector<8x16xf32>
  %1 = vector.extract_strided_slice %0
    {offsets = [2, 2], sizes = [4, 4], strides = [1, 1]}
      : vector<8x16xf32> to vector<4x4xf32>
  return %1 : vector<4x4xf32>
}

// -----

// Negative test where the extract is not a subset of the element inserted.
// CHECK-LABEL: negative_extract_strided_fold
//  CHECK-SAME: (%[[ARG0:.*]]: vector<4x4xf32>, %[[ARG1:.*]]: vector<8x16xf32>
//       CHECK:   %[[INS:.*]] = vector.insert_strided_slice %[[ARG0]], %[[ARG1]]
//  CHECK-SAME:     {offsets = [2, 2], strides = [1, 1]}
//  CHECK-SAME:       : vector<4x4xf32> into vector<8x16xf32>
//       CHECK:   %[[EXT:.*]] = vector.extract_strided_slice %[[INS]]
//  CHECK-SAME:     {offsets = [2, 2], sizes = [6, 4], strides = [1, 1]}
//  CHECK-SAME:       : vector<8x16xf32> to vector<6x4xf32>
//  CHECK-NEXT:   return %[[EXT]] : vector<6x4xf32>
func.func @negative_extract_strided_fold(%a: vector<4x4xf32>, %b: vector<8x16xf32>)
  -> (vector<6x4xf32>) {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1, 1]}
    : vector<4x4xf32> into vector<8x16xf32>
  %1 = vector.extract_strided_slice %0
    {offsets = [2, 2], sizes = [6, 4], strides = [1, 1]}
      : vector<8x16xf32> to vector<6x4xf32>
  return %1 : vector<6x4xf32>
}

// -----

// Case where we need to go through 2 level of insert element.
// CHECK-LABEL: extract_strided_fold_insert
//  CHECK-SAME: (%[[ARG0:.*]]: vector<2x8xf32>, %[[ARG1:.*]]: vector<1x4xf32>,
//  CHECK-NEXT:   %[[EXT:.*]] = vector.extract_strided_slice %[[ARG1]]
//  CHECK-SAME:     {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]}
//  CHECK-SAME:       : vector<1x4xf32> to vector<1x1xf32>
//  CHECK-NEXT:   return %[[EXT]] : vector<1x1xf32>
func.func @extract_strided_fold_insert(%a: vector<2x8xf32>, %b: vector<1x4xf32>,
                                  %c : vector<1x4xf32>) -> (vector<1x1xf32>) {
  %0 = vector.insert_strided_slice %b, %a {offsets = [0, 1], strides = [1, 1]}
    : vector<1x4xf32> into vector<2x8xf32>
  %1 = vector.insert_strided_slice %c, %0 {offsets = [1, 0], strides = [1, 1]}
    : vector<1x4xf32> into vector<2x8xf32>
  %2 = vector.extract_strided_slice %1
      {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]}
        : vector<2x8xf32> to vector<1x1xf32>
  return %2 : vector<1x1xf32>
}

// -----

// CHECK-LABEL: transpose_3D_identity
// CHECK-SAME: ([[ARG:%.*]]: vector<4x3x2xf32>)
func.func @transpose_3D_identity(%arg : vector<4x3x2xf32>) -> vector<4x3x2xf32> {
  // CHECK-NOT: transpose
  %0 = vector.transpose %arg, [0, 1, 2] : vector<4x3x2xf32> to vector<4x3x2xf32>
  // CHECK-NEXT: return [[ARG]]
  return %0 : vector<4x3x2xf32>
}

// -----

// CHECK-LABEL: transpose_2D_sequence
// CHECK-SAME: ([[ARG:%.*]]: vector<4x3xf32>)
func.func @transpose_2D_sequence(%arg : vector<4x3xf32>) -> vector<4x3xf32> {
  // CHECK-NOT: transpose
  %0 = vector.transpose %arg, [1, 0] : vector<4x3xf32> to vector<3x4xf32>
  %1 = vector.transpose %0, [0, 1] : vector<3x4xf32> to vector<3x4xf32>
  %2 = vector.transpose %1, [1, 0] : vector<3x4xf32> to vector<4x3xf32>
  %3 = vector.transpose %2, [0, 1] : vector<4x3xf32> to vector<4x3xf32>
  // CHECK: [[ADD:%.*]] = arith.addf [[ARG]], [[ARG]]
  %4 = arith.addf %2, %3 : vector<4x3xf32>
  // CHECK-NEXT: return [[ADD]]
  return %4 : vector<4x3xf32>
}

// -----

// CHECK-LABEL: transpose_3D_sequence
// CHECK-SAME: ([[ARG:%.*]]: vector<4x3x2xf32>)
func.func @transpose_3D_sequence(%arg : vector<4x3x2xf32>) -> vector<4x3x2xf32> {
  // CHECK: [[T0:%.*]] = vector.transpose [[ARG]], [2, 1, 0]
  %0 = vector.transpose %arg, [1, 2, 0] : vector<4x3x2xf32> to vector<3x2x4xf32>
  %1 = vector.transpose %0, [1, 0, 2] : vector<3x2x4xf32> to vector<2x3x4xf32>
  // CHECK: [[T1:%.*]] = vector.transpose %arg0, [2, 1, 0]
  %2 = vector.transpose %1, [2, 1, 0] : vector<2x3x4xf32> to vector<4x3x2xf32>
  %3 = vector.transpose %2, [2, 1, 0] : vector<4x3x2xf32> to vector<2x3x4xf32>
  // CHECK: [[MUL:%.*]] = arith.mulf [[T0]], [[T1]]
  %4 = arith.mulf %1, %3 : vector<2x3x4xf32>
  // CHECK: [[T5:%.*]] = vector.transpose [[MUL]], [2, 1, 0]
  %5 = vector.transpose %4, [2, 1, 0] : vector<2x3x4xf32> to vector<4x3x2xf32>
  // CHECK-NOT: transpose
  %6 = vector.transpose %3, [2, 1, 0] : vector<2x3x4xf32> to vector<4x3x2xf32>
  // CHECK: [[ADD:%.*]] = arith.addf [[T5]], [[ARG]]
  %7 = arith.addf %5, %6 : vector<4x3x2xf32>
  // CHECK-NEXT: return [[ADD]]
  return %7 : vector<4x3x2xf32>
}

// -----

// CHECK-LABEL: cast_transfers
func.func @cast_transfers(%A: memref<4x8xf32>) -> (vector<4x8xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %0 = memref.cast %A : memref<4x8xf32> to memref<?x?xf32>

  // CHECK: vector.transfer_read %{{.*}} {in_bounds = [true, true]} : memref<4x8xf32>, vector<4x8xf32>
  %1 = vector.transfer_read %0[%c0, %c0], %f0 : memref<?x?xf32>, vector<4x8xf32>

  // CHECK: vector.transfer_write %{{.*}} {in_bounds = [true, true]} : vector<4x8xf32>, memref<4x8xf32>
  vector.transfer_write %1, %0[%c0, %c0] : vector<4x8xf32>, memref<?x?xf32>
  return %1 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: cast_transfers
func.func @cast_transfers(%A: tensor<4x8xf32>) -> (vector<4x8xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %0 = tensor.cast %A : tensor<4x8xf32> to tensor<?x?xf32>

  // CHECK: vector.transfer_read %{{.*}} {in_bounds = [true, true]} : tensor<4x8xf32>, vector<4x8xf32>
  %1 = vector.transfer_read %0[%c0, %c0], %f0 : tensor<?x?xf32>, vector<4x8xf32>

  return %1 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func @insert_extract_transpose_2d(
//  CHECK-SAME: %[[V:[a-zA-Z0-9]*]]: vector<2x3xf32>,
//  CHECK-SAME: %[[F0:[a-zA-Z0-9]*]]: f32,
//  CHECK-SAME: %[[F1:[a-zA-Z0-9]*]]: f32,
//  CHECK-SAME: %[[F2:[a-zA-Z0-9]*]]: f32,
//  CHECK-SAME: %[[F3:[a-zA-Z0-9]*]]: f32
func.func @insert_extract_transpose_2d(
    %v: vector<2x3xf32>, %f0: f32, %f1: f32, %f2: f32, %f3: f32)
-> (f32, f32, f32)
{
  %0 = vector.insert %f0, %v[0, 0] : f32 into vector<2x3xf32>
  %1 = vector.insert %f1, %0[0, 1] : f32 into vector<2x3xf32>
  %2 = vector.insert %f2, %1[1, 0] : f32 into vector<2x3xf32>
  %3 = vector.insert %f3, %2[1, 1] : f32 into vector<2x3xf32>
  %4 = vector.transpose %3, [1, 0] : vector<2x3xf32> to vector<3x2xf32>
  %5 = vector.insert %f3, %4[1, 0] : f32 into vector<3x2xf32>
  %6 = vector.transpose %5, [1, 0] : vector<3x2xf32> to vector<2x3xf32>

  // Expected %f2 from %2 = vector.insert %f2, %1[1, 0].
  %r1 = vector.extract %3[1, 0] : f32 from vector<2x3xf32>

  // Expected %f1 from %1 = vector.insert %f1, %0[0, 1] followed by
  // transpose [1, 0].
  %r2 = vector.extract %4[1, 0] : f32 from vector<3x2xf32>

  // Expected %f2 from %2 = vector.insert %f2, %1[1, 0] followed by double
  // transpose [1, 0].
  %r3 = vector.extract %6[1, 0] : f32 from vector<2x3xf32>

  // CHECK-NEXT: return %[[F2]], %[[F1]], %[[F2]] : f32, f32, f32
  return %r1, %r2, %r3 : f32, f32, f32
}

// -----

// CHECK-LABEL: insert_extract_chain
//  CHECK-SAME: %[[V334:[a-zA-Z0-9]*]]: vector<3x3x4xf32>
//  CHECK-SAME: %[[V34:[a-zA-Z0-9]*]]: vector<3x4xf32>
//  CHECK-SAME: %[[V4:[a-zA-Z0-9]*]]: vector<4xf32>
func.func @insert_extract_chain(%v334: vector<3x3x4xf32>, %v34: vector<3x4xf32>, %v4: vector<4xf32>)
    -> (vector<4xf32>, vector<4xf32>, vector<3x4xf32>, vector<3x4xf32>) {
  // CHECK-NEXT: %[[A34:.*]] = vector.insert
  %A34 = vector.insert %v34, %v334[0]: vector<3x4xf32> into vector<3x3x4xf32>
  // CHECK-NEXT: %[[B34:.*]] = vector.insert
  %B34 = vector.insert %v34, %A34[1]: vector<3x4xf32> into vector<3x3x4xf32>
  // CHECK-NEXT: %[[A4:.*]] = vector.insert
  %A4 = vector.insert %v4, %B34[1, 0]: vector<4xf32> into vector<3x3x4xf32>
  // CHECK-NEXT: %[[B4:.*]] = vector.insert
  %B4 = vector.insert %v4, %A4[1, 1]: vector<4xf32> into vector<3x3x4xf32>

  // Case 2.a. [1, 1] == insertpos ([1, 1])
  // Match %A4 insertionpos and fold to its source(i.e. %V4).
   %r0 = vector.extract %B4[1, 1]: vector<4xf32> from vector<3x3x4xf32>

  // Case 3.a. insertpos ([1]) is a prefix of [1, 0].
  // Traverse %B34 to its source(i.e. %V34@[*0*]).
  // CHECK-NEXT: %[[R1:.*]] = vector.extract %[[V34]][0]
   %r1 = vector.extract %B34[1, 0]: vector<4xf32> from vector<3x3x4xf32>

  // Case 4. [1] is a prefix of insertpos ([1, 1]).
  // Cannot traverse %B4.
  // CHECK-NEXT: %[[R2:.*]] = vector.extract %[[B4]][1]
   %r2 = vector.extract %B4[1]: vector<3x4xf32> from vector<3x3x4xf32>

  // Case 5. [0] is disjoint from insertpos ([1, 1]).
  // Traverse %B4 to its dest(i.e. %A4@[0]).
  // Traverse %A4 to its dest(i.e. %B34@[0]).
  // Traverse %B34 to its dest(i.e. %A34@[0]).
  // Match %A34 insertionpos and fold to its source(i.e. %V34).
   %r3 = vector.extract %B4[0]: vector<3x4xf32> from vector<3x3x4xf32>

  // CHECK: return %[[V4]], %[[R1]], %[[R2]], %[[V34]]
  return %r0, %r1, %r2, %r3:
    vector<4xf32>, vector<4xf32>, vector<3x4xf32>, vector<3x4xf32>
}

// -----

// CHECK-LABEL: func @insert_extract_transpose_3d(
//  CHECK-SAME: %[[V234:[a-zA-Z0-9]*]]: vector<2x3x4xf32>
func.func @insert_extract_transpose_3d(
  %v234: vector<2x3x4xf32>, %v43: vector<4x3xf32>, %f0: f32)
    -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<3x4xf32>) {

  %a432 = vector.transpose %v234, [2, 1, 0] : vector<2x3x4xf32> to vector<4x3x2xf32>
  %b432 = vector.insert %f0, %a432[0, 0, 1] : f32 into vector<4x3x2xf32>
  %c234 = vector.transpose %b432, [2, 1, 0] : vector<4x3x2xf32> to vector<2x3x4xf32>
  // Case 1. %c234 = transpose [2,1,0] posWithSentinels [1,2,-1] -> [-1,2,1]
  // Case 5. %b432 = insert [0,0,1] (inter([.,2,1], [.,0,1]) == 0) prop to %v432
  // Case 1. %a432 = transpose [2,1,0] posWithSentinels [-1,2,1] -> [1,2,-1]
  // can extract directly from %v234, the rest folds.
  // CHECK: %[[R0:.*]] = vector.extract %[[V234]][1, 2]
  %r0 = vector.extract %c234[1, 2] : vector<4xf32> from vector<2x3x4xf32>

  // CHECK-NEXT: vector.transpose
  // CHECK-NEXT: vector.insert
  // CHECK-NEXT: %[[F234:.*]] = vector.transpose
  %d432 = vector.transpose %v234, [2, 1, 0] : vector<2x3x4xf32> to vector<4x3x2xf32>
  %e432 = vector.insert %f0, %d432[0, 2, 1] : f32 into vector<4x3x2xf32>
  %f234 = vector.transpose %e432, [2, 1, 0] : vector<4x3x2xf32> to vector<2x3x4xf32>
  // Case 1. %c234 = transpose [2,1,0] posWithSentinels [1,2,-1] -> [-1,2,1]
  // Case 4. %b432 = insert [0,0,1] (inter([.,2,1], [.,2,1]) != 0)
  // Bail, cannot do better than the current.
  // CHECK: %[[R1:.*]] = vector.extract %[[F234]]
  %r1 = vector.extract %f234[1, 2] : vector<4xf32> from vector<2x3x4xf32>

  // CHECK-NEXT: vector.transpose
  // CHECK-NEXT: vector.insert
  // CHECK-NEXT: %[[H234:.*]] = vector.transpose
  %g243 = vector.transpose %v234, [0, 2, 1] : vector<2x3x4xf32> to vector<2x4x3xf32>
  %h243 = vector.insert %v43, %g243[0] : vector<4x3xf32> into vector<2x4x3xf32>
  %i234 = vector.transpose %h243, [0, 2, 1] : vector<2x4x3xf32> to vector<2x3x4xf32>
  // Case 1. %i234 = transpose [0,2,1] posWithSentinels [0,-1,-2] -> [0,-2,-1]
  // Case 3.b. %b432 = insert [0] is prefix of [0,.,.] but internal transpose.
  // Bail, cannot do better than the current.
  // CHECK: %[[R2:.*]] = vector.extract %[[H234]][0, 1]
  %r2 = vector.extract %i234[0, 1] : vector<4xf32> from vector<2x3x4xf32>

  // CHECK-NEXT: vector.transpose
  // CHECK-NEXT: vector.insert
  // CHECK-NEXT: %[[K234:.*]] = vector.transpose
  %j243 = vector.transpose %v234, [0, 2, 1] : vector<2x3x4xf32> to vector<2x4x3xf32>
  %k243 = vector.insert %v43, %j243[0] : vector<4x3xf32> into vector<2x4x3xf32>
  %l234 = vector.transpose %k243, [0, 2, 1] : vector<2x4x3xf32> to vector<2x3x4xf32>
  // Case 1. %i234 = transpose [0,2,1] posWithSentinels [0,-1,-2] -> [0,-2,-1]
  // Case 2.b. %b432 = insert [0] == [0,.,.] but internal transpose.
  // Bail, cannot do better than the current.
  // CHECK: %[[R3:.*]] = vector.extract %[[K234]][0]
  %r3 = vector.extract %l234[0] : vector<3x4xf32> from vector<2x3x4xf32>

  // CHECK-NEXT: return %[[R0]], %[[R1]], %[[R2]], %[[R3]]
  return %r0, %r1, %r2, %r3: vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<3x4xf32>
}

// -----

// CHECK-LABEL: fold_extracts
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: vector<3x4x5x6xf32>
func.func @fold_extracts(%a : vector<3x4x5x6xf32>) -> (f32, vector<4x5x6xf32>) {
  %b = vector.extract %a[0] : vector<4x5x6xf32> from vector<3x4x5x6xf32>
  %c = vector.extract %b[1, 2] : vector<6xf32> from vector<4x5x6xf32>
  //  CHECK-NEXT: vector.extract %[[A]][0, 1, 2, 3] : f32 from vector<3x4x5x6xf32>
  %d = vector.extract %c[3] : f32 from vector<6xf32>

  //  CHECK-NEXT: vector.extract %[[A]][0] : vector<4x5x6xf32> from vector<3x4x5x6xf32>
  %e = vector.extract %a[0] : vector<4x5x6xf32> from vector<3x4x5x6xf32>

  //  CHECK-NEXT: return
  return %d, %e : f32, vector<4x5x6xf32>
}

// -----

// CHECK-LABEL: fold_extract_transpose
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: vector<3x4x5x6xf32>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]*]]: vector<3x6x5x6xf32>
func.func @fold_extract_transpose(
    %a : vector<3x4x5x6xf32>, %b : vector<3x6x5x6xf32>) -> (
      vector<6xf32>, vector<6xf32>, vector<6xf32>) {
  // [3] is a proper most minor identity map in transpose.
  // Permutation is a self inverse and we have.
  // [0, 2, 1] ^ -1 o [0, 1, 2] = [0, 2, 1] o [0, 1, 2]
  //                            = [0, 2, 1]
  //  CHECK-NEXT: vector.extract %[[A]][0, 2, 1] : vector<6xf32> from vector<3x4x5x6xf32>
  %0 = vector.transpose %a, [0, 2, 1, 3] : vector<3x4x5x6xf32> to vector<3x5x4x6xf32>
  %1 = vector.extract %0[0, 1, 2] : vector<6xf32> from vector<3x5x4x6xf32>

  // [3] is a proper most minor identity map in transpose.
  // Permutation is a not self inverse and we have.
  // [1, 2, 0] ^ -1 o [0, 1, 2] = [2, 0, 1] o [0, 1, 2]
  //                            = [2, 0, 1]
  //  CHECK-NEXT: vector.extract %[[A]][2, 0, 1] : vector<6xf32> from vector<3x4x5x6xf32>
  %2 = vector.transpose %a, [1, 2, 0, 3] : vector<3x4x5x6xf32> to vector<4x5x3x6xf32>
  %3 = vector.extract %2[0, 1, 2] : vector<6xf32> from vector<4x5x3x6xf32>

  // Not a minor identity map so intra-vector level has been permuted
  //  CHECK-NEXT: vector.transpose %[[B]], [0, 2, 3, 1]
  //  CHECK-NEXT: vector.extract %{{.*}}[0, 1, 2]
  %4 = vector.transpose %b, [0, 2, 3, 1] : vector<3x6x5x6xf32> to vector<3x5x6x6xf32>
  %5 = vector.extract %4[0, 1, 2] : vector<6xf32> from vector<3x5x6x6xf32>

  return %1, %3, %5 : vector<6xf32>, vector<6xf32>, vector<6xf32>
}

// -----

// CHECK-LABEL: fold_extract_broadcast_same_input_output_scalar
//  CHECK-SAME:   %[[A:.*]]: f32
//       CHECK:   return %[[A]] : f32
func.func @fold_extract_broadcast_same_input_output_scalar(%a : f32,
  %idx0 : index, %idx1 : index, %idx2 : index) -> f32 {
  %b = vector.broadcast %a : f32 to vector<1x2x4xf32>
  %r = vector.extract %b[%idx0, %idx1, %idx2] : f32 from vector<1x2x4xf32>
  return %r : f32
}

// -----

// CHECK-LABEL: fold_extract_broadcast_same_input_output_vec
//  CHECK-SAME:   %[[A:.*]]: vector<4xf32>
//       CHECK:   return %[[A]] : vector<4xf32>
func.func @fold_extract_broadcast_same_input_output_vec(%a : vector<4xf32>,
  %idx0 : index, %idx1 : index) -> vector<4xf32> {
  %b = vector.broadcast %a : vector<4xf32> to vector<1x2x4xf32>
  %r = vector.extract %b[%idx0, %idx1] : vector<4xf32> from vector<1x2x4xf32>
  return %r : vector<4xf32>
}

// -----

// CHECK-LABEL: fold_extract_broadcast_0dvec_input_scalar_output
//  CHECK-SAME:   %[[A:.*]]: vector<f32>
//       CHECK:   %[[B:.+]] = vector.extract %[[A]][] : f32 from vector<f32>
//       CHECK:   return %[[B]] : f32
func.func @fold_extract_broadcast_0dvec_input_scalar_output(%a : vector<f32>,
  %idx0 : index, %idx1 : index, %idx2: index) -> f32 {
  %b = vector.broadcast %a : vector<f32> to vector<1x2x4xf32>
  %r = vector.extract %b[%idx0, %idx1, %idx2] : f32 from vector<1x2x4xf32>
  return %r : f32
}

// -----

// CHECK-LABEL: negative_fold_extract_broadcast
//       CHECK:   vector.broadcast %{{.*}} : vector<1x1xf32> to vector<1x1x4xf32>
//       CHECK:   vector.extract %{{.*}}[0, 0] : vector<4xf32> from vector<1x1x4xf32>
func.func @negative_fold_extract_broadcast(%a : vector<1x1xf32>) -> vector<4xf32> {
  %b = vector.broadcast %a : vector<1x1xf32> to vector<1x1x4xf32>
  %r = vector.extract %b[0, 0] : vector<4xf32> from vector<1x1x4xf32>
  return %r : vector<4xf32>
}

// -----

// CHECK-LABEL: fold_extract_splatlike
//  CHECK-SAME:   %[[A:.*]]: f32
//       CHECK:   return %[[A]] : f32
func.func @fold_extract_splatlike(%a : f32, %idx0 : index, %idx1 : index, %idx2 : index) -> f32 {
  %b = vector.broadcast %a : f32 to vector<1x2x4xf32>
  %r = vector.extract %b[%idx0, %idx1, %idx2] : f32 from vector<1x2x4xf32>
  return %r : f32
}

// -----

// CHECK-LABEL: fold_extract_vector_from_splat
//       CHECK: vector.broadcast {{.*}} f32 to vector<4xf32>
func.func @fold_extract_vector_from_splat(%a : f32, %idx0 : index, %idx1 : index) -> vector<4xf32> {
  %b = vector.broadcast %a : f32 to vector<1x2x4xf32>
  %r = vector.extract %b[%idx0, %idx1] : vector<4xf32> from vector<1x2x4xf32>
  return %r : vector<4xf32>
}

// -----

// CHECK-LABEL: fold_extract_broadcast_dim1_broadcasting
//  CHECK-SAME:   %[[A:.*]]: vector<2x1xf32>
//  CHECK-SAME:   %[[IDX:.*]]: index, %[[IDX1:.*]]: index, %[[IDX2:.*]]: index
//       CHECK:   %[[R:.*]] = vector.extract %[[A]][%[[IDX1]], 0] : f32 from vector<2x1xf32>
//       CHECK:   return %[[R]] : f32
func.func @fold_extract_broadcast_dim1_broadcasting(%a : vector<2x1xf32>,
  %idx : index, %idx1 : index, %idx2 : index) -> f32 {
  %b = vector.broadcast %a : vector<2x1xf32> to vector<1x2x4xf32>
  %r = vector.extract %b[%idx, %idx1, %idx2] : f32 from vector<1x2x4xf32>
  return %r : f32
}

// -----

// CHECK-LABEL: fold_extract_broadcast_to_lower_rank
//  CHECK-SAME:   %[[A:.*]]: vector<2x4xf32>
//  CHECK-SAME:   %[[IDX0:.*]]: index, %[[IDX1:.*]]: index
//       CHECK:   %[[B:.+]] = vector.extract %[[A]][%[[IDX1]]] : vector<4xf32> from vector<2x4xf32>
//       CHECK:   return %[[B]] : vector<4xf32>
// rank(extract_output) < rank(broadcast_input)
func.func @fold_extract_broadcast_to_lower_rank(%a : vector<2x4xf32>,
  %idx0 : index, %idx1 : index) -> vector<4xf32> {
  %b = vector.broadcast %a : vector<2x4xf32> to vector<1x2x4xf32>
  %r = vector.extract %b[%idx0, %idx1] : vector<4xf32> from vector<1x2x4xf32>
  return %r : vector<4xf32>
}

// -----

// Test where the shape_cast is broadcast-like.
// CHECK-LABEL: fold_extract_shape_cast_to_lower_rank
//  CHECK-SAME:   %[[A:.*]]: vector<2x4xf32>
//  CHECK-SAME:   %[[IDX0:.*]]: index, %[[IDX1:.*]]: index
//       CHECK:   %[[B:.+]] = vector.extract %[[A]][%[[IDX1]]] : vector<4xf32> from vector<2x4xf32>
//       CHECK:   return %[[B]] : vector<4xf32>
func.func @fold_extract_shape_cast_to_lower_rank(%a : vector<2x4xf32>,
  %idx0 : index, %idx1 : index) -> vector<4xf32> {
  %b = vector.shape_cast %a : vector<2x4xf32> to vector<1x2x4xf32>
  %r = vector.extract %b[%idx0, %idx1] : vector<4xf32> from vector<1x2x4xf32>
  return %r : vector<4xf32>
}

// -----

// Test where the shape_cast is not broadcast-like, even though it prepends 1s.
// CHECK-LABEL: negative_fold_extract_shape_cast_to_lower_rank
//  CHECK-NEXT: vector.shape_cast
//  CHECK-NEXT: vector.extract
//  CHECK-NEXT: return
func.func @negative_fold_extract_shape_cast_to_lower_rank(%a : vector<2x4xf32>,
  %idx0 : index, %idx1 : index) -> vector<2xf32> {
  %b = vector.shape_cast %a : vector<2x4xf32> to vector<1x4x2xf32>
  %r = vector.extract %b[%idx0, %idx1] : vector<2xf32> from vector<1x4x2xf32>
  return %r : vector<2xf32>
}

// -----

// CHECK-LABEL: fold_extract_broadcast_to_higher_rank
//       CHECK:   %[[B:.*]] = vector.broadcast %{{.*}} : f32 to vector<4xf32>
//       CHECK:   return %[[B]] : vector<4xf32>
// rank(extract_output) > rank(broadcast_input)
func.func @fold_extract_broadcast_to_higher_rank(%a : f32, %idx0 : index, %idx1 : index)
  -> vector<4xf32> {
  %b = vector.broadcast %a : f32 to vector<1x2x4xf32>
  %r = vector.extract %b[%idx0, %idx1] : vector<4xf32> from vector<1x2x4xf32>
  return %r : vector<4xf32>
}

// -----

// CHECK-LABEL: fold_extract_broadcast_to_equal_rank
//  CHECK-SAME:   %[[A:.*]]: vector<1xf32>
//       CHECK:   %[[R:.*]] = vector.broadcast %[[A]] : vector<1xf32> to vector<8xf32>
//       CHECK:   return %[[R]] : vector<8xf32>
// rank(extract_output) == rank(broadcast_input)
func.func @fold_extract_broadcast_to_equal_rank(%a : vector<1xf32>, %idx0 : index)
  -> vector<8xf32> {
  %b = vector.broadcast %a : vector<1xf32> to vector<1x8xf32>
  %r = vector.extract %b[%idx0] : vector<8xf32> from vector<1x8xf32>
  return %r : vector<8xf32>
}

// -----

// CHECK-LABEL: fold_extract_broadcastlike_shape_cast
//  CHECK-SAME:   %[[A:.*]]: vector<1xf32>
//       CHECK:   %[[R:.*]] = vector.broadcast %[[A]] : vector<1xf32> to vector<1x1xf32>
//       CHECK:   return %[[R]] : vector<1x1xf32>
func.func @fold_extract_broadcastlike_shape_cast(%a : vector<1xf32>, %idx0 : index)
  -> vector<1x1xf32> {
  %s = vector.shape_cast %a : vector<1xf32> to vector<1x1x1xf32>
  %r = vector.extract %s[%idx0] : vector<1x1xf32> from vector<1x1x1xf32>
  return %r : vector<1x1xf32>
}

// -----

// CHECK-LABEL: @fold_extract_shuffle
//  CHECK-SAME:   %[[A:.*]]: vector<8xf32>, %[[B:.*]]: vector<8xf32>
//   CHECK-NOT:   vector.shuffle
//       CHECK:   vector.extract %[[A]][0] : f32 from vector<8xf32>
//       CHECK:   vector.extract %[[B]][0] : f32 from vector<8xf32>
//       CHECK:   vector.extract %[[A]][7] : f32 from vector<8xf32>
//       CHECK:   vector.extract %[[B]][7] : f32 from vector<8xf32>
func.func @fold_extract_shuffle(%a : vector<8xf32>, %b : vector<8xf32>)
                                -> (f32, f32, f32, f32) {
  %shuffle = vector.shuffle %a, %b [0, 8, 7, 15] : vector<8xf32>, vector<8xf32>
  %e0 = vector.extract %shuffle[0] : f32 from vector<4xf32>
  %e1 = vector.extract %shuffle[1] : f32 from vector<4xf32>
  %e2 = vector.extract %shuffle[2] : f32 from vector<4xf32>
  %e3 = vector.extract %shuffle[3] : f32 from vector<4xf32>
  return %e0, %e1, %e2, %e3 : f32, f32, f32, f32
}

// -----

// CHECK-LABEL: func @fold_extract_shapecast
//  CHECK-SAME: (%[[A0:.*]]: vector<5x1x3x2xf32>, %[[A1:.*]]: vector<8x4x2xf32>
//       CHECK:   %[[R0:.*]] = vector.extract %[[A0]][1, 0, 1, 1] : f32 from vector<5x1x3x2xf32>
//       CHECK:   %[[R1:.*]] = vector.extract %[[A0]][1, 0, 2] : vector<2xf32> from vector<5x1x3x2xf32>
//       CHECK:   %[[R2:.*]] = vector.extract %[[A1]][7] : vector<4x2xf32> from vector<8x4x2xf32>
//       CHECK:   return %[[R0]], %[[R1]], %[[R2]], %[[A1]] : f32, vector<2xf32>, vector<4x2xf32>, vector<8x4x2xf32>
func.func @fold_extract_shapecast(%arg0 : vector<5x1x3x2xf32>,
                             %arg1 : vector<8x4x2xf32>)
  -> (f32, vector<2xf32>, vector<4x2xf32>, vector<8x4x2xf32>) {
  %0 = vector.shape_cast %arg0 : vector<5x1x3x2xf32> to vector<15x2xf32>
  %1 = vector.shape_cast %arg1 : vector<8x4x2xf32> to vector<4x2x4x2xf32>
  %2 = vector.shape_cast %arg1 : vector<8x4x2xf32> to vector<1x8x4x2xf32>
  %r1 = vector.extract %0[4, 1] : f32 from vector<15x2xf32>
  %r2 = vector.extract %0[5] : vector<2xf32> from vector<15x2xf32>
  %r3 = vector.extract %1[3, 1] : vector<4x2xf32> from vector<4x2x4x2xf32>
  %r4 = vector.extract %2[0] : vector<8x4x2xf32> from vector<1x8x4x2xf32>
  return %r1, %r2, %r3, %r4 : f32, vector<2xf32>, vector<4x2xf32>, vector<8x4x2xf32>
}

// -----

// CHECK-LABEL: fold_extract_shapecast_0d_result
//  CHECK-SAME: %[[IN:.*]]: vector<1x1x1xf32>
//       CHECK:   %[[R:.*]] = vector.extract %[[IN]][0, 0, 0] : f32 from vector<1x1x1xf32>
//       CHECK:   return %[[R]] : f32
func.func @fold_extract_shapecast_0d_result(%arg0 : vector<1x1x1xf32>) -> f32 {
  %0 = vector.shape_cast %arg0 : vector<1x1x1xf32> to vector<f32>
  %r = vector.extract %0[] : f32 from vector<f32>
  return %r : f32
}

// -----

// CHECK-LABEL: fold_extract_shapecast_0d_source
//  CHECK-SAME: %[[IN:.*]]: vector<f32>
//       CHECK:   %[[R:.*]] = vector.extract %[[IN]][] : f32 from vector<f32>
//       CHECK:   return %[[R]] : f32
func.func @fold_extract_shapecast_0d_source(%arg0 : vector<f32>) -> f32 {
  %0 = vector.shape_cast %arg0 : vector<f32> to vector<1xf32>
  %r = vector.extract %0[0] : f32 from vector<1xf32>
  return %r : f32
}

// -----

// CHECK-LABEL: negative_fold_extract_shapecast
//       CHECK:   %[[V:.*]] = vector.shape_cast %{{.*}} : vector<16xf32> to vector<2x4x2xf32>
//       CHECK:   %[[R:.*]] = vector.extract %[[V]][1] : vector<4x2xf32> from vector<2x4x2xf32>
//       CHECK:   return %[[R]] : vector<4x2xf32>
func.func @negative_fold_extract_shapecast(%arg0 : vector<16xf32>) -> vector<4x2xf32> {
  %0 = vector.shape_cast %arg0 : vector<16xf32> to vector<2x4x2xf32>
  %r = vector.extract %0[1] : vector<4x2xf32> from vector<2x4x2xf32>
  return %r : vector<4x2xf32>
}

// -----

// CHECK-LABEL: fold_extract_shapecast_to_shapecast
//  CHECK-SAME: (%[[ARG:.+]]: vector<3x4xf32>)
//       CHECK:   %[[R:.+]] = vector.shape_cast %[[ARG]] : vector<3x4xf32> to vector<12xf32>
//       CHECK:   return %[[R]]
func.func @fold_extract_shapecast_to_shapecast(%arg0 : vector<3x4xf32>) -> vector<12xf32> {
  %0 = vector.shape_cast %arg0 : vector<3x4xf32> to vector<1x12xf32>
  %r = vector.extract %0[0] : vector<12xf32> from vector<1x12xf32>
  return %r : vector<12xf32>
}

// -----

// CHECK-LABEL: func @extract_no_fold_scalar_to_0d(
//  CHECK-SAME:     %[[v:.*]]: vector<f32>)
//       CHECK:   %[[extract:.*]] = vector.extract %[[v]][] : f32 from vector<f32>
//       CHECK:   return %[[extract]]
func.func @extract_no_fold_scalar_to_0d(%v: vector<f32>) -> f32 {
  %0 = vector.extract %v[] : f32 from vector<f32>
  return %0 : f32
}

// -----

// CHECK-LABEL: func @insert_fold_same_rank(
//  CHECK-SAME:     %[[v:.*]]: vector<2x2xf32>)
//       CHECK:      %[[CST:.+]] = arith.constant
//  CHECK-SAME:                    : vector<2x2xf32>
//       CHECK-NOT:  vector.insert
//       CHECK:   return %[[CST]]
func.func @insert_fold_same_rank(%v: vector<2x2xf32>) -> vector<2x2xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<2x2xf32>
  %0 = vector.insert %cst, %v [] : vector<2x2xf32> into vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

// -----

// CHECK-LABEL: func @insert_no_fold_scalar_to_0d(
//  CHECK-SAME:     %[[v:.*]]: vector<f32>)
//       CHECK:   %[[cst:.*]] = arith.constant dense<0.000000e+00> : vector<f32>
//       CHECK:   return %[[cst]]
func.func @insert_no_fold_scalar_to_0d(%v: vector<f32>) -> vector<f32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.insert %cst, %v [] : f32 into vector<f32>
  return %0 : vector<f32>
}

// -----

// CHECK-LABEL: fold_expand_collapse
//       CHECK:   %[[A:.*]] = vector.shape_cast %{{.*}} : vector<1x1x64xf32> to vector<8x8xf32>
//       CHECK:   return %[[A]] : vector<8x8xf32>
func.func @dont_fold_expand_collapse(%arg0: vector<1x1x64xf32>) -> vector<8x8xf32> {
    %0 = vector.shape_cast %arg0 : vector<1x1x64xf32> to vector<1x1x8x8xf32>
    %1 = vector.shape_cast %0 : vector<1x1x8x8xf32> to vector<8x8xf32>
    return %1 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: func @fold_broadcast_shapecast
//  CHECK-SAME: (%[[V:.+]]: vector<4xf32>)
//       CHECK:   return %[[V]]
func.func @fold_broadcast_shapecast(%arg0: vector<4xf32>) -> vector<4xf32> {
    %0 = vector.broadcast %arg0 : vector<4xf32> to vector<1x1x4xf32>
    %1 = vector.shape_cast %0 : vector<1x1x4xf32> to vector<4xf32>
    return %1 : vector<4xf32>
}

// -----

// CHECK-LABEL: func @canonicalize_broadcast_shapecast_scalar
//       CHECK:   vector.broadcast
//   CHECK-NOT:   vector.shape_cast
func.func @canonicalize_broadcast_shapecast_scalar(%arg0: f32) -> vector<1xf32> {
    %0 = vector.broadcast %arg0 : f32 to vector<1x1x1xf32>
    %1 = vector.shape_cast %0 : vector<1x1x1xf32> to vector<1xf32>
    return %1 : vector<1xf32>
}

// -----

// CHECK-LABEL: func @dont_fold_broadcast_shapecast_diff_shape
//       CHECK:   vector.broadcast
//       CHECK:   vector.shape_cast
func.func @dont_fold_broadcast_shapecast_diff_shape(%arg0: vector<4xf32>) -> vector<8xf32> {
    %0 = vector.broadcast %arg0 : vector<4xf32> to vector<1x2x4xf32>
    %1 = vector.shape_cast %0 : vector<1x2x4xf32> to vector<8xf32>
    return %1 : vector<8xf32>
}

// -----

// CHECK-LABEL: func @canonicalize_broadcast_shapecast_to_broadcast
//       CHECK:   vector.broadcast
//   CHECK-NOT:   vector.shape_cast
func.func @canonicalize_broadcast_shapecast_to_broadcast(%arg0: vector<3xf32>) -> vector<8x3xf32> {
    %0 = vector.broadcast %arg0 : vector<3xf32> to vector<2x4x3xf32>
    %1 = vector.shape_cast %0 : vector<2x4x3xf32> to vector<8x3xf32>
    return %1 : vector<8x3xf32>
}

// -----

// CHECK-LABEL: func @canonicalize_broadcast_shapecast_to_broadcast_ones
//       CHECK:   vector.broadcast {{.*}} vector<1x1xi8> to vector<1x1x6x1x4xi8>
//   CHECK-NOT:   vector.shape_cast
func.func @canonicalize_broadcast_shapecast_to_broadcast_ones(%arg0: vector<1x1xi8>) -> vector<1x1x6x1x4xi8> {
  %0 = vector.broadcast %arg0 : vector<1x1xi8> to vector<6x4xi8>
  %1 = vector.shape_cast %0 : vector<6x4xi8> to vector<1x1x6x1x4xi8>
  return %1 : vector<1x1x6x1x4xi8>
}

// -----

// CHECK-LABEL: func @canonicalize_broadcast_shapecast_to_broadcast_scalar
//       CHECK:   vector.broadcast {{.*}} f32 to vector<3x4x1xf32>
//   CHECK-NOT:   vector.shape_cast
func.func @canonicalize_broadcast_shapecast_to_broadcast_scalar(%arg0: f32) -> vector<3x4x1xf32> {
  %0 = vector.broadcast %arg0 : f32 to vector<12xf32>
  %1 = vector.shape_cast %0 : vector<12xf32> to vector<3x4x1xf32>
  return %1 : vector<3x4x1xf32>
}

// -----

// In this test, broadcast (2)->(1,2,1) is not legal, but shape_cast (2)->(1,2,1) is.
// CHECK-LABEL: func @canonicalize_broadcast_shapecast_to_shapcast
//   CHECK-NOT:   vector.broadcast
//       CHECK:   vector.shape_cast {{.+}} : vector<2xf32> to vector<1x2x1xf32>
func.func @canonicalize_broadcast_shapecast_to_shapcast(%arg0 : vector<2xf32>) -> vector<1x2x1xf32> {
  %0 = vector.broadcast %arg0 : vector<2xf32> to vector<1x2xf32>
  %1 = vector.shape_cast %0 : vector<1x2xf32> to vector<1x2x1xf32>
  return %1 : vector<1x2x1xf32>
}

// -----

// In this test, broadcast (1)->(1,1) and shape_cast (1)->(1,1) are both legal. shape_cast is chosen.
// CHECK-LABEL: func @canonicalize_broadcast_shapecast_both_possible
//   CHECK-NOT:   vector.broadcast
//       CHECK:   vector.shape_cast {{.+}} : vector<1xf32> to vector<1x1xf32>
func.func @canonicalize_broadcast_shapecast_both_possible(%arg0: vector<1xf32>) -> vector<1x1xf32> {
    %0 = vector.broadcast %arg0 : vector<1xf32> to vector<1x1x1xf32>
    %1 = vector.shape_cast %0 : vector<1x1x1xf32> to vector<1x1xf32>
    return %1 : vector<1x1xf32>
}

// -----

// CHECK-LABEL: func @canonicalize_shapecast_broadcast_to_broadcast_prepend_dim
//   CHECK-NOT:   vector.shape_cast
//       CHECK:   vector.broadcast {{.+}} : vector<2xf32> to vector<32x2xf32>
func.func @canonicalize_shapecast_broadcast_to_broadcast_prepend_dim(%arg0 : vector<2xf32>) -> vector<32x2xf32> {
  %0 = vector.shape_cast %arg0 : vector<2xf32> to vector<1x2xf32>
  %1 = vector.broadcast %0 : vector<1x2xf32> to vector<32x2xf32>
  return %1 : vector<32x2xf32>
}

// -----

// CHECK-LABEL:   func.func @canonicalize_shapecast_broadcast_to_broadcast_prepend_dim2(
// CHECK-SAME:      %[[ARG0:.*]]: vector<2x1xf32>) -> vector<32x2x1xf32> {
// CHECK:           %[[VAL_0:.*]] = vector.broadcast %[[ARG0]] : vector<2x1xf32> to vector<32x2x1xf32>
// CHECK:           return %[[VAL_0]] : vector<32x2x1xf32>
// CHECK:         }
func.func @canonicalize_shapecast_broadcast_to_broadcast_prepend_dim2(%arg0 : vector<2x1xf32>) -> vector<32x2x1xf32> {
  %0 = vector.shape_cast %arg0 : vector<2x1xf32> to vector<1x2x1xf32>
  %1 = vector.broadcast %0 : vector<1x2x1xf32> to vector<32x2x1xf32>
  return %1 : vector<32x2x1xf32>
}

// -----

// CHECK-LABEL:   func.func @canonicalize_shapecast_broadcast_to_broadcast_prepend_dim3(
// CHECK-SAME:      %[[ARG0:.*]]: vector<2x1xf32>) -> vector<32x2x4xf32> {
// CHECK:           %[[VAL_0:.*]] = vector.broadcast %[[ARG0]] : vector<2x1xf32> to vector<32x2x4xf32>
// CHECK:           return %[[VAL_0]] : vector<32x2x4xf32>
// CHECK:         }
func.func @canonicalize_shapecast_broadcast_to_broadcast_prepend_dim3(%arg0 : vector<2x1xf32>) -> vector<32x2x4xf32> {
  %0 = vector.shape_cast %arg0 : vector<2x1xf32> to vector<1x2x1xf32>
  %1 = vector.broadcast %0 : vector<1x2x1xf32> to vector<32x2x4xf32>
  return %1 : vector<32x2x4xf32>
}

// -----

// CHECK-LABEL:   func.func @canonicalize_shapecast_broadcast_to_broadcast_remove_leading_dim(
// CHECK-SAME:      %[[ARG0:.*]]: vector<1x2xf32>) -> vector<32x2xf32> {
// CHECK:           %[[VAL_0:.*]] = vector.broadcast %[[ARG0]] : vector<1x2xf32> to vector<32x2xf32>
// CHECK:           return %[[VAL_0]] : vector<32x2xf32>
// CHECK:         }
func.func @canonicalize_shapecast_broadcast_to_broadcast_remove_leading_dim(%arg0 : vector<1x2xf32>) -> vector<32x2xf32> {
  %0 = vector.shape_cast %arg0 : vector<1x2xf32> to vector<2xf32>
  %1 = vector.broadcast %0 : vector<2xf32> to vector<32x2xf32>
  return %1 : vector<32x2xf32>
}

// -----

// CHECK-LABEL: func @negative_canonicalize_shapecast_broadcast_invalid_shape
//       CHECK:   vector.shape_cast {{.+}} : vector<64xf32> to vector<4x16xf32>
//       CHECK:   vector.broadcast {{.+}} : vector<4x16xf32> to vector<2x4x16xf32>
func.func @negative_canonicalize_shapecast_broadcast_invalid_shape(%arg0 : vector<64xf32>) -> vector<2x4x16xf32> {
  %0 = vector.shape_cast %arg0 : vector<64xf32> to vector<4x16xf32>
  %1 = vector.broadcast %0 : vector<4x16xf32> to vector<2x4x16xf32>
  return %1 : vector<2x4x16xf32>
}

// -----

// CHECK-LABEL: func @negative_canonicalize_shapecast_broadcast_invalid_broadcasted_dims
//       CHECK:   vector.shape_cast {{.+}} : vector<2x1xf32> to vector<1x2xf32>
//       CHECK:   vector.broadcast {{.+}} : vector<1x2xf32> to vector<2x2xf32>
func.func @negative_canonicalize_shapecast_broadcast_invalid_broadcasted_dims(%arg0 : vector<2x1xf32>) -> vector<2x2xf32> {
  %0 = vector.shape_cast %arg0 : vector<2x1xf32> to vector<1x2xf32>
  %1 = vector.broadcast %0 : vector<1x2xf32> to vector<2x2xf32>
  return %1 : vector<2x2xf32>
}

// -----

// CHECK-LABEL:   func.func @negative_canonicalize_shapecast_broadcast_to_broadcast_append_dim(
// CHECK-SAME:      %[[ARG0:.*]]: vector<2xf32>) -> vector<2x4xf32> {
// CHECK:           %[[VAL_0:.*]] = vector.shape_cast %[[ARG0]] : vector<2xf32> to vector<2x1xf32>
// CHECK:           %[[VAL_1:.*]] = vector.broadcast %[[VAL_0]] : vector<2x1xf32> to vector<2x4xf32>
// CHECK:           return %[[VAL_1]] : vector<2x4xf32>
// CHECK:         }
func.func @negative_canonicalize_shapecast_broadcast_to_broadcast_append_dim(%arg0 : vector<2xf32>) -> vector<2x4xf32> {
  %0 = vector.shape_cast %arg0 : vector<2xf32> to vector<2x1xf32>
  %1 = vector.broadcast %0 : vector<2x1xf32> to vector<2x4xf32>
  return %1 : vector<2x4xf32>
}

// -----

// CHECK-LABEL:   func.func @negative_canonicalize_shapecast_broadcast_to_broadcast_remove_trailing_dim(
// CHECK-SAME:      %[[ARG0:.*]]: vector<2x1xf32>) -> vector<32x2xf32> {
// CHECK:           %[[VAL_0:.*]] = vector.shape_cast %[[ARG0]] : vector<2x1xf32> to vector<2xf32>
// CHECK:           %[[VAL_1:.*]] = vector.broadcast %[[VAL_0]] : vector<2xf32> to vector<32x2xf32>
// CHECK:           return %[[VAL_1]] : vector<32x2xf32>
// CHECK:         }
func.func @negative_canonicalize_shapecast_broadcast_to_broadcast_remove_trailing_dim(%arg0 : vector<2x1xf32>) -> vector<32x2xf32> {
  %0 = vector.shape_cast %arg0 : vector<2x1xf32> to vector<2xf32>
  %1 = vector.broadcast %0 : vector<2xf32> to vector<32x2xf32>
  return %1 : vector<32x2xf32>
}

// -----

// CHECK-LABEL: fold_vector_transfer_masks
func.func @fold_vector_transfer_masks(%A: memref<?x?xf32>) -> (vector<4x8xf32>, vector<4x[4]xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK: %[[F0:.+]] = arith.constant 0.000000e+00 : f32
  %f0 = arith.constant 0.0 : f32

  %mask = vector.constant_mask [8, 4] : vector<8x4xi1>

  %arith_all_true_mask = arith.constant dense<true> : vector<4x[4]xi1>

  // CHECK: vector.transfer_read %{{.*}}, %[[F0]] {permutation_map
  %1 = vector.transfer_read %A[%c0, %c0], %f0, %mask
      {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<?x?xf32>, vector<4x8xf32>

  // CHECK: vector.transfer_write {{.*}}[%[[C0]], %[[C0]]] {permutation_map
  vector.transfer_write %1, %A[%c0, %c0], %mask
      {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : vector<4x8xf32>, memref<?x?xf32>

  // CHECK: vector.transfer_read %{{.*}}, %[[F0]] :
  %2 = vector.transfer_read %A[%c0, %c0], %f0, %arith_all_true_mask : memref<?x?xf32>, vector<4x[4]xf32>

  // CHECK: vector.transfer_write {{.*}}[%[[C0]], %[[C0]]] :
  vector.transfer_write %2, %A[%c0, %c0], %arith_all_true_mask : vector<4x[4]xf32>, memref<?x?xf32>

  // CHECK: return
  return %1, %2 : vector<4x8xf32>, vector<4x[4]xf32>
}

// -----

// CHECK-LABEL: fold_vector_transfers
func.func @fold_vector_transfers(%A: memref<?x8xf32>) -> (vector<4x8xf32>, vector<4x9xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

  // CHECK: vector.transfer_read %{{.*}} {in_bounds = [false, true]}
  %1 = vector.transfer_read %A[%c0, %c0], %f0 : memref<?x8xf32>, vector<4x8xf32>

  // CHECK: vector.transfer_write %{{.*}} {in_bounds = [false, true]}
  vector.transfer_write %1, %A[%c0, %c0] : vector<4x8xf32>, memref<?x8xf32>

  // Both dims may be out-of-bounds, attribute is elided.
  // CHECK: vector.transfer_read %{{.*}}
  // CHECK-NOT: in_bounds
  %2 = vector.transfer_read %A[%c0, %c0], %f0 : memref<?x8xf32>, vector<4x9xf32>

  // Both dims may be out-of-bounds, attribute is elided.
  // CHECK: vector.transfer_write %{{.*}}
  // CHECK-NOT: in_bounds
  vector.transfer_write %2, %A[%c0, %c0] : vector<4x9xf32>, memref<?x8xf32>

  // CHECK: return
  return %1, %2 : vector<4x8xf32>, vector<4x9xf32>
}

// -----

// CHECK-LABEL: bitcast_folding
//  CHECK-SAME:   %[[A:.*]]: vector<4x8xf32>
//  CHECK-SAME:   %[[B:.*]]: vector<2xi32>
//  CHECK:        return %[[A]], %[[B]] : vector<4x8xf32>, vector<2xi32>
func.func @bitcast_folding(%I1: vector<4x8xf32>, %I2: vector<2xi32>) -> (vector<4x8xf32>, vector<2xi32>) {
  %0 = vector.bitcast %I1 : vector<4x8xf32> to vector<4x8xf32>
  %1 = vector.bitcast %I2 : vector<2xi32> to vector<4xi16>
  %2 = vector.bitcast %1 : vector<4xi16> to vector<2xi32>
  return %0, %2 : vector<4x8xf32>, vector<2xi32>
}

// -----

// CHECK-LABEL: func @bitcast_f16_to_f32
//              bit pattern: 0x40004000
//       CHECK-DAG: %[[CST1:.+]] = arith.constant dense<2.00390625> : vector<4xf32>
//              bit pattern: 0x00000000
//       CHECK-DAG: %[[CST0:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
//       CHECK: return %[[CST0]], %[[CST1]]
func.func @bitcast_f16_to_f32() -> (vector<4xf32>, vector<4xf32>) {
  %cst0 = arith.constant dense<0.0> : vector<8xf16> // bit pattern: 0x0000
  %cst1 = arith.constant dense<2.0> : vector<8xf16> // bit pattern: 0x4000
  %cast0 = vector.bitcast %cst0: vector<8xf16> to vector<4xf32>
  %cast1 = vector.bitcast %cst1: vector<8xf16> to vector<4xf32>
  return %cast0, %cast1: vector<4xf32>, vector<4xf32>
}

// -----

// CHECK-LABEL: func @bitcast_i8_to_i32
//              bit pattern: 0xA0A0A0A0
//       CHECK-DAG: %[[CST1:.+]] = arith.constant dense<-1600085856> : vector<4xi32>
//              bit pattern: 0x00000000
//       CHECK-DAG: %[[CST0:.+]] = arith.constant dense<0> : vector<4xi32>
//       CHECK: return %[[CST0]], %[[CST1]]
func.func @bitcast_i8_to_i32() -> (vector<4xi32>, vector<4xi32>) {
  %cst0 = arith.constant dense<0> : vector<16xi8> // bit pattern: 0x00
  %cst1 = arith.constant dense<160> : vector<16xi8> // bit pattern: 0xA0
  %cast0 = vector.bitcast %cst0: vector<16xi8> to vector<4xi32>
  %cast1 = vector.bitcast %cst1: vector<16xi8> to vector<4xi32>
  return %cast0, %cast1: vector<4xi32>, vector<4xi32>
}

// -----

// CHECK-LABEL: broadcast_poison
//       CHECK:  %[[POISON:.*]] = ub.poison : vector<4x6xi8>
//       CHECK:  return %[[POISON]] : vector<4x6xi8>
func.func @broadcast_poison() -> vector<4x6xi8> {
  %poison = ub.poison : vector<6xi8>
  %broadcast = vector.broadcast %poison : vector<6xi8> to vector<4x6xi8>
  return %broadcast : vector<4x6xi8>
}

// -----

// CHECK-LABEL:  broadcast_splat_constant
//       CHECK:  %[[CONST:.*]] = arith.constant dense<1> : vector<4x6xi8>
//       CHECK:  return %[[CONST]] : vector<4x6xi8>
func.func @broadcast_splat_constant() -> vector<4x6xi8> {
  %cst = arith.constant dense<1> : vector<6xi8>
  %broadcast = vector.broadcast %cst : vector<6xi8> to vector<4x6xi8>
  return %broadcast : vector<4x6xi8>
}

// -----

// CHECK-LABEL: broadcast_folding1
//       CHECK: %[[CST:.*]] = arith.constant dense<42> : vector<4xi32>
//   CHECK-NOT: vector.broadcast
//       CHECK: return %[[CST]]
func.func @broadcast_folding1() -> vector<4xi32> {
  %0 = arith.constant 42 : i32
  %1 = vector.broadcast %0 : i32 to vector<4xi32>
  return %1 : vector<4xi32>
}

// -----

// CHECK-LABEL: @broadcast_folding2
//       CHECK: %[[CST:.*]] = arith.constant dense<42> : vector<4x16xi32>
//   CHECK-NOT: vector.broadcast
//       CHECK: return %[[CST]]
func.func @broadcast_folding2() -> vector<4x16xi32> {
  %0 = arith.constant 42 : i32
  %1 = vector.broadcast %0 : i32 to vector<16xi32>
  %2 = vector.broadcast %1 : vector<16xi32> to vector<4x16xi32>
  return %2 : vector<4x16xi32>
}

// -----

// CHECK-LABEL: @fold_consecutive_broadcasts(
//  CHECK-SAME:                              %[[ARG0:.*]]: i32
//       CHECK: %[[RESULT:.*]] = vector.broadcast %[[ARG0]] : i32 to vector<4x16xi32>
//       CHECK: return %[[RESULT]]
func.func @fold_consecutive_broadcasts(%a : i32) -> vector<4x16xi32> {
  %1 = vector.broadcast %a : i32 to vector<16xi32>
  %2 = vector.broadcast %1 : vector<16xi32> to vector<4x16xi32>
  return %2 : vector<4x16xi32>
}

// -----

// CHECK-LABEL: shape_cast_splat_constant
//       CHECK-DAG: %[[CST1:.*]] = arith.constant dense<1> : vector<3x4x2xi32>
//       CHECK-DAG: %[[CST0:.*]] = arith.constant dense<2.000000e+00> : vector<20x2xf32>
//       CHECK: return %[[CST0]], %[[CST1]] : vector<20x2xf32>, vector<3x4x2xi32>
func.func @shape_cast_splat_constant() -> (vector<20x2xf32>, vector<3x4x2xi32>) {
  %cst = arith.constant dense<2.000000e+00> : vector<5x4x2xf32>
  %cst_1 = arith.constant dense<1> : vector<12x2xi32>
  %0 = vector.shape_cast %cst : vector<5x4x2xf32> to vector<20x2xf32>
  %1 = vector.shape_cast %cst_1 : vector<12x2xi32> to vector<3x4x2xi32>
  return %0, %1 : vector<20x2xf32>, vector<3x4x2xi32>
}

// -----

// Test of shape_cast's fold method:
// shape_cast(constant) -> constant.
//
// CHECK-LABEL: @shape_cast_dense_int_constant
//               CHECK: %[[CST:.*]] = arith.constant
// CHECK-SAME{LITERAL}: dense<[[2, 3, 5], [7, 11, 13]]>
//               CHECK: return %[[CST]] : vector<2x3xi8>
func.func @shape_cast_dense_int_constant() -> vector<2x3xi8> {
  %cst = arith.constant dense<[2, 3, 5, 7, 11, 13]> : vector<6xi8>
  %0 = vector.shape_cast %cst : vector<6xi8> to vector<2x3xi8>
  return %0 : vector<2x3xi8>
}

// -----

// Test of shape_cast fold's method:
// (shape_cast(const_x), const_x) -> (const_x_folded, const_x)
//
// CHECK-LABEL: @shape_cast_dense_float_constant
//  CHECK-DAG: %[[CST0:.*]] = {{.*}}1.000000e+00, 2.000000e+00{{.*}} vector<1x2xf32>
//  CHECK-DAG: %[[CST1:.*]] = {{.*}}1.000000e+00, 2.000000e+00{{.*}} vector<2xf32>
//      CHECK: return %[[CST1]], %[[CST0]] : vector<2xf32>, vector<1x2xf32>
func.func @shape_cast_dense_float_constant() -> (vector<2xf32>, vector<1x2xf32>){
  %cst = arith.constant dense<[[1.0, 2.0]]> : vector<1x2xf32>
  %0 = vector.shape_cast %cst : vector<1x2xf32> to vector<2xf32>
  return %0, %cst : vector<2xf32>, vector<1x2xf32>
}

// -----

// CHECK-LABEL: shape_cast_poison
//       CHECK-DAG: %[[CST1:.*]] = ub.poison : vector<3x4x2xi32>
//       CHECK-DAG: %[[CST0:.*]] = ub.poison : vector<20x2xf32>
//       CHECK: return %[[CST0]], %[[CST1]] : vector<20x2xf32>, vector<3x4x2xi32>
func.func @shape_cast_poison() -> (vector<20x2xf32>, vector<3x4x2xi32>) {
  %poison = ub.poison : vector<5x4x2xf32>
  %poison_1 = ub.poison : vector<12x2xi32>
  %0 = vector.shape_cast %poison : vector<5x4x2xf32> to vector<20x2xf32>
  %1 = vector.shape_cast %poison_1 : vector<12x2xi32> to vector<3x4x2xi32>
  return %0, %1 : vector<20x2xf32>, vector<3x4x2xi32>
}

// -----

// CHECK-LABEL: extract_strided_constant
//       CHECK-DAG: %[[CST1:.*]] = arith.constant dense<1> : vector<2x13x3xi32>
//       CHECK-DAG: %[[CST0:.*]] = arith.constant dense<2.000000e+00> : vector<12x2xf32>
//       CHECK: return %[[CST0]], %[[CST1]] : vector<12x2xf32>, vector<2x13x3xi32>
func.func @extract_strided_constant() -> (vector<12x2xf32>, vector<2x13x3xi32>) {
  %cst = arith.constant dense<2.000000e+00> : vector<29x7xf32>
  %cst_1 = arith.constant dense<1> : vector<4x37x9xi32>
  %0 = vector.extract_strided_slice %cst
    {offsets = [2, 3], sizes = [12, 2], strides = [1, 1]}
      : vector<29x7xf32> to vector<12x2xf32>
  %1 = vector.extract_strided_slice %cst_1
    {offsets = [1, 2, 5], sizes = [2, 13, 3], strides = [1, 1, 1]}
      : vector<4x37x9xi32> to vector<2x13x3xi32>
  return %0, %1 : vector<12x2xf32>, vector<2x13x3xi32>
}

// -----

// CHECK-LABEL: extract_strided_broadcast
//       CHECK:   %[[B:.*]] = vector.broadcast %{{.*}} : vector<4xf16> to vector<2x4xf16>
//  CHECK-NEXT:   return %[[B]] : vector<2x4xf16>
func.func @extract_strided_broadcast(%arg0: vector<4xf16>) -> vector<2x4xf16> {
 %0 = vector.broadcast %arg0 : vector<4xf16> to vector<16x4xf16>
 %1 = vector.extract_strided_slice %0
  {offsets = [0, 0], sizes = [2, 4], strides = [1, 1]} :
  vector<16x4xf16> to vector<2x4xf16>
  return %1 : vector<2x4xf16>
}

// -----

// CHECK-LABEL: extract_strided_broadcast2
//       CHECK:   %[[E:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0], sizes = [2], strides = [1]} : vector<4xf16> to vector<2xf16>
//  CHECK-NEXT:   %[[B:.*]] = vector.broadcast %[[E]] : vector<2xf16> to vector<2x2xf16>
//  CHECK-NEXT:   return %[[B]] : vector<2x2xf16>
func.func @extract_strided_broadcast2(%arg0: vector<4xf16>) -> vector<2x2xf16> {
 %0 = vector.broadcast %arg0 : vector<4xf16> to vector<16x4xf16>
 %1 = vector.extract_strided_slice %0
  {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} :
  vector<16x4xf16> to vector<2x2xf16>
  return %1 : vector<2x2xf16>
}

// -----

// CHECK-LABEL: func @extract_strided_broadcast3
//  CHECK-SAME: (%[[ARG:.+]]: vector<1xf32>)
//       CHECK: %[[V:.+]] = vector.broadcast %[[ARG]] : vector<1xf32> to vector<1x4xf32>
//       CHECK: return %[[V]]
func.func @extract_strided_broadcast3(%arg0: vector<1xf32>) -> vector<1x4xf32> {
 %0 = vector.broadcast %arg0 : vector<1xf32> to vector<1x8xf32>
 %1 = vector.extract_strided_slice %0
      {offsets = [0, 4], sizes = [1, 4], strides = [1, 1]}
      : vector<1x8xf32> to vector<1x4xf32>
  return %1 : vector<1x4xf32>
}

// -----

// CHECK-LABEL: func @extract_strided_broadcast4
//  CHECK-SAME: (%[[ARG:.+]]: f32)
//       CHECK: %[[V:.+]] = vector.broadcast %[[ARG]] : f32 to vector<1x4xf32>
//       CHECK: return %[[V]]
func.func @extract_strided_broadcast4(%arg0: f32) -> vector<1x4xf32> {
 %0 = vector.broadcast %arg0 : f32 to vector<1x8xf32>
 %1 = vector.extract_strided_slice %0
      {offsets = [0, 4], sizes = [1, 4], strides = [1, 1]}
      : vector<1x8xf32> to vector<1x4xf32>
  return %1 : vector<1x4xf32>
}

// -----

// Check the case where the same dimension is both broadcasted and sliced 
// CHECK-LABEL: func @extract_strided_broadcast5
//  CHECK-SAME: (%[[ARG:.+]]: vector<2x1xf32>)
//       CHECK: %[[V:.+]] = vector.broadcast %[[ARG]] : vector<2x1xf32> to vector<2x4xf32>
//       CHECK: return %[[V]]
func.func @extract_strided_broadcast5(%arg0: vector<2x1xf32>) -> vector<2x4xf32> {
 %0 = vector.broadcast %arg0 : vector<2x1xf32> to vector<2x8xf32>
 %1 = vector.extract_strided_slice %0
      {offsets = [0, 4], sizes = [2, 4], strides = [1, 1]}
      : vector<2x8xf32> to vector<2x4xf32>
  return %1 : vector<2x4xf32>
}

// -----

// CHECK-LABEL: consecutive_shape_cast
//       CHECK:   %[[C:.*]] = vector.shape_cast %{{.*}} : vector<16xf16> to vector<4x4xf16>
//  CHECK-NEXT:   return %[[C]] : vector<4x4xf16>
func.func @consecutive_shape_cast(%arg0: vector<16xf16>) -> vector<4x4xf16> {
  %0 = vector.shape_cast %arg0 : vector<16xf16> to vector<2x8xf16>
  %1 = vector.shape_cast %0 : vector<2x8xf16> to vector<4x4xf16>
  return %1 : vector<4x4xf16>
}

// -----

// CHECK-LABEL: func @dead_transfer_op
//   CHECK-NOT:   vector.transfer_read
//   CHECK-NOT:   vector.transfer_write
//       CHECK:   return
func.func @dead_transfer_op(%arg0 : tensor<4x4xf32>, %arg1 : memref<4x4xf32>,
                       %v0 : vector<1x4xf32>) {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %r = vector.transfer_read %arg1[%c0, %c0], %cf0 :
    memref<4x4xf32>, vector<1x4xf32>
  %w = vector.transfer_write %v0, %arg0[%c0, %c0] :
    vector<1x4xf32>, tensor<4x4xf32>
  return
}

// -----

// CHECK-LABEL: func @dead_load
//   CHECK-NOT:   vector.maskedload
//   CHECK-NOT:   vector.gather
//   CHECK-NOT:   vector.expandload
//       CHECK:   return
func.func @dead_load(%base: memref<?xf32>, %indices: vector<16xi32>,
                          %mask: vector<16xi1>, %passthru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %0 = vector.maskedload %base[%c0], %mask, %passthru :
    memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %1 = vector.gather %base[%c0][%indices], %mask, %passthru :
    memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %2 = vector.expandload %base[%c0], %mask, %passthru :
    memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return
}

// -----

#contraction_accesses0 = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#contraction_trait0 = {
  indexing_maps = #contraction_accesses0,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @contractions
//  CHECK-SAME:   %[[A:[0-9a-zA-Z]+]]: vector<2x3xf32>
//  CHECK-SAME:   %[[B:[0-9a-zA-Z]+]]: vector<3x4xf32>
//  CHECK-SAME:   %[[C:[0-9a-zA-Z]+]]: vector<2x4xf32>
//  CHECK-SAME:   %[[A_I8:[0-9a-zA-Z]+]]: vector<2x3xi8>
//  CHECK-SAME:   %[[B_I8:[0-9a-zA-Z]+]]: vector<3x4xi8>
//  CHECK-SAME:   %[[C_I8:[0-9a-zA-Z]+]]: vector<2x4xi8>
func.func @contractions(%a: vector<2x3xf32>, %b: vector<3x4xf32>, %c: vector<2x4xf32>,
                   %a_i8: vector<2x3xi8>, %b_i8: vector<3x4xi8>, %c_i8: vector<2x4xi8>)
  -> (vector<2x4xf32>, vector<2x4xi8>)
{
  // CHECK-NOT: arith.constant
  %vf_0 = arith.constant dense <0.0>: vector<2x4xf32>
  // CHECK-NOT: arith.addf
  //     CHECK: %[[D:.*]] = vector.contract {{.*}} %[[A]], %[[B]], %[[C]]
  %0 = vector.contract #contraction_trait0 %a, %b, %vf_0:
    vector<2x3xf32>, vector<3x4xf32> into vector<2x4xf32>
  // CHECK-NOT: arith.addf
  %1 = arith.addf %0, %c: vector<2x4xf32>

  // CHECK-NOT: arith.constant
  %vi8_0 = arith.constant dense <0>: vector<2x4xi8>
  // CHECK-NOT: arith.addi
  //     CHECK: %[[D_I8:.*]] = vector.contract {{.*}} %[[A_I8]], %[[B_I8]], %[[C_I8]]
  %i8_0 = vector.contract #contraction_trait0 %a_i8, %b_i8, %vi8_0:
    vector<2x3xi8>, vector<3x4xi8> into vector<2x4xi8>
  // CHECK-NOT: arith.addi
  %i8_1 = arith.addi %i8_0, %c_i8: vector<2x4xi8>

  // CHECK: return %[[D]], %[[D_I8]]
  return %1, %i8_1: vector<2x4xf32>, vector<2x4xi8>
}

// -----

// CHECK-LABEL: func @transfer_folding_1
//  CHECK-SAME:   %[[T0:[0-9a-zA-Z]+]]: tensor<2x3x4xf32>
//  CHECK-SAME:   %[[T1:[0-9a-zA-Z]+]]: tensor<2x3x4xf32>
func.func @transfer_folding_1(%t0: tensor<2x3x4xf32>, %t1: tensor<2x3x4xf32>)
  -> (tensor<2x3x4xf32>, tensor<2x3x4xf32>, tensor<2x3x4xf32>)
{
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %v = vector.transfer_read %t0[%c0, %c0, %c0], %pad {in_bounds = [true, true, true]} :
    tensor<2x3x4xf32>, vector<2x3x4xf32>

  %r0 = vector.transfer_write %v, %t1[%c0, %c0, %c0] {in_bounds = [true, true, true]} :
    vector<2x3x4xf32>, tensor<2x3x4xf32>

  %t2 = "test.constant"() { value = dense<6.0> : tensor<2x3x4xf32>} : () -> (tensor<2x3x4xf32>)
  %r1 = vector.transfer_write %v, %t2[%c0, %c0, %c0] {in_bounds = [true, true, true]} :
    vector<2x3x4xf32>, tensor<2x3x4xf32>


  // CHECK-NEXT: some_op_that_may_have_side_effects
  %t3 = "some_op_that_may_have_side_effects"() : () -> (tensor<2x3x4xf32>)
  %r2 = vector.transfer_write %v, %t0[%c0, %c0, %c0] {in_bounds = [true, true, true]} :
    vector<2x3x4xf32>, tensor<2x3x4xf32>

  // CHECK-NEXT: return %[[T0]], %[[T0]], %[[T0]]
  return %r0, %r1, %r2: tensor<2x3x4xf32>, tensor<2x3x4xf32>, tensor<2x3x4xf32>
}

// -----

// CHECK-LABEL: func @store_after_load_tensor
//  CHECK-SAME: (%[[ARG:.*]]: tensor<4x4xf32>)
//   CHECK-NOT:   vector.transfer_read
//   CHECK-NOT:   vector.transfer_write
//       CHECK:   return %[[ARG]] : tensor<4x4xf32>
func.func @store_after_load_tensor(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c1, %c0], %cf0 :
    tensor<4x4xf32>, vector<1x4xf32>
  %w0 = vector.transfer_write %0, %arg0[%c1, %c0] :
    vector<1x4xf32>, tensor<4x4xf32>
  return %w0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @negative_store_after_load_tensor
//       CHECK:   vector.transfer_read
//       CHECK:   vector.transfer_write
//       CHECK:   return
func.func @negative_store_after_load_tensor(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c1, %c0], %cf0 :
    tensor<4x4xf32>, vector<1x4xf32>
  %w0 = vector.transfer_write %0, %arg0[%c0, %c0] :
    vector<1x4xf32>, tensor<4x4xf32>
  return %w0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @store_to_load_tensor
//  CHECK-SAME: (%[[ARG:.*]]: tensor<4x4xf32>, %[[V0:.*]]: vector<1x4xf32>, %[[V1:.*]]: vector<1x4xf32>)
//   CHECK-NOT:   vector.transfer_write
//   CHECK-NOT:   vector.transfer_read
//       CHECK:   return %[[V0]] : vector<1x4xf32>
func.func @store_to_load_tensor(%arg0 : tensor<4x4xf32>,
  %v0 : vector<1x4xf32>, %v1 : vector<1x4xf32>) -> vector<1x4xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %w1 = vector.transfer_write %v1, %w0[%c2, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %0 = vector.transfer_read %w1[%c1, %c0], %cf0 {in_bounds = [true, true]} :
    tensor<4x4xf32>, vector<1x4xf32>
  return %0 : vector<1x4xf32>
}

// -----

// CHECK-LABEL: func @negative_store_to_load_tensor
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_write
//       CHECK:   %[[V:.*]] = vector.transfer_read
//       CHECK:   return %[[V]] : vector<1x4xf32>
func.func @negative_store_to_load_tensor(%arg0 : tensor<4x4xf32>,
  %v0 : vector<1x4xf32>, %v1 : vector<1x4xf32>, %i : index) -> vector<1x4xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %w1 = vector.transfer_write %v0, %w0[%i, %i] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %0 = vector.transfer_read %w1[%c1, %c0], %cf0 {in_bounds = [true, true]} :
    tensor<4x4xf32>, vector<1x4xf32>
  return %0 : vector<1x4xf32>
}

// -----

// CHECK-LABEL: func @store_to_load_tensor_broadcast
//  CHECK-SAME: (%[[ARG:.*]]: tensor<4x4xf32>, %[[V0:.*]]: vector<4x2xf32>)
//       CHECK:   %[[B:.*]] = vector.broadcast %[[V0]] : vector<4x2xf32> to vector<6x4x2xf32>
//       CHECK:   %[[T:.*]] = vector.transpose %[[B]], [1, 2, 0] : vector<6x4x2xf32> to vector<4x2x6xf32>
//       CHECK:   return %[[T]] : vector<4x2x6xf32>
func.func @store_to_load_tensor_broadcast(%arg0 : tensor<4x4xf32>,
  %v0 : vector<4x2xf32>) -> vector<4x2x6xf32> {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %w0 = vector.transfer_write %v0, %arg0[%c0, %c0] {in_bounds = [true, true]} :
    vector<4x2xf32>, tensor<4x4xf32>
  %0 = vector.transfer_read %w0[%c0, %c0], %cf0 {in_bounds = [true, true, true],
  permutation_map = affine_map<(d0, d1) -> (d0, d1, 0)>} :
    tensor<4x4xf32>, vector<4x2x6xf32>
  return %0 : vector<4x2x6xf32>
}

// -----

// CHECK-LABEL: func @negative_store_to_load_tensor_memref
//   CHECK-NOT:   vector.broadcast
//   CHECK-NOT:   vector.transpose
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_read
func.func @negative_store_to_load_tensor_memref(
    %arg0 : tensor<?x?xf32>,
    %arg1 : memref<?x?xf32>,
    %v0 : vector<4x2xf32>
  ) -> vector<4x2xf32>
{
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  vector.transfer_write %v0, %arg1[%c0, %c0] {in_bounds = [true, true]} :
    vector<4x2xf32>, memref<?x?xf32>
  %0 = vector.transfer_read %arg0[%c0, %c0], %cf0 {in_bounds = [true, true]} :
    tensor<?x?xf32>, vector<4x2xf32>
  return %0 : vector<4x2xf32>
}

// -----

// CHECK-LABEL: func @negative_store_to_load_tensor_no_actual_broadcast
//   CHECK-NOT:   vector.broadcast
//   CHECK-NOT:   vector.transpose
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_read
func.func @negative_store_to_load_tensor_no_actual_broadcast(%arg0 : tensor<?x?xf32>,
  %v0 : vector<4x2xf32>) -> vector<4x2xf32> {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %w0 = vector.transfer_write %v0, %arg0[%c0, %c0] :
    vector<4x2xf32>, tensor<?x?xf32>
  %0 = vector.transfer_read %w0[%c0, %c0], %cf0 {in_bounds = [true, true]} :
    tensor<?x?xf32>, vector<4x2xf32>
  return %0 : vector<4x2xf32>
}

// -----

// CHECK-LABEL: func @negative_store_to_load_tensor_broadcast_out_of_bounds
//   CHECK-NOT:   vector.broadcast
//   CHECK-NOT:   vector.transpose
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_read
func.func @negative_store_to_load_tensor_broadcast_out_of_bounds(%arg0 : tensor<?x?xf32>,
  %v0 : vector<4x2xf32>) -> vector<4x2x6xf32> {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %w0 = vector.transfer_write %v0, %arg0[%c0, %c0] :
    vector<4x2xf32>, tensor<?x?xf32>
  %0 = vector.transfer_read %w0[%c0, %c0], %cf0 {in_bounds = [true, true, true],
  permutation_map = affine_map<(d0, d1) -> (d0, d1, 0)>} :
    tensor<?x?xf32>, vector<4x2x6xf32>
  return %0 : vector<4x2x6xf32>
}

// -----

// CHECK-LABEL: func @negative_store_to_load_tensor_broadcast_masked
//   CHECK-NOT:   vector.broadcast
//   CHECK-NOT:   vector.transpose
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_read
func.func @negative_store_to_load_tensor_broadcast_masked(
    %arg0 : tensor<?x?xf32>, %v0 : vector<4x2xf32>, %mask : vector<4x2xi1>)
  -> vector<4x2x6xf32>
{
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %w0 = vector.transfer_write %v0, %arg0[%c0, %c0], %mask {in_bounds = [true, true]} :
    vector<4x2xf32>, tensor<?x?xf32>
  %0 = vector.transfer_read %w0[%c0, %c0], %cf0 {in_bounds = [true, true, true],
  permutation_map = affine_map<(d0, d1) -> (d0, d1, 0)>} :
    tensor<?x?xf32>, vector<4x2x6xf32>
  return %0 : vector<4x2x6xf32>
}

// -----

// CHECK-LABEL: func @store_to_load_tensor_broadcast_scalable
//  CHECK-SAME: (%[[ARG:.*]]: tensor<?xf32>, %[[V0:.*]]: vector<[4]xf32>)
//       CHECK:   %[[B:.*]] = vector.broadcast %[[V0]] : vector<[4]xf32> to vector<6x[4]xf32>
//       CHECK:   return %[[B]] : vector<6x[4]xf32>
func.func @store_to_load_tensor_broadcast_scalable(%arg0 : tensor<?xf32>,
  %v0 : vector<[4]xf32>) -> vector<6x[4]xf32> {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %w0 = vector.transfer_write %v0, %arg0[%c0] {in_bounds = [true]} :
    vector<[4]xf32>, tensor<?xf32>
  %0 = vector.transfer_read %w0[%c0], %cf0 {in_bounds = [true, true],
  permutation_map = affine_map<(d0) -> (0, d0)>} :
    tensor<?xf32>, vector<6x[4]xf32>
  return %0 : vector<6x[4]xf32>
}

// -----

// CHECK-LABEL: func @store_to_load_tensor_perm_broadcast
//  CHECK-SAME: (%[[ARG:.*]]: tensor<4x4x4xf32>, %[[V0:.*]]: vector<4x1xf32>)
//       CHECK:   %[[B:.*]] = vector.broadcast %[[V0]] : vector<4x1xf32> to vector<100x5x4x1xf32>
//       CHECK:   %[[T:.*]] = vector.transpose %[[B]], [3, 0, 2, 1] : vector<100x5x4x1xf32> to vector<1x100x4x5xf32>
//       CHECK:   return %[[T]] : vector<1x100x4x5xf32>
func.func @store_to_load_tensor_perm_broadcast(%arg0 : tensor<4x4x4xf32>,
  %v0 : vector<4x1xf32>) -> vector<1x100x4x5xf32> {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %w0 = vector.transfer_write %v0, %arg0[%c0, %c0, %c0] {in_bounds = [true, true],
  permutation_map = affine_map<(d0, d1, d2) -> (d2, d1)>} :
    vector<4x1xf32>, tensor<4x4x4xf32>
  %0 = vector.transfer_read %w0[%c0, %c0, %c0], %cf0 {in_bounds = [true, true, true, true],
  permutation_map = affine_map<(d0, d1, d2) -> (d1, 0, d2, 0)>} :
    tensor<4x4x4xf32>, vector<1x100x4x5xf32>
  return %0 : vector<1x100x4x5xf32>
}

// -----


// CHECK-LABEL: func @dead_store_tensor
//   CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:      %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:      %[[C2:.*]] = arith.constant 2 : index
//   CHECK-NOT:   vector.transfer_write {{.*}}, {{.*}}[%[[C1]], %[[C0]]
//       CHECK:   vector.transfer_write {{.*}}, {{.*}}[%[[C2]], %[[C0]]
//       CHECK:   %[[VTW:.*]] = vector.transfer_write {{.*}}, {{.*}}[%[[C1]], %[[C0]]
//       CHECK:   return %[[VTW]] : tensor<4x4xf32>
func.func @dead_store_tensor(%arg0 : tensor<4x4xf32>,
  %v0 : vector<1x4xf32>, %v1 : vector<1x4xf32>, %i : index) -> tensor<4x4xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %w1 = vector.transfer_write %v0, %w0[%c2, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %w2 = vector.transfer_write %v1, %w1[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  return %w2 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @negative_dead_store_tensor
//   CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:      %[[C1:.*]] = arith.constant 1 : index
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_read
//       CHECK:   %[[VTW:.*]] = vector.transfer_write {{.*}}, {{.*}}[%[[C1]], %[[C0]]]
//       CHECK:   return %[[VTW]] : tensor<4x4xf32>
func.func @negative_dead_store_tensor(%arg0 : tensor<4x4xf32>,
  %v0 : vector<1x4xf32>, %v1 : vector<1x4xf32>, %i : index) -> tensor<4x4xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %w1 = vector.transfer_write %v0, %w0[%c2, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %0 = vector.transfer_read %w1[%i, %i], %cf0 {in_bounds = [true, true]} :
    tensor<4x4xf32>, vector<1x4xf32>
  %x = arith.addf %0, %0 : vector<1x4xf32>
  %w2 = vector.transfer_write %x, %w0[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  return %w2 : tensor<4x4xf32>
}

// -----

//       CHECK: #[[$MAP:[0-9a-z]+]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func @swap_extract_slice_transfer_write
//  CHECK-SAME:   %[[VEC:.*]]: vector<8x4xf32>
//  CHECK-SAME:   %[[INIT_TENSOR:.*]]: tensor<4x8xf32>,
//  CHECK-SAME:   %[[ITER_ARG:.*]]: tensor<64x64xf32>,
//  CHECK-SAME:   %[[IV:.*]]: index, %[[SZ:.*]]: index)
func.func @swap_extract_slice_transfer_write(%arg0 : vector<8x4xf32>,
                                             %arg1 : tensor<4x8xf32>,
                                             %arg2 : tensor<64x64xf32>,
                                             %iv : index, %sz : index) -> tensor<64x64xf32> {
  //       CHECK:   %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index

  //       CHECK:   %[[T0:.*]] = tensor.extract_slice %[[ITER_ARG]]
  //  CHECK-SAME:                 [%[[IV]], 16] [%[[SZ]], 8]
  //       CHECK:   %[[T1:.*]] = vector.transfer_write %[[VEC]]
  //  CHECK-SAME:                 %[[T0]][%[[C0]], %[[C0]]]
  //  CHECK-SAME:                 in_bounds = [true, false]
  //  CHECK-SAME:                 permutation_map = #[[$MAP]]
  //       CHECK:   %[[T2:.*]] = tensor.insert_slice %[[T1]] into %[[ITER_ARG]]
  //  CHECK-SAME:                 [%[[IV]], 16] [%[[SZ]], 8]
  %0 = vector.transfer_write %arg0, %arg1[%c0, %c0] {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : vector<8x4xf32>, tensor<4x8xf32>
  %1 = tensor.extract_slice %0[0, 0] [%sz, 8] [1, 1] : tensor<4x8xf32> to tensor<?x8xf32>
  %2 = tensor.insert_slice %1 into %arg2[%iv, 16] [%sz, 8] [1, 1] : tensor<?x8xf32> into tensor<64x64xf32>

  //       CHECK:   return %[[T2]]
  func.return %2 : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func @do_not_swap_extract_slice_transfer_write
//  CHECK-SAME:   %[[VEC:.*]]: vector<8xf32>,
//  CHECK-SAME:   %[[VEC_SMALL:.*]]: vector<4xf32>,
//  CHECK-SAME:   %[[INIT_TENSOR:.*]]: tensor<8xf32>,
//  CHECK-SAME:   %[[ITER_ARG:.*]]: tensor<64xf32>,
//  CHECK-SAME:   %[[IV:.*]]: index, %[[SZ:.*]]: index)
func.func @do_not_swap_extract_slice_transfer_write(%arg0 : vector<8xf32>,
                                                    %arg1 : vector<4xf32>,
                                                    %arg2 : tensor<8xf32>,
                                                    %arg3 : tensor<64xf32>,
                                                    %iv : index, %sz : index) -> (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) {
  //       CHECK:   %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index

  // Don't swap if the extracted and inserted slices do not match.
  //       CHECK:   %[[T0:.*]] = vector.transfer_write %[[VEC]]
  //       CHECK:   %[[T1:.*]] = tensor.extract_slice %[[T0]]
  //       CHECK:   %[[T2:.*]] = tensor.insert_slice %[[T1]]
  %0 = vector.transfer_write %arg0, %arg2[%c0] {in_bounds = [true]} : vector<8xf32>, tensor<8xf32>
  %1 = tensor.extract_slice %0[0] [%iv] [1] : tensor<8xf32> to tensor<?xf32>
  %2 = tensor.insert_slice %1 into %arg3[%iv] [%sz] [1] : tensor<?xf32> into tensor<64xf32>

  // Don't swap if the TransferWriteOp takes a small vector.
  //       CHECK:   %[[T3:.*]] = vector.transfer_write %[[VEC_SMALL]]
  //       CHECK:   %[[T4:.*]] = tensor.extract_slice %[[T3]]
  //       CHECK:   %[[T5:.*]] = tensor.insert_slice %[[T4]]
  %3 = vector.transfer_write %arg1, %arg2[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<8xf32>
  %4 = tensor.extract_slice %3[0] [%sz] [1] : tensor<8xf32> to tensor<?xf32>
  %5 = tensor.insert_slice %4 into %arg3[%iv] [%sz] [1] : tensor<?xf32> into tensor<64xf32>

  // Don't swap if the one of the operations is rank-reducing.
  //       CHECK:   %[[T6:.*]] = vector.transfer_write %[[VEC]]
  //       CHECK:   %[[T7:.*]] = tensor.extract_slice %[[T6]]
  //       CHECK:   %[[T8:.*]] = tensor.insert_slice %[[T7]]
  %6 = vector.transfer_write %arg0, %arg2[%c0] {in_bounds = [true]} : vector<8xf32>, tensor<8xf32>
  %7 = tensor.extract_slice %6[0] [1] [1] : tensor<8xf32> to tensor<f32>
  %8 = tensor.insert_slice %7 into %arg3[%iv] [1] [1] : tensor<f32> into tensor<64xf32>

  //       CHECK:   return %[[T2]], %[[T5]], %[[T8]]
  func.return %2, %5, %8 : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
}

// -----

// CHECK-LABEL: func @vector_multi_reduction_single_parallel(
//  CHECK-SAME:     %[[v:.*]]: vector<2xf32>,
func.func @vector_multi_reduction_single_parallel(%arg0: vector<2xf32>, %acc: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <mul>, %arg0, %acc [] : vector<2xf32> to vector<2xf32>

//       CHECK:     return %[[v]] : vector<2xf32>
    return %0 : vector<2xf32>
}

// -----

// CHECK-LABEL: func @masked_vector_multi_reduction_single_parallel(
//  CHECK-SAME:     %[[VAL_0:.*]]: vector<2xf32>, %{{.*}}: vector<2xf32>,
func.func @masked_vector_multi_reduction_single_parallel(%arg0: vector<2xf32>, %acc: vector<2xf32>, %mask: vector<2xi1>) -> vector<2xf32> {
    %0 = vector.mask %mask { vector.multi_reduction <mul>, %arg0, %acc [] : vector<2xf32> to vector<2xf32> } : vector<2xi1> -> vector<2xf32>
//       CHECK:   return %[[VAL_0]] : vector<2xf32>
    return %0 : vector<2xf32>
}

// -----

// CHECK-LABEL: func @vector_multi_reduction_unit_dimensions(
//  CHECK-SAME: %[[SOURCE:.+]]: vector<5x1x4x1x20xf32>, %[[ACC:.+]]: vector<5x4x20xf32>
func.func @vector_multi_reduction_unit_dimensions(%source: vector<5x1x4x1x20xf32>, %acc: vector<5x4x20xf32>) -> vector<5x4x20xf32> {
//       CHECK:   %[[CAST:.+]] = vector.shape_cast  %[[SOURCE]] : vector<5x1x4x1x20xf32> to vector<5x4x20xf32>
//       CHECK:   %[[RESULT:.+]] = arith.mulf  %[[ACC]], %[[CAST]] : vector<5x4x20xf32>
    %0 = vector.multi_reduction <mul>, %source, %acc [1, 3] : vector<5x1x4x1x20xf32> to vector<5x4x20xf32>

//       CHECK:     return %[[RESULT]] : vector<5x4x20xf32>
    return %0 : vector<5x4x20xf32>
}

// -----

// CHECK-LABEL:   func.func @vector_multi_reduction_scalable(
// CHECK-SAME:     %[[VAL_0:.*]]: vector<1x[4]x1xf32>,
// CHECK-SAME:     %[[VAL_1:.*]]: vector<1x[4]xf32>,
// CHECK-SAME:     %[[VAL_2:.*]]: vector<1x[4]x1xi1>)
func.func @vector_multi_reduction_scalable(%source: vector<1x[4]x1xf32>,
                                           %acc: vector<1x[4]xf32>,
                                           %mask: vector<1x[4]x1xi1>) -> vector<1x[4]xf32> {
// CHECK:           %[[VAL_3:.*]] = vector.shape_cast %[[VAL_2]] : vector<1x[4]x1xi1> to vector<1x[4]xi1>
// CHECK:           %[[VAL_4:.*]] = vector.shape_cast %[[VAL_0]] : vector<1x[4]x1xf32> to vector<1x[4]xf32>
// CHECK:           %[[VAL_5:.*]] = arith.addf %[[VAL_1]], %[[VAL_4]] : vector<1x[4]xf32>
// CHECK:           %[[VAL_6:.*]] = arith.select %[[VAL_3]], %[[VAL_5]], %[[VAL_4]] : vector<1x[4]xi1>, vector<1x[4]xf32>
    %0 = vector.mask %mask { vector.multi_reduction <add>, %source, %acc [2] : vector<1x[4]x1xf32> to vector<1x[4]xf32> } :
          vector<1x[4]x1xi1> -> vector<1x[4]xf32>

    return %0 : vector<1x[4]xf32>
}

// -----

// CHECK-LABEL: func @masked_vector_multi_reduction_unit_dimensions
//  CHECK-SAME: %[[VAL_0:.*]]: vector<5x1x4x1x20xf32>, %[[VAL_1:.*]]: vector<5x4x20xf32>,
//  CHECK-SAME: %[[VAL_2:.*]]: vector<5x1x4x1x20xi1>)
func.func @masked_vector_multi_reduction_unit_dimensions(%source: vector<5x1x4x1x20xf32>,
                                                         %acc: vector<5x4x20xf32>,
                                                         %mask: vector<5x1x4x1x20xi1>) -> vector<5x4x20xf32> {
//       CHECK:   %[[VAL_3:.*]] = vector.shape_cast %[[VAL_2]] : vector<5x1x4x1x20xi1> to vector<5x4x20xi1>
//       CHECK:   %[[VAL_4:.*]] = vector.shape_cast %[[VAL_0]] : vector<5x1x4x1x20xf32> to vector<5x4x20xf32>
//       CHECK:   %[[VAL_5:.*]] = arith.mulf %[[VAL_1]], %[[VAL_4]] : vector<5x4x20xf32>
//       CHECK:   %[[VAL_6:.*]] = arith.select %[[VAL_3]], %[[VAL_5]], %[[VAL_4]] : vector<5x4x20xi1>, vector<5x4x20xf32>
%0 = vector.mask %mask { vector.multi_reduction <mul>, %source, %acc [1, 3] : vector<5x1x4x1x20xf32> to vector<5x4x20xf32> } :
           vector<5x1x4x1x20xi1> -> vector<5x4x20xf32>
    return %0 : vector<5x4x20xf32>
}

// -----

// CHECK-LABEL: func @vector_multi_reduction_unit_dimensions_fail(
//  CHECK-SAME: %[[SRC:.+]]: vector<5x1x4x1x20xf32>, %[[ACCUM:.+]]: vector<5x1x20xf32>
func.func @vector_multi_reduction_unit_dimensions_fail(%source: vector<5x1x4x1x20xf32>, %acc: vector<5x1x20xf32>) -> vector<5x1x20xf32> {
//       CHECK:   %[[RES:.+]] = vector.multi_reduction  <mul>, %[[SRC]], %[[ACCUM]] [1, 2] : vector<5x1x4x1x20xf32> to vector<5x1x20xf32>
    %0 = vector.multi_reduction <mul>, %source, %acc [1, 2] : vector<5x1x4x1x20xf32> to vector<5x1x20xf32>

//       CHECK:     return %[[RES]] : vector<5x1x20xf32>
    return %0 : vector<5x1x20xf32>
}

// -----

// CHECK-LABEL: func @vector_multi_reduction_unit_dimensions_single_elem(
//  CHECK-SAME: %[[SOURCE:.+]]: vector<1x1x1xf32>, %[[ACC:.+]]: f32
func.func @vector_multi_reduction_unit_dimensions_single_elem(%source: vector<1x1x1xf32>, %acc: f32) -> f32 {
//       CHECK:   %[[CAST:.+]] = vector.extract  %[[SOURCE]][0, 0, 0] : f32 from vector<1x1x1xf32>
//       CHECK:   %[[RESULT:.+]] = arith.mulf  %[[ACC]], %[[CAST]] : f32
    %0 = vector.multi_reduction <mul>, %source, %acc [0,1,2] : vector<1x1x1xf32> to f32

//       CHECK:     return %[[RESULT]] : f32
    return %0 : f32
}

// -----

// CHECK-LABEL: func @masked_vector_multi_reduction_unit_dimensions_single_elem(
//  CHECK-SAME: %[[VAL_0:.*]]: vector<1x1x1xf32>, %[[VAL_1:.*]]: f32,
//  CHECK-SAME: %[[VAL_2:.*]]: vector<1x1x1xi1>)
func.func @masked_vector_multi_reduction_unit_dimensions_single_elem(%source: vector<1x1x1xf32>, %acc: f32, %mask: vector<1x1x1xi1>) -> f32 {
      // CHECK:           %[[VAL_3:.*]] = vector.extract %[[VAL_2]][0, 0, 0] : i1 from vector<1x1x1xi1>
      // CHECK:           %[[VAL_4:.*]] = vector.extract %[[VAL_0]][0, 0, 0] : f32 from vector<1x1x1xf32>
      // CHECK:           %[[VAL_5:.*]] = arith.mulf %[[VAL_1]], %[[VAL_4]] : f32
      // CHECK:           %[[VAL_6:.*]] = arith.select %[[VAL_3]], %[[VAL_5]], %[[VAL_4]] : f32
  %0 = vector.mask %mask { vector.multi_reduction <mul>, %source, %acc [0,1,2] : vector<1x1x1xf32> to f32 } : vector<1x1x1xi1> -> f32
    return %0 : f32
}

// -----

// CHECK-LABEL: func @insert_strided_slice_full_range
//  CHECK-SAME: %[[SOURCE:.+]]: vector<16x16xf16>, %{{.+}}: vector<16x16xf16>
func.func @insert_strided_slice_full_range(%source: vector<16x16xf16>, %dest: vector<16x16xf16>) -> vector<16x16xf16> {
  %0 = vector.insert_strided_slice %source, %dest {offsets = [0, 0], strides = [1, 1]} : vector<16x16xf16> into vector<16x16xf16>
  // CHECK: return %[[SOURCE]]
  return %0: vector<16x16xf16>
}

// -----

// CHECK-LABEL: extract_strided_splatlike
//       CHECK:   %[[B:.*]] = vector.broadcast %{{.*}} f16 to vector<2x4xf16>
//  CHECK-NEXT:   return %[[B]] : vector<2x4xf16>
func.func @extract_strided_splatlike(%arg0: f16) -> vector<2x4xf16> {
 %0 = vector.broadcast %arg0 : f16 to vector<16x4xf16>
 %1 = vector.extract_strided_slice %0
  {offsets = [1, 0], sizes = [2, 4], strides = [1, 1]} :
  vector<16x4xf16> to vector<2x4xf16>
  return %1 : vector<2x4xf16>
}

// -----

// CHECK-LABEL: func @insert_extract_to_broadcast
//  CHECK-SAME: (%[[ARG0:.*]]: vector<1x1x4xf32>, %[[ARG1:.*]]: vector<4xf32>)
//       CHECK:   %[[V0:.*]] = vector.extract %[[ARG0]][0, 0] : vector<4xf32> from vector<1x1x4xf32>
//       CHECK:   %[[V1:.*]] = vector.broadcast %[[ARG1]] : vector<4xf32> to vector<1x1x4xf32>
//       CHECK:   return %[[V0]], %[[V1]] : vector<4xf32>, vector<1x1x4xf32>
func.func @insert_extract_to_broadcast(%arg0 : vector<1x1x4xf32>,
  %arg1 : vector<4xf32>) -> (vector<4xf32>, vector<1x1x4xf32>) {
  %0 = vector.extract %arg0[0, 0] : vector<4xf32> from vector<1x1x4xf32>
  %1 = vector.insert %arg1, %arg0 [0, 0] : vector<4xf32> into vector<1x1x4xf32>
  return %0, %1 : vector<4xf32>, vector<1x1x4xf32>
}

// -----

// CHECK-LABEL: func.func @extract_splat_constant
//   CHECK-DAG:   %[[CST1:.*]] = arith.constant 1 : i32
//   CHECK-DAG:   %[[CST0:.*]] = arith.constant dense<2.000000e+00> : vector<7xf32>
//  CHECK-NEXT:   return %[[CST0]], %[[CST1]] : vector<7xf32>, i32
func.func @extract_splat_constant() -> (vector<7xf32>, i32) {
  %cst = arith.constant dense<2.000000e+00> : vector<29x7xf32>
  %cst_1 = arith.constant dense<1> : vector<4x37x9xi32>
  %0 = vector.extract %cst[2] : vector<7xf32> from vector<29x7xf32>
  %1 = vector.extract %cst_1[1, 4, 5] : i32 from vector<4x37x9xi32>
  return %0, %1 : vector<7xf32>, i32
}

// -----

// CHECK-LABEL: func.func @extract_1d_constant
//   CHECK-DAG: %[[I32CST:.*]] = arith.constant 3 : i32
//   CHECK-DAG: %[[IDXCST:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[F32CST:.*]] = arith.constant 2.000000e+00 : f32
//  CHECK-NEXT: return %[[I32CST]], %[[IDXCST]], %[[F32CST]] : i32, index, f32
func.func @extract_1d_constant() -> (i32, index, f32) {
  %icst = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
  %e = vector.extract %icst[2] : i32 from vector<4xi32>
  %idx_cst = arith.constant dense<[0, 1, 2]> : vector<3xindex>
  %f = vector.extract %idx_cst[1] : index from vector<3xindex>
  %fcst = arith.constant dense<[2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<3xf32>
  %g = vector.extract %fcst[0] : f32 from vector<3xf32>
  return %e, %f, %g : i32, index, f32
}

// -----

// CHECK-LABEL: func.func @extract_2d_constant
//   CHECK-DAG: %[[ACST:.*]] = arith.constant 0 : i32
//   CHECK-DAG: %[[BCST:.*]] = arith.constant 2 : i32
//   CHECK-DAG: %[[CCST:.*]] = arith.constant 3 : i32
//   CHECK-DAG: %[[DCST:.*]] = arith.constant 5 : i32
//  CHECK-NEXT: return %[[ACST]], %[[BCST]], %[[CCST]], %[[DCST]] : i32, i32, i32, i32
func.func @extract_2d_constant() -> (i32, i32, i32, i32) {
  %cst = arith.constant dense<[[0, 1, 2], [3, 4, 5]]> : vector<2x3xi32>
  %a = vector.extract %cst[0, 0] : i32 from vector<2x3xi32>
  %b = vector.extract %cst[0, 2] : i32 from vector<2x3xi32>
  %c = vector.extract %cst[1, 0] : i32 from vector<2x3xi32>
  %d = vector.extract %cst[1, 2] : i32 from vector<2x3xi32>
  return %a, %b, %c, %d : i32, i32, i32, i32
}

// -----

// CHECK-LABEL: func.func @extract_vector_2d_constant
//   CHECK-DAG: %[[ACST:.*]] = arith.constant dense<[0, 1, 2]> : vector<3xi32>
//   CHECK-DAG: %[[BCST:.*]] = arith.constant dense<[3, 4, 5]> : vector<3xi32>
//  CHECK-NEXT: return %[[ACST]], %[[BCST]] : vector<3xi32>, vector<3xi32>
func.func @extract_vector_2d_constant() -> (vector<3xi32>, vector<3xi32>) {
  %cst = arith.constant dense<[[0, 1, 2], [3, 4, 5]]> : vector<2x3xi32>
  %a = vector.extract %cst[0] : vector<3xi32> from vector<2x3xi32>
  %b = vector.extract %cst[1] : vector<3xi32> from vector<2x3xi32>
  return %a, %b : vector<3xi32>, vector<3xi32>
}

// -----

// CHECK-LABEL: func.func @extract_3d_constant
//   CHECK-DAG: %[[ACST:.*]] = arith.constant 0 : i32
//   CHECK-DAG: %[[BCST:.*]] = arith.constant 1 : i32
//   CHECK-DAG: %[[CCST:.*]] = arith.constant 9 : i32
//   CHECK-DAG: %[[DCST:.*]] = arith.constant 10 : i32
//  CHECK-NEXT: return %[[ACST]], %[[BCST]], %[[CCST]], %[[DCST]] : i32, i32, i32, i32
func.func @extract_3d_constant() -> (i32, i32, i32, i32) {
  %cst = arith.constant dense<[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]]> : vector<2x3x2xi32>
  %a = vector.extract %cst[0, 0, 0] : i32 from vector<2x3x2xi32>
  %b = vector.extract %cst[0, 0, 1] : i32 from vector<2x3x2xi32>
  %c = vector.extract %cst[1, 1, 1] : i32 from vector<2x3x2xi32>
  %d = vector.extract %cst[1, 2, 0] : i32 from vector<2x3x2xi32>
  return %a, %b, %c, %d : i32, i32, i32, i32
}

// -----

// CHECK-LABEL: func.func @extract_vector_3d_constant
//   CHECK-DAG: %[[ACST:.*]] = arith.constant dense<{{\[\[0, 1\], \[2, 3\], \[4, 5\]\]}}> : vector<3x2xi32>
//   CHECK-DAG: %[[BCST:.*]] = arith.constant dense<{{\[\[6, 7\], \[8, 9\], \[10, 11\]\]}}> : vector<3x2xi32>
//   CHECK-DAG: %[[CCST:.*]] = arith.constant dense<[8, 9]> : vector<2xi32>
//   CHECK-DAG: %[[DCST:.*]] = arith.constant dense<[10, 11]> : vector<2xi32>
//  CHECK-NEXT: return %[[ACST]], %[[BCST]], %[[CCST]], %[[DCST]] : vector<3x2xi32>, vector<3x2xi32>, vector<2xi32>, vector<2xi32>
func.func @extract_vector_3d_constant() -> (vector<3x2xi32>, vector<3x2xi32>, vector<2xi32>, vector<2xi32>) {
  %cst = arith.constant dense<[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]]> : vector<2x3x2xi32>
  %a = vector.extract %cst[0] : vector<3x2xi32> from vector<2x3x2xi32>
  %b = vector.extract %cst[1] : vector<3x2xi32> from vector<2x3x2xi32>
  %c = vector.extract %cst[1, 1] : vector<2xi32> from vector<2x3x2xi32>
  %d = vector.extract %cst[1, 2] : vector<2xi32> from vector<2x3x2xi32>
  return %a, %b, %c, %d : vector<3x2xi32>, vector<3x2xi32>, vector<2xi32>, vector<2xi32>
}

// -----

// CHECK-LABEL: func.func @extract_splat_vector_3d_constant
//   CHECK-DAG: %[[ACST:.*]] = arith.constant dense<0> : vector<2xi32>
//   CHECK-DAG: %[[BCST:.*]] = arith.constant dense<4> : vector<2xi32>
//   CHECK-DAG: %[[CCST:.*]] = arith.constant dense<5> : vector<2xi32>
//  CHECK-NEXT: return %[[ACST]], %[[BCST]], %[[CCST]] : vector<2xi32>, vector<2xi32>, vector<2xi32>
func.func @extract_splat_vector_3d_constant() -> (vector<2xi32>, vector<2xi32>, vector<2xi32>) {
  %cst = arith.constant dense<[[[0, 0], [1, 1], [2, 2]], [[3, 3], [4, 4], [5, 5]]]> : vector<2x3x2xi32>
  %a = vector.extract %cst[0, 0] : vector<2xi32> from vector<2x3x2xi32>
  %b = vector.extract %cst[1, 1] : vector<2xi32> from vector<2x3x2xi32>
  %c = vector.extract %cst[1, 2] : vector<2xi32> from vector<2x3x2xi32>
  return %a, %b, %c : vector<2xi32>, vector<2xi32>, vector<2xi32>
}

// -----

// CHECK-LABEL: func.func @extract_strided_slice_1d_constant
//   CHECK-DAG: %[[ACST:.*]] = arith.constant dense<[0, 1, 2]> : vector<3xi32>
//   CHECK-DAG: %[[BCST:.*]] = arith.constant dense<[1, 2]> : vector<2xi32>
//   CHECK-DAG: %[[CCST:.*]] = arith.constant dense<2> : vector<1xi32>
//  CHECK-NEXT: return %[[ACST]], %[[BCST]], %[[CCST]] : vector<3xi32>, vector<2xi32>, vector<1xi32>
func.func @extract_strided_slice_1d_constant() -> (vector<3xi32>, vector<2xi32>, vector<1xi32>) {
  %cst = arith.constant dense<[0, 1, 2]> : vector<3xi32>
  %a = vector.extract_strided_slice %cst
   {offsets = [0], sizes = [3], strides = [1]} : vector<3xi32> to vector<3xi32>
  %b = vector.extract_strided_slice %cst
   {offsets = [1], sizes = [2], strides = [1]} : vector<3xi32> to vector<2xi32>
  %c = vector.extract_strided_slice %cst
   {offsets = [2], sizes = [1], strides = [1]} : vector<3xi32> to vector<1xi32>
  return %a, %b, %c : vector<3xi32>, vector<2xi32>, vector<1xi32>
}

// -----

// CHECK-LABEL: func.func @extract_strided_slice_2d_constant
//   CHECK-DAG: %[[ACST:.*]] = arith.constant dense<0> : vector<1x1xi32>
//   CHECK-DAG: %[[BCST:.*]] = arith.constant dense<{{\[\[4, 5\]\]}}> : vector<1x2xi32>
//   CHECK-DAG: %[[CCST:.*]] = arith.constant dense<{{\[\[1, 2\], \[4, 5\]\]}}> : vector<2x2xi32>
//  CHECK-NEXT: return %[[ACST]], %[[BCST]], %[[CCST]] : vector<1x1xi32>, vector<1x2xi32>, vector<2x2xi32>
func.func @extract_strided_slice_2d_constant() -> (vector<1x1xi32>, vector<1x2xi32>, vector<2x2xi32>) {
  %cst = arith.constant dense<[[0, 1, 2], [3, 4, 5]]> : vector<2x3xi32>
  %a = vector.extract_strided_slice %cst
   {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<2x3xi32> to vector<1x1xi32>
  %b = vector.extract_strided_slice %cst
   {offsets = [1, 1], sizes = [1, 2], strides = [1, 1]} : vector<2x3xi32> to vector<1x2xi32>
  %c = vector.extract_strided_slice %cst
   {offsets = [0, 1], sizes = [2, 2], strides = [1, 1]} : vector<2x3xi32> to vector<2x2xi32>
  return %a, %b, %c : vector<1x1xi32>, vector<1x2xi32>, vector<2x2xi32>
}

// -----

// CHECK-LABEL: func.func @extract_strided_slice_3d_constant
//   CHECK-DAG: %[[ACST:.*]] = arith.constant dense<{{\[\[\[8, 9\], \[10, 11\]\]\]}}> : vector<1x2x2xi32>
//   CHECK-DAG: %[[BCST:.*]] = arith.constant dense<{{\[\[\[2, 3\]\]\]}}> : vector<1x1x2xi32>
//   CHECK-DAG: %[[CCST:.*]] = arith.constant dense<{{\[\[\[6, 7\]\], \[\[10, 11\]\]\]}}> : vector<2x1x2xi32>
//   CHECK-DAG: %[[DCST:.*]] = arith.constant dense<11> : vector<1x1x1xi32>
//  CHECK-NEXT: return %[[ACST]], %[[BCST]], %[[CCST]], %[[DCST]]
func.func @extract_strided_slice_3d_constant() -> (vector<1x2x2xi32>, vector<1x1x2xi32>, vector<2x1x2xi32>, vector<1x1x1xi32>) {
  %cst = arith.constant dense<[[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]> : vector<3x2x2xi32>
  %a = vector.extract_strided_slice %cst
   {offsets = [2], sizes = [1], strides = [1]} : vector<3x2x2xi32> to vector<1x2x2xi32>
  %b = vector.extract_strided_slice %cst
   {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]} : vector<3x2x2xi32> to vector<1x1x2xi32>
  %c = vector.extract_strided_slice %cst
   {offsets = [1, 1, 0], sizes = [2, 1, 2], strides = [1, 1, 1]} : vector<3x2x2xi32> to vector<2x1x2xi32>
  %d = vector.extract_strided_slice %cst
   {offsets = [2, 1, 1], sizes = [1, 1, 1], strides = [1, 1, 1]} : vector<3x2x2xi32> to vector<1x1x1xi32>
  return %a, %b, %c, %d : vector<1x2x2xi32>, vector<1x1x2xi32>, vector<2x1x2xi32>, vector<1x1x1xi32>
}

// -----

// CHECK-LABEL: extract_extract_strided
//  CHECK-SAME: %[[A:.*]]: vector<32x16x4xf16>
//       CHECK: %[[V:.*]] = vector.extract %[[A]][9, 7] : vector<4xf16> from vector<32x16x4xf16>
//       CHECK: return %[[V]] : vector<4xf16>
func.func @extract_extract_strided(%arg0: vector<32x16x4xf16>) -> vector<4xf16> {
 %1 = vector.extract_strided_slice %arg0
  {offsets = [7, 3], sizes = [10, 8], strides = [1, 1]} :
  vector<32x16x4xf16> to vector<10x8x4xf16>
  %2 = vector.extract %1[2, 4] : vector<4xf16> from vector<10x8x4xf16>
  return %2 : vector<4xf16>
}

// -----

// CHECK-LABEL: extract_insert_strided
//  CHECK-SAME: %[[A:.*]]: vector<6x4xf32>
//       CHECK: %[[V:.*]] = vector.extract %[[A]][0, 2] : f32 from vector<6x4xf32>
//       CHECK: return %[[V]] : f32
func.func @extract_insert_strided(%a: vector<6x4xf32>, %b: vector<8x16xf32>)
  -> f32 {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1, 1]}
    : vector<6x4xf32> into vector<8x16xf32>
  %2 = vector.extract %0[2, 4] : f32 from vector<8x16xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: extract_insert_rank_reduce
//  CHECK-SAME: %[[A:.*]]: vector<4xf32>
//       CHECK: %[[V:.*]] = vector.extract %[[A]][2] : f32 from vector<4xf32>
//       CHECK: return %[[V]] : f32
func.func @extract_insert_rank_reduce(%a: vector<4xf32>, %b: vector<8x16xf32>)
  -> f32 {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1]}
    : vector<4xf32> into vector<8x16xf32>
  %2 = vector.extract %0[2, 4] : f32 from vector<8x16xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: negative_extract_insert
//       CHECK: vector.insert_strided_slice
//       CHECK: vector.extract
func.func @negative_extract_insert(%a: vector<2x15xf32>, %b: vector<12x8x16xf32>)
  -> vector<16xf32> {
  %0 = vector.insert_strided_slice %a, %b {offsets = [4, 2, 0], strides = [1, 1]}
    : vector<2x15xf32> into vector<12x8x16xf32>
  %2 = vector.extract %0[4, 2] : vector<16xf32> from vector<12x8x16xf32>
  return %2 : vector<16xf32>
}

// -----

// CHECK-LABEL: extract_insert_chain
//  CHECK-SAME: (%[[A:.*]]: vector<2x16xf32>, %[[B:.*]]: vector<12x8x16xf32>, %[[C:.*]]: vector<2x16xf32>)
//       CHECK: %[[V:.*]] = vector.extract %[[C]][0] : vector<16xf32> from vector<2x16xf32>
//       CHECK: return %[[V]] : vector<16xf32>
func.func @extract_insert_chain(%a: vector<2x16xf32>, %b: vector<12x8x16xf32>, %c: vector<2x16xf32>)
  -> vector<16xf32> {
  %0 = vector.insert_strided_slice %c, %b {offsets = [4, 2, 0], strides = [1, 1]}
    : vector<2x16xf32> into vector<12x8x16xf32>
  %1 = vector.insert_strided_slice %a, %0 {offsets = [0, 2, 0], strides = [1, 1]}
    : vector<2x16xf32> into vector<12x8x16xf32>
  %2 = vector.extract %1[4, 2] : vector<16xf32> from vector<12x8x16xf32>
  return %2 : vector<16xf32>
}

// -----

// CHECK-LABEL: extract_from_extract_chain_should_not_fold_dynamic_extracts
//  CHECK-SAME: (%[[VEC:.*]]: vector<2x4xf32>, %[[IDX:.*]]: index)
//       CHECK: %[[A:.*]] = vector.extract %[[VEC]][%[[IDX]]] : vector<4xf32> from vector<2x4xf32>
//       CHECK: %[[B:.*]] = vector.extract %[[A]][1] : f32 from vector<4xf32>
func.func @extract_from_extract_chain_should_not_fold_dynamic_extracts(%v: vector<2x4xf32>, %index: index) -> f32 {
  %0 = vector.extract %v[%index] : vector<4xf32> from vector<2x4xf32>
  %1 = vector.extract %0[1] : f32 from vector<4xf32>
  return %1 : f32
}

// -----

// CHECK-LABEL: extract_extract_strided2
//  CHECK-SAME: %[[A:.*]]: vector<2x4xf32>
//       CHECK: %[[V:.*]] = vector.extract %[[A]][1] : vector<4xf32> from vector<2x4xf32>
//       CHECK: return %[[V]] : vector<4xf32>
func.func @extract_extract_strided2(%A: vector<2x4xf32>)
  -> (vector<4xf32>) {
 %0 = vector.extract_strided_slice %A {offsets = [1, 0], sizes = [1, 4], strides = [1, 1]} : vector<2x4xf32> to vector<1x4xf32>
 %1 = vector.extract %0[0] : vector<4xf32> from vector<1x4xf32>
 return %1 : vector<4xf32>
}

// -----

// CHECK-LABEL: func @splatlike_fold
//  CHECK-NEXT: [[V:%.*]] = arith.constant dense<1.000000e+00> : vector<4xf32>
//  CHECK-NEXT: return [[V]] : vector<4xf32>
func.func @splatlike_fold() -> vector<4xf32> {
  %c = arith.constant 1.0 : f32
  %v = vector.broadcast %c : f32 to vector<4xf32>
  return %v : vector<4xf32>

}

// -----

// CHECK-LABEL: func @shuffle_1d
//       CHECK:   %[[V:.+]] = arith.constant dense<[3, 2, 5, 1]> : vector<4xi32>
//       CHECK:   return %[[V]]
func.func @shuffle_1d() -> vector<4xi32> {
  %v0 = arith.constant dense<[0, 1, 2]> : vector<3xi32>
  %v1 = arith.constant dense<[3, 4, 5]> : vector<3xi32>
  %shuffle = vector.shuffle %v0, %v1 [3, 2, 5, 1] : vector<3xi32>, vector<3xi32>
  return %shuffle : vector<4xi32>
}

// -----

// Ensure shuffle dense resource elements not crash.

// CHECK-LABEL: func.func @shuffle_1d_dense_resource
//       CHECK:    vector.shuffle
func.func @shuffle_1d_dense_resource() -> vector<4xi32> {
  %v0 = arith.constant dense_resource<__elided__> : vector<3xi32>
  %v1 = arith.constant dense_resource<__elided__> : vector<3xi32>
  %shuffle = vector.shuffle %v0, %v1 [3, 2, 5, 1] : vector<3xi32>, vector<3xi32>
  return %shuffle : vector<4xi32>
}

// -----

// Check that poison indices pick the first element of the first non-poison
// input vector. That is, %v[0] (i.e., 5) in this test.

// CHECK-LABEL: func @shuffle_1d_poison_idx
//       CHECK:   %[[V:.+]] = arith.constant dense<[13, 10, 15, 10]> : vector<4xi32>
//       CHECK:   return %[[V]]
func.func @shuffle_1d_poison_idx() -> vector<4xi32> {
  %v0 = arith.constant dense<[10, 11, 12]> : vector<3xi32>
  %v1 = arith.constant dense<[13, 14, 15]> : vector<3xi32>
  %shuffle = vector.shuffle %v0, %v1 [3, -1, 5, -1] : vector<3xi32>, vector<3xi32>
  return %shuffle : vector<4xi32>
}

// -----

// CHECK-LABEL: func @shuffle_1d_rhs_lhs_poison
//   CHECK-NOT:   vector.shuffle
//       CHECK:   %[[V:.+]] = ub.poison : vector<4xi32>
//       CHECK:   return %[[V]]
func.func @shuffle_1d_rhs_lhs_poison() -> vector<4xi32> {
  %v0 = ub.poison : vector<3xi32>
  %v1 = ub.poison : vector<3xi32>
  %shuffle = vector.shuffle %v0, %v1 [3, 1, 5, 4] : vector<3xi32>, vector<3xi32>
  return %shuffle : vector<4xi32>
}

// -----

// CHECK-LABEL: func @shuffle_1d_lhs_poison
//   CHECK-NOT:   vector.shuffle
//       CHECK:   %[[V:.+]] = arith.constant dense<[11, 12, 11, 11]> : vector<4xi32>
//       CHECK:   return %[[V]]
func.func @shuffle_1d_lhs_poison() -> vector<4xi32> {
  %v0 = arith.constant dense<[11, 12, 13]> : vector<3xi32>
  %v1 = ub.poison : vector<3xi32>
  %shuffle = vector.shuffle %v0, %v1 [3, 1, 5, 4] : vector<3xi32>, vector<3xi32>
  return %shuffle : vector<4xi32>
}

// -----

// CHECK-LABEL: func @shuffle_1d_rhs_poison
//   CHECK-NOT:   vector.shuffle
//       CHECK:   %[[V:.+]] = arith.constant dense<[11, 11, 13, 12]> : vector<4xi32>
//       CHECK:   return %[[V]]
func.func @shuffle_1d_rhs_poison() -> vector<4xi32> {
  %v0 = ub.poison : vector<3xi32>
  %v1 = arith.constant dense<[11, 12, 13]> : vector<3xi32>
  %shuffle = vector.shuffle %v0, %v1 [3, 1, 5, 4] : vector<3xi32>, vector<3xi32>
  return %shuffle : vector<4xi32>
}

// -----

// CHECK-LABEL: func @shuffle_canonicalize_0d
func.func @shuffle_canonicalize_0d(%v0 : vector<i32>, %v1 : vector<i32>) -> vector<1xi32> {
  // CHECK: vector.broadcast %{{.*}} : vector<i32> to vector<1xi32>
  %shuffle = vector.shuffle %v0, %v1 [0] : vector<i32>, vector<i32>
  return %shuffle : vector<1xi32>
}

// -----

// CHECK-LABEL: func @shuffle_fold1
//       CHECK:   %arg0 : vector<4xi32>
func.func @shuffle_fold1(%v0 : vector<4xi32>, %v1 : vector<2xi32>) -> vector<4xi32> {
  %shuffle = vector.shuffle %v0, %v1 [0, 1, 2, 3] : vector<4xi32>, vector<2xi32>
  return %shuffle : vector<4xi32>
}

// -----

// CHECK-LABEL: func @shuffle_fold2
//       CHECK:   %arg1 : vector<2xi32>
func.func @shuffle_fold2(%v0 : vector<4xi32>, %v1 : vector<2xi32>) -> vector<2xi32> {
  %shuffle = vector.shuffle %v0, %v1 [4, 5] : vector<4xi32>, vector<2xi32>
  return %shuffle : vector<2xi32>
}

// -----

// CHECK-LABEL: func @shuffle_fold3
//       CHECK:   return %arg0 : vector<4x5x6xi32>
func.func @shuffle_fold3(%v0 : vector<4x5x6xi32>, %v1 : vector<2x5x6xi32>) -> vector<4x5x6xi32> {
  %shuffle = vector.shuffle %v0, %v1 [0, 1, 2, 3] : vector<4x5x6xi32>, vector<2x5x6xi32>
  return %shuffle : vector<4x5x6xi32>
}

// -----

// CHECK-LABEL: func @shuffle_fold4
//       CHECK:   return %arg1 : vector<2x5x6xi32>
func.func @shuffle_fold4(%v0 : vector<4x5x6xi32>, %v1 : vector<2x5x6xi32>) -> vector<2x5x6xi32> {
  %shuffle = vector.shuffle %v0, %v1 [4, 5] : vector<4x5x6xi32>, vector<2x5x6xi32>
  return %shuffle : vector<2x5x6xi32>
}

// -----

// CHECK-LABEL: func @shuffle_nofold1
//       CHECK:   %[[V:.+]] = vector.shuffle %arg0, %arg1 [0, 1, 2, 3, 4] : vector<4xi32>, vector<2xi32>
//       CHECK:   return %[[V]]
func.func @shuffle_nofold1(%v0 : vector<4xi32>, %v1 : vector<2xi32>) -> vector<5xi32> {
  %shuffle = vector.shuffle %v0, %v1 [0, 1, 2, 3, 4] : vector<4xi32>, vector<2xi32>
  return %shuffle : vector<5xi32>
}

// -----

// CHECK-LABEL: func @transpose_splatlike_constant
//       CHECK:   %[[CST:.+]] = arith.constant dense<5.000000e+00> : vector<8x4xf32>
//       CHECK:   return %[[CST]]
func.func @transpose_splatlike_constant() -> vector<8x4xf32> {
  %cst = arith.constant dense<5.0> : vector<4x8xf32>
  %0 = vector.transpose %cst, [1, 0] : vector<4x8xf32> to vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// CHECK-LABEL:   func @transpose_splatlike2(
//  CHECK-SAME:     %[[VAL_0:.*]]: f32) -> vector<3x4xf32> {
//       CHECK:     %[[VAL_1:.*]] = vector.broadcast %[[VAL_0]] : f32 to vector<3x4xf32>
//       CHECK:     return %[[VAL_1]] : vector<3x4xf32>
//       CHECK:     }
func.func @transpose_splatlike2(%arg : f32) -> vector<3x4xf32> {
  %splat = vector.broadcast %arg : f32 to vector<4x3xf32>
  %0 = vector.transpose %splat, [1, 0] : vector<4x3xf32> to vector<3x4xf32>
  return %0 : vector<3x4xf32>
}

// -----

// CHECK-LABEL: transpose_poison
//       CHECK:  %[[POISON:.*]] = ub.poison : vector<4x6xi8>
//       CHECK:  return %[[POISON]] : vector<4x6xi8>
func.func @transpose_poison() -> vector<4x6xi8> {
  %poison = ub.poison : vector<6x4xi8>
  %transpose = vector.transpose %poison, [1, 0] : vector<6x4xi8> to vector<4x6xi8>
  return %transpose : vector<4x6xi8>
}

// -----

// CHECK-LABEL: func.func @insert_1d_constant
//   CHECK-DAG: %[[ACST:.*]] = arith.constant dense<[9, 1, 2]> : vector<3xi32>
//   CHECK-DAG: %[[BCST:.*]] = arith.constant dense<[0, 9, 2]> : vector<3xi32>
//   CHECK-DAG: %[[CCST:.*]] = arith.constant dense<[0, 1, 9]> : vector<3xi32>
//  CHECK-NEXT: return %[[ACST]], %[[BCST]], %[[CCST]] : vector<3xi32>, vector<3xi32>, vector<3xi32>
func.func @insert_1d_constant() -> (vector<3xi32>, vector<3xi32>, vector<3xi32>) {
  %vcst = arith.constant dense<[0, 1, 2]> : vector<3xi32>
  %icst = arith.constant 9 : i32
  %a = vector.insert %icst, %vcst[0] : i32 into vector<3xi32>
  %b = vector.insert %icst, %vcst[1] : i32 into vector<3xi32>
  %c = vector.insert %icst, %vcst[2] : i32 into vector<3xi32>
  return %a, %b, %c : vector<3xi32>, vector<3xi32>, vector<3xi32>
}

// -----

// CHECK-LABEL: func.func @insert_2d_constant
//   CHECK-DAG: %[[ACST:.*]] = arith.constant dense<{{\[\[99, 1, 2\], \[3, 4, 5\]\]}}> : vector<2x3xi32>
//   CHECK-DAG: %[[BCST:.*]] = arith.constant dense<{{\[\[0, 1, 2\], \[3, 4, 99\]\]}}> : vector<2x3xi32>
//   CHECK-DAG: %[[CCST:.*]] = arith.constant dense<{{\[\[90, 91, 92\], \[3, 4, 5\]\]}}> : vector<2x3xi32>
//   CHECK-DAG: %[[DCST:.*]] = arith.constant dense<{{\[\[0, 1, 2\], \[90, 91, 92\]\]}}> : vector<2x3xi32>
//  CHECK-NEXT: return %[[ACST]], %[[BCST]], %[[CCST]], %[[DCST]]
func.func @insert_2d_constant() -> (vector<2x3xi32>, vector<2x3xi32>, vector<2x3xi32>, vector<2x3xi32>) {
  %vcst = arith.constant dense<[[0, 1, 2], [3, 4, 5]]> : vector<2x3xi32>
  %cst_scalar = arith.constant 99 : i32
  %cst_1d = arith.constant dense<[90, 91, 92]> : vector<3xi32>
  %a = vector.insert %cst_scalar, %vcst[0, 0] : i32 into vector<2x3xi32>
  %b = vector.insert %cst_scalar, %vcst[1, 2] : i32 into vector<2x3xi32>
  %c = vector.insert %cst_1d, %vcst[0] : vector<3xi32> into vector<2x3xi32>
  %d = vector.insert %cst_1d, %vcst[1] : vector<3xi32> into vector<2x3xi32>
  return %a, %b, %c, %d : vector<2x3xi32>, vector<2x3xi32>, vector<2x3xi32>, vector<2x3xi32>
}

// -----

// +---------------------------------------------------------------------------
// Tests for InsertChainFullyInitialized .
// +---------------------------------------------------------------------------
// This pattern should fire when all vector elements are overwritten by vector.insert
// at static positions, replacing the insert chain with vector.from_elements.
// CHECK-LABEL: func.func @fully_insert_scalar_to_vector(
//  CHECK-SAME: %[[DEST:.+]]: vector<2xi64>, %[[SRC1:.+]]: i64, %[[SRC2:.+]]: i64)
//       CHECK: %[[RES:.+]] = vector.from_elements %[[SRC1]], %[[SRC2]] : vector<2xi64>
//  CHECK-NEXT: return %[[RES]]
func.func @fully_insert_scalar_to_vector(%dest : vector<2xi64>, %src1 : i64, %src2 : i64) -> vector<2xi64> {
  %v1 = vector.insert %src1, %dest[0] : i64 into vector<2xi64>
  %v2 = vector.insert %src2, %v1[1] : i64 into vector<2xi64>
  return %v2 : vector<2xi64>
}

// -----

// Same as the above test, but with vector insertions.
// CHECK-LABEL: func.func @fully_insert_vector_to_vector(
//  CHECK-SAME: %[[DEST:.+]]: vector<2x2xi64>, %[[SRC1:.+]]: vector<2xi64>, %[[SRC2:.+]]: vector<2xi64>)
//       CHECK: %[[ELE1:.+]]:2 = vector.to_elements %[[SRC1]] : vector<2xi64>
//       CHECK: %[[ELE2:.+]]:2 = vector.to_elements %[[SRC2]] : vector<2xi64>
//       CHECK: %[[RES:.+]] = vector.from_elements %[[ELE1]]#0, %[[ELE1]]#1, %[[ELE2]]#0, %[[ELE2]]#1 : vector<2x2xi64>
//  CHECK-NEXT: return %[[RES]]
func.func @fully_insert_vector_to_vector(%dest : vector<2x2xi64>, %src1 : vector<2xi64>, %src2 : vector<2xi64>) -> vector<2x2xi64> {
  %v1 = vector.insert %src1, %dest[0] : vector<2xi64> into vector<2x2xi64>
  %v2 = vector.insert %src2, %v1[1] : vector<2xi64> into vector<2x2xi64>
  return %v2 : vector<2x2xi64>
}

// -----

// Test InsertChainFullyInitialized pattern with overlapping insertions.
// 1. The first op inserts %src2 at [0,0].
// 2. The second op inserts %src1 at [0,0], [0,1], overwriting %src2 at [0,0].
// 3. The third op inserts %src1 at [1,0], [1,1].
// 4. The fourth op inserts %src2 at [1,1], overwriting the existing value at [1,1].
// CHECK-LABEL: func.func @fully_insert_to_vector_overlap_1(
//  CHECK-SAME: %[[DEST:.+]]: vector<2x2xi64>, %[[SRC1:.+]]: vector<2xi64>, %[[SRC2:.+]]: i64)
//       CHECK: %[[ELE1:.+]]:2 = vector.to_elements %[[SRC1]] : vector<2xi64>
//       CHECK: %[[ELE2:.+]]:2 = vector.to_elements %[[SRC1]] : vector<2xi64>
//       CHECK: %[[RES:.+]] = vector.from_elements %[[ELE1]]#0, %[[ELE1]]#1, %[[ELE2]]#0, %[[SRC2]] : vector<2x2xi64>
//  CHECK-NEXT: return %[[RES]]
func.func @fully_insert_to_vector_overlap_1(%dest : vector<2x2xi64>, %src1 : vector<2xi64>, %src2 : i64) -> vector<2x2xi64> {
  %v0 = vector.insert %src2, %dest[0, 0] : i64 into vector<2x2xi64>
  %v1 = vector.insert %src1, %v0[0] : vector<2xi64> into vector<2x2xi64>
  %v2 = vector.insert %src1, %v1[1] : vector<2xi64> into vector<2x2xi64>
  %v3 = vector.insert %src2, %v2[1, 1] : i64 into vector<2x2xi64>
  return %v3 : vector<2x2xi64>
}

// -----

// Test InsertChainFullyInitialized pattern with overlapping insertions.
// The vector inserted at last should overwrite the previously inserted scalars.
// CHECK-LABEL: func.func @fully_insert_to_vector_overlap_2(
//  CHECK-SAME: %[[DEST:.+]]: vector<2x2xi64>, %[[SRC1:.+]]: i64, %[[SRC2:.+]]: i64, %[[SRC3:.+]]: vector<2xi64>, %[[SRC4:.+]]: vector<2xi64>)
//       CHECK: %[[ELE1:.+]]:2 = vector.to_elements %[[SRC3]] : vector<2xi64>
//       CHECK: %[[ELE2:.+]]:2 = vector.to_elements %[[SRC4]] : vector<2xi64>
//       CHECK: %[[RES:.+]] = vector.from_elements %[[ELE1]]#0, %[[ELE1]]#1, %[[ELE2]]#0, %[[ELE2]]#1 : vector<2x2xi64>
//  CHECK-NEXT: return %[[RES]]
func.func @fully_insert_to_vector_overlap_2(%dest : vector<2x2xi64>, %src1 : i64, %src2 : i64, %src3 : vector<2xi64>, %src4 : vector<2xi64>) -> vector<2x2xi64> {
  %v0 = vector.insert %src1, %dest[0, 0] : i64 into vector<2x2xi64>
  %v1 = vector.insert %src2, %v0[0, 1] : i64 into vector<2x2xi64>
  %v2 = vector.insert %src3, %v1[0] : vector<2xi64> into vector<2x2xi64>
  %v3 = vector.insert %src4, %v2[1] : vector<2xi64> into vector<2x2xi64>
  return %v3 : vector<2x2xi64>
}

// -----

// Negative test for InsertChainFullyInitialized pattern when only some elements are overwritten.
// CHECK-LABEL: func.func @negative_partially_insert_vector_to_vector(
//  CHECK-SAME: %[[DEST:.+]]: vector<2x2xi64>, %[[SRC1:.+]]: vector<2xi64>, %[[SRC2:.+]]: i64)
//       CHECK: %[[V0:.+]] = vector.insert %[[SRC1]], %[[DEST]] [0] : vector<2xi64> into vector<2x2xi64>
//       CHECK: %[[V1:.+]] = vector.insert %[[SRC2]], %[[V0]] [0, 0] : i64 into vector<2x2xi64>
//       CHECK: return %[[V1]]
func.func @negative_partially_insert_vector_to_vector(%dest : vector<2x2xi64>, %src1 : vector<2xi64>, %src2 : i64) -> vector<2x2xi64> {
  %v1 = vector.insert %src1, %dest[0] : vector<2xi64> into vector<2x2xi64>
  %v2 = vector.insert %src2, %v1[0, 0] : i64 into vector<2x2xi64>
  return %v2 : vector<2x2xi64>
}

// -----

// Negative test when intermediate results have more than one user.
// CHECK-LABEL: func.func @negative_intermediate_insert_multiple_users(
//  CHECK-SAME: %[[DEST:.+]]: vector<3xi64>, %[[SRC1:.+]]: i64, %[[SRC2:.+]]: i64, %[[SRC3:.+]]: i64, %[[SRC4:.+]]: i64)
//       CHECK: %[[V0:.+]] = vector.insert %[[SRC1]], %[[DEST]] [0] : i64 into vector<3xi64>
//       CHECK: %[[V1:.+]] = vector.insert %[[SRC2]], %[[V0]] [1] : i64 into vector<3xi64>
//       CHECK: %[[V2:.+]] = vector.insert %[[SRC3]], %[[V1]] [2] : i64 into vector<3xi64>
//       CHECK: %[[V3:.+]] = vector.insert %[[SRC4]], %[[V1]] [2] : i64 into vector<3xi64>
func.func @negative_intermediate_insert_multiple_users(%dest : vector<3xi64>, %src1 : i64, %src2 : i64, %src3 : i64, %src4 : i64) -> (vector<3xi64>, vector<3xi64>) {
  %v1 = vector.insert %src1, %dest[0] : i64 into vector<3xi64>
  %v2 = vector.insert %src2, %v1[1] : i64 into vector<3xi64>
  %v3_0 = vector.insert %src3, %v2[2] : i64 into vector<3xi64>
  %v3_1 = vector.insert %src4, %v2[2] : i64 into vector<3xi64>
  return %v3_0, %v3_1 : vector<3xi64>, vector<3xi64>
}

// +---------------------------------------------------------------------------
// End of  Tests For InsertChainFullyInitialized.
// +---------------------------------------------------------------------------

// -----

// CHECK-LABEL: func.func @insert_2d_splat_constant
//   CHECK-DAG: %[[ACST:.*]] = arith.constant dense<0> : vector<2x3xi32>
//   CHECK-DAG: %[[BCST:.*]] = arith.constant dense<{{\[\[99, 0, 0\], \[0, 0, 0\]\]}}> : vector<2x3xi32>
//   CHECK-DAG: %[[CCST:.*]] = arith.constant dense<{{\[\[0, 0, 0\], \[0, 99, 0\]\]}}> : vector<2x3xi32>
//   CHECK-DAG: %[[DCST:.*]] = arith.constant dense<{{\[\[33, 33, 33\], \[0, 0, 0\]\]}}> : vector<2x3xi32>
//   CHECK-DAG: %[[ECST:.*]] = arith.constant dense<{{\[\[0, 0, 0\], \[33, 33, 33\]\]}}> : vector<2x3xi32>
//  CHECK-NEXT: return %[[ACST]], %[[BCST]], %[[CCST]], %[[DCST]], %[[ECST]]
func.func @insert_2d_splat_constant()
  -> (vector<2x3xi32>, vector<2x3xi32>, vector<2x3xi32>, vector<2x3xi32>, vector<2x3xi32>) {
  %vcst = arith.constant dense<0> : vector<2x3xi32>
  %cst_zero = arith.constant 0 : i32
  %cst_scalar = arith.constant 99 : i32
  %cst_1d = arith.constant dense<33> : vector<3xi32>
  %a = vector.insert %cst_zero, %vcst[0, 0] : i32 into vector<2x3xi32>
  %b = vector.insert %cst_scalar, %vcst[0, 0] : i32 into vector<2x3xi32>
  %c = vector.insert %cst_scalar, %vcst[1, 1] : i32 into vector<2x3xi32>
  %d = vector.insert %cst_1d, %vcst[0] : vector<3xi32> into vector<2x3xi32>
  %e = vector.insert %cst_1d, %vcst[1] : vector<3xi32> into vector<2x3xi32>
  return %a, %b, %c, %d, %e : vector<2x3xi32>, vector<2x3xi32>, vector<2x3xi32>, vector<2x3xi32>, vector<2x3xi32>
}

// -----

// CHECK-LABEL: func @reduce_one_element_vector_extract
//  CHECK-SAME: (%[[V:.+]]: vector<1xf32>)
//       CHECK:   %[[S:.+]] = vector.extract %[[V]][0] : f32 from vector<1xf32>
//       CHECK:   return %[[S]] : f32
func.func @reduce_one_element_vector_extract(%a : vector<1xf32>) -> f32 {
  %s = vector.reduction <add>, %a : vector<1xf32> into f32
  return %s : f32
}

// -----

// CHECK-LABEL: func @masked_reduce_one_element_vector_extract
//  CHECK-SAME: %[[VAL_0:.*]]: vector<1xf32>, %[[VAL_1:.*]]: vector<1xi1>)
func.func @masked_reduce_one_element_vector_extract(%a : vector<1xf32>, %mask : vector<1xi1>) -> f32 {
//       CHECK:   %[[VAL_2:.*]] = vector.extract %[[VAL_0]][0] : f32 from vector<1xf32>
  %s = vector.mask %mask { vector.reduction <add>, %a : vector<1xf32> into f32 }
         : vector<1xi1> -> f32
  return %s : f32
}

// -----

// CHECK-LABEL: func @reduce_one_element_vector_addf
//  CHECK-SAME: (%[[V:.+]]: vector<1xf32>, %[[B:.+]]: f32)
//       CHECK:   %[[A:.+]] = vector.extract %[[V]][0] : f32 from vector<1xf32>
//       CHECK:   %[[S:.+]] = arith.addf %[[A]], %arg1 : f32
//       CHECK:   return %[[S]]
func.func @reduce_one_element_vector_addf(%a : vector<1xf32>, %b: f32) -> f32 {
  %s = vector.reduction <add>, %a, %b : vector<1xf32> into f32
  return %s : f32
}

// -----

// CHECK-LABEL: func @reduce_one_element_vector_addf_fastmath
//  CHECK-SAME: (%[[V:.+]]: vector<1xf32>, %[[B:.+]]: f32)
//       CHECK:   %[[A:.+]] = vector.extract %[[V]][0] : f32 from vector<1xf32>
//       CHECK:   %[[S:.+]] = arith.addf %[[A]], %arg1 fastmath<nnan,ninf> : f32
//       CHECK:   return %[[S]]
func.func @reduce_one_element_vector_addf_fastmath(%a : vector<1xf32>, %b: f32) -> f32 {
  %s = vector.reduction <add>, %a, %b fastmath<nnan,ninf> : vector<1xf32> into f32
  return %s : f32
}

// -----

// CHECK-LABEL: func @masked_reduce_one_element_vector_addf
//  CHECK-SAME: %[[VAL_0:.*]]: vector<1xf32>, %[[VAL_1:.*]]: f32,
//  CHECK-SAME: %[[VAL_2:.*]]: vector<1xi1>)
func.func @masked_reduce_one_element_vector_addf(%a: vector<1xf32>,
                                                 %b: f32,
                                                 %mask: vector<1xi1>) -> f32 {
//       CHECK:   %[[VAL_3:.*]] = vector.extract %[[VAL_2]][0] : i1 from vector<1xi1>
//       CHECK:   %[[VAL_4:.*]] = vector.extract %[[VAL_0]][0] : f32 from vector<1xf32>
//       CHECK:   %[[VAL_5:.*]] = arith.addf %[[VAL_4]], %[[VAL_1]] : f32
//       CHECK:   %[[VAL_6:.*]] = arith.select %[[VAL_3]], %[[VAL_5]], %[[VAL_1]] : f32
  %s = vector.mask %mask { vector.reduction <add>, %a, %b : vector<1xf32> into f32 }
         : vector<1xi1> -> f32
  return %s : f32
}

// -----

// CHECK-LABEL: func @reduce_one_element_vector_mulf
//  CHECK-SAME: (%[[V:.+]]: vector<1xf32>, %[[B:.+]]: f32)
//       CHECK:   %[[A:.+]] = vector.extract %[[V]][0] : f32 from vector<1xf32>
//       CHECK:   %[[S:.+]] = arith.mulf %[[A]], %arg1 : f32
//       CHECK:   return %[[S]]
func.func @reduce_one_element_vector_mulf(%a : vector<1xf32>, %b: f32) -> f32 {
  %s = vector.reduction <mul>, %a, %b : vector<1xf32> into f32
  return %s : f32
}

// -----

// CHECK-LABEL: func @dont_reduce_one_element_vector
//       CHECK: vector.reduction
func.func @dont_reduce_one_element_vector(%a : vector<4xf32>) -> f32 {
  %s = vector.reduction <add>, %a : vector<4xf32> into f32
  return %s : f32
}

// -----

// CHECK-LABEL: func @reduce_one_element_vector_maximumf
//  CHECK-SAME: (%[[V:.+]]: vector<1xf32>, %[[B:.+]]: f32)
//       CHECK:   %[[A:.+]] = vector.extract %[[V]][0] : f32 from vector<1xf32>
//       CHECK:   %[[S:.+]] = arith.maximumf %[[A]], %[[B]] : f32
//       CHECK:   return %[[S]]
func.func @reduce_one_element_vector_maximumf(%a : vector<1xf32>, %b: f32) -> f32 {
  %s = vector.reduction <maximumf>, %a, %b : vector<1xf32> into f32
  return %s : f32
}

// -----

// CHECK-LABEL: func @bitcast(
//  CHECK-SAME:               %[[ARG:.*]]: vector<4x8xf32>) -> vector<4x16xi16> {
//       CHECK: vector.bitcast %[[ARG:.*]] : vector<4x8xf32> to vector<4x16xi16>
func.func @bitcast(%a: vector<4x8xf32>) -> vector<4x16xi16> {
  %0 = vector.bitcast %a : vector<4x8xf32> to vector<4x8xi32>
  %1 = vector.bitcast %0 : vector<4x8xi32> to vector<4x16xi16>
  return %1 : vector<4x16xi16>
}

// -----

// CHECK-LABEL: @insert_strided_slice_splatlike
//  CHECK-SAME: (%[[ARG:.*]]: f32)
//  CHECK-NEXT:   %[[SPLAT:.*]] = vector.broadcast %[[ARG]] : f32 to vector<8x16xf32>
//  CHECK-NEXT:   return %[[SPLAT]] : vector<8x16xf32>
func.func @insert_strided_slice_splatlike(%x: f32) -> (vector<8x16xf32>) {
  %splat0 = vector.broadcast %x : f32 to vector<4x4xf32>
  %splat1 = vector.broadcast %x : f32 to vector<8x16xf32>
  %0 = vector.insert_strided_slice %splat0, %splat1 {offsets = [2, 2], strides = [1, 1]}
    : vector<4x4xf32> into vector<8x16xf32>
  return %0 : vector<8x16xf32>
}


// -----

// CHECK-LABEL: @insert_extract_strided_slice
//  CHECK-SAME: (%[[ARG:.*]]: vector<8x16xf32>)
//  CHECK-NEXT:   return %[[ARG]] : vector<8x16xf32>
func.func @insert_extract_strided_slice(%x: vector<8x16xf32>) -> (vector<8x16xf32>) {
  %0 = vector.extract_strided_slice %x {offsets = [0, 8], sizes = [2, 4], strides = [1, 1]}
        : vector<8x16xf32> to vector<2x4xf32>
  %1 = vector.insert_strided_slice %0, %x {offsets = [0, 8], strides = [1, 1]}
        : vector<2x4xf32> into vector<8x16xf32>
  return %1 : vector<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @insert_strided_1d_constant
//   CHECK-DAG: %[[ACST:.*]] = arith.constant dense<[4, 1, 2]> : vector<3xi32>
//   CHECK-DAG: %[[BCST:.*]] = arith.constant dense<[0, 1, 4]> : vector<3xi32>
//   CHECK-DAG: %[[CCST:.*]] = arith.constant dense<[5, 6, 2]> : vector<3xi32>
//   CHECK-DAG: %[[DCST:.*]] = arith.constant dense<[0, 5, 6]> : vector<3xi32>
//   CHECK-DAG: %[[ECST:.*]] = arith.constant dense<[7, 8, 9]> : vector<3xi32>
//  CHECK-NEXT: return %[[ACST]], %[[BCST]], %[[CCST]], %[[DCST]], %[[ECST]]
func.func @insert_strided_1d_constant() ->
  (vector<3xi32>, vector<3xi32>, vector<3xi32>, vector<3xi32>, vector<3xi32>) {
  %vcst = arith.constant dense<[0, 1, 2]> : vector<3xi32>
  %cst_1 = arith.constant dense<4> : vector<1xi32>
  %cst_2 = arith.constant dense<[5, 6]> : vector<2xi32>
  %cst_3 = arith.constant dense<[7, 8, 9]> : vector<3xi32>
  %a = vector.insert_strided_slice %cst_1, %vcst {offsets = [0], strides = [1]} : vector<1xi32> into vector<3xi32>
  %b = vector.insert_strided_slice %cst_1, %vcst {offsets = [2], strides = [1]} : vector<1xi32> into vector<3xi32>
  %c = vector.insert_strided_slice %cst_2, %vcst {offsets = [0], strides = [1]} : vector<2xi32> into vector<3xi32>
  %d = vector.insert_strided_slice %cst_2, %vcst {offsets = [1], strides = [1]} : vector<2xi32> into vector<3xi32>
  %e = vector.insert_strided_slice %cst_3, %vcst {offsets = [0], strides = [1]} : vector<3xi32> into vector<3xi32>
  return %a, %b, %c, %d, %e : vector<3xi32>, vector<3xi32>, vector<3xi32>, vector<3xi32>, vector<3xi32>
}

// -----

// CHECK-LABEL: func.func @insert_strided_2d_constant
//   CHECK-DAG: %[[ACST:.*]] = arith.constant dense<{{\[\[0, 1\], \[9, 3\], \[4, 5\]\]}}> : vector<3x2xi32>
//   CHECK-DAG: %[[BCST:.*]] = arith.constant dense<{{\[\[0, 1\], \[2, 3\], \[4, 9\]\]}}> : vector<3x2xi32>
//   CHECK-DAG: %[[CCST:.*]] = arith.constant dense<{{\[\[18, 19\], \[2, 3\], \[4, 5\]\]}}> : vector<3x2xi32>
//   CHECK-DAG: %[[DCST:.*]] = arith.constant dense<{{\[\[0, 1\], \[18, 19\], \[4, 5\]\]}}> : vector<3x2xi32>
//   CHECK-DAG: %[[ECST:.*]] = arith.constant dense<{{\[\[0, 1\], \[2, 3\], \[18, 19\]\]}}> : vector<3x2xi32>
//   CHECK-DAG: %[[FCST:.*]] = arith.constant dense<{{\[\[28, 29\], \[38, 39\], \[4, 5\]\]}}> : vector<3x2xi32>
//   CHECK-DAG: %[[GCST:.*]] = arith.constant dense<{{\[\[0, 1\], \[28, 29\], \[38, 39\]\]}}> : vector<3x2xi32>
//  CHECK-NEXT: return %[[ACST]], %[[BCST]], %[[CCST]], %[[DCST]], %[[ECST]], %[[FCST]], %[[GCST]]
func.func @insert_strided_2d_constant() ->
  (vector<3x2xi32>, vector<3x2xi32>, vector<3x2xi32>, vector<3x2xi32>, vector<3x2xi32>, vector<3x2xi32>, vector<3x2xi32>) {
  %vcst = arith.constant dense<[[0, 1], [2, 3], [4, 5]]> : vector<3x2xi32>
  %cst_1 = arith.constant dense<9> : vector<1xi32>
  %cst_2 = arith.constant dense<[18, 19]> : vector<2xi32>
  %cst_3 = arith.constant dense<[[28, 29], [38, 39]]> : vector<2x2xi32>
  %a = vector.insert_strided_slice %cst_1, %vcst {offsets = [1, 0], strides = [1]} : vector<1xi32> into vector<3x2xi32>
  %b = vector.insert_strided_slice %cst_1, %vcst {offsets = [2, 1], strides = [1]} : vector<1xi32> into vector<3x2xi32>
  %c = vector.insert_strided_slice %cst_2, %vcst {offsets = [0, 0], strides = [1]} : vector<2xi32> into vector<3x2xi32>
  %d = vector.insert_strided_slice %cst_2, %vcst {offsets = [1, 0], strides = [1]} : vector<2xi32> into vector<3x2xi32>
  %e = vector.insert_strided_slice %cst_2, %vcst {offsets = [2, 0], strides = [1]} : vector<2xi32> into vector<3x2xi32>
  %f = vector.insert_strided_slice %cst_3, %vcst {offsets = [0, 0], strides = [1, 1]} : vector<2x2xi32> into vector<3x2xi32>
  %g = vector.insert_strided_slice %cst_3, %vcst {offsets = [1, 0], strides = [1, 1]} : vector<2x2xi32> into vector<3x2xi32>
  return %a, %b, %c, %d, %e, %f, %g :
    vector<3x2xi32>, vector<3x2xi32>, vector<3x2xi32>, vector<3x2xi32>, vector<3x2xi32>, vector<3x2xi32>, vector<3x2xi32>
}

// -----

// CHECK-LABEL: func @shuffle_splatlike
//  CHECK-SAME:   (%[[ARG:.*]]: i32)
//  CHECK-NEXT:   %[[SPLAT:.*]] = vector.broadcast %[[ARG]] : i32 to vector<4xi32>
//  CHECK-NEXT:   return %[[SPLAT]] : vector<4xi32>
func.func @shuffle_splatlike(%x : i32) -> vector<4xi32> {
  %v0 = vector.broadcast %x : i32 to vector<4xi32>
  %v1 = vector.broadcast %x : i32 to vector<2xi32>
  %shuffle = vector.shuffle %v0, %v1 [2, 3, 4, 5] : vector<4xi32>, vector<2xi32>
  return %shuffle : vector<4xi32>
}


// -----

// CHECK-LABEL: func @insert_splatlike
//  CHECK-SAME:   (%[[ARG:.*]]: i32)
//  CHECK-NEXT:   %[[SPLAT:.*]] = vector.broadcast %[[ARG]] : i32 to vector<2x4x3xi32>
//  CHECK-NEXT:   return %[[SPLAT]] : vector<2x4x3xi32>
func.func @insert_splatlike(%x : i32) -> vector<2x4x3xi32> {
  %v0 = vector.broadcast %x : i32 to vector<4x3xi32>
  %v1 = vector.broadcast %x : i32 to vector<2x4x3xi32>
  %insert = vector.insert %v0, %v1[0] : vector<4x3xi32> into vector<2x4x3xi32>
  return %insert : vector<2x4x3xi32>
}

// -----

// CHECK-LABEL: func.func @transfer_read_from_rank_reducing_extract_slice
//       CHECK:   tensor.extract_slice
//       CHECK:   vector.transfer_read
func.func @transfer_read_from_rank_reducing_extract_slice(%src: tensor<1x8x8x8xf32>, %i1: index, %i2: index, %i3: index, %i4: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.extract_slice %src[0, %i1, %i2, %i3] [1, 4, 1, 4] [1, 1, 1, 1] : tensor<1x8x8x8xf32> to tensor<1x4x4xf32>
  %1 = vector.transfer_read %0[%c0, %i4, %c0], %f0 {in_bounds = [true]} : tensor<1x4x4xf32>, vector<4xf32>
  return %1 : vector<4xf32>
}

// -----

// CHECK-LABEL: func.func @extract_from_broadcast
func.func @extract_from_broadcast(%src: vector<1x1x1xf32>) -> vector<1xf32> {
  %0 = vector.broadcast %src : vector<1x1x1xf32> to vector<1x1x32x1xf32>

  //  CHECK-NEXT:   %0 = vector.extract {{.*}}[0, 0] : vector<1xf32> from vector<1x1x1xf32>
  //  CHECK-NEXT:   return %0 : vector<1xf32>
  %1 = vector.extract %0[0, 0, 31] : vector<1xf32> from vector<1x1x32x1xf32>
  return %1: vector<1xf32>
}

// CHECK-LABEL: func.func @extract_from_stretch_broadcast
func.func @extract_from_stretch_broadcast(%src: vector<3x1x2xf32>) -> f32 {
  //  CHECK-NEXT:  %0 = vector.extract {{.*}}[0, 0, 0] : f32 from vector<3x1x2xf32>
  //  CHECK-NEXT:  return %0 : f32
  %0 = vector.broadcast %src : vector<3x1x2xf32> to vector<3x4x2xf32>
  %1 = vector.extract %0[0, 2, 0] : f32 from vector<3x4x2xf32>
  return %1: f32
}

// -----
// CHECK-LABEL: func.func @extract_strided_slice_of_constant_mask
func.func @extract_strided_slice_of_constant_mask() -> vector<5x7xi1>{
  //  CHECK-NEXT:   %[[RES:.*]] = vector.constant_mask [5, 4] : vector<5x7xi1>
  //  CHECK-NEXT:   return %[[RES]] : vector<5x7xi1>
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %mask = vector.create_mask %c10, %c4 : vector<12x7xi1>
  %res = vector.extract_strided_slice %mask {offsets = [3], sizes = [5], strides = [1]} : vector<12x7xi1> to vector<5x7xi1>
  return %res : vector<5x7xi1>
}

// -----

// CHECK-LABEL: func.func @fold_0d_vector_reduction
func.func @fold_0d_vector_reduction(%arg0: vector<f32>) -> f32 {
  // CHECK-NEXT: %[[RES:.*]] = vector.extract %arg{{.*}}[] : f32 from vector<f32>
  // CHECK-NEXT: return %[[RES]] : f32
  %0 = vector.reduction <add>, %arg0 : vector<f32> into f32
  return %0 : f32
}

// -----

// CHECK-LABEL: func @empty_vector_mask
func.func @empty_vector_mask(%mask : vector<8xi1>) {
//   CHECK-NOT:   vector.mask
  vector.mask %mask { } : vector<8xi1>
  return
}

// -----

// CHECK-LABEL: func @empty_vector_mask_with_return
//  CHECK-SAME:     %[[IN:.*]]: vector<8xf32>
func.func @empty_vector_mask_with_return(%a : vector<8xf32>, %mask : vector<8xi1>) -> vector<8xf32> {
//   CHECK-NOT:   vector.mask
//       CHECK:   return %[[IN]] : vector<8xf32>
  %0 = vector.mask %mask { vector.yield %a : vector<8xf32> } : vector<8xi1> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// -----

// CHECK-LABEL: func @empty_vector_mask_with_passthru
//  CHECK-SAME:     %[[IN:.*]]: vector<8xf32>, %[[MASK:.*]]: vector<8xi1>, %[[PASSTHRU:.*]]: vector<8xf32>
func.func @empty_vector_mask_with_passthru(%a : vector<8xf32>, %mask : vector<8xi1>,
                                           %passthru : vector<8xf32>) -> vector<8xf32> {
//   CHECK-NOT:   vector.mask
//       CHECK:   %[[SEL:.*]] = arith.select %[[MASK]], %[[IN]], %[[PASSTHRU]] : vector<8xi1>, vector<8xf32>
//       CHECK:   return %[[SEL]] : vector<8xf32>
  %0 = vector.mask %mask, %passthru { vector.yield %a : vector<8xf32> } : vector<8xi1> -> vector<8xf32>
  return %0 : vector<8xf32>
}

// -----

// CHECK-LABEL: func @all_true_vector_mask
//  CHECK-SAME:     %[[IN:.*]]: tensor<3x4xf32>
func.func @all_true_vector_mask(%ta : tensor<3x4xf32>) -> vector<3x4xf32> {
//   CHECK-NOT:   vector.mask
//       CHECK:   %[[LD:.*]] = vector.transfer_read %[[IN]]
//       CHECK:   return %[[LD]] : vector<3x4xf32>
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %all_true = vector.constant_mask [3, 4] : vector<3x4xi1>
  %0 = vector.mask %all_true { vector.transfer_read %ta[%c0, %c0], %cf0 : tensor<3x4xf32>, vector<3x4xf32> } : vector<3x4xi1> -> vector<3x4xf32>
  return %0 : vector<3x4xf32>
}

// -----

// CHECK-LABEL: func @all_true_vector_mask_no_result(
func.func @all_true_vector_mask_no_result(%a : vector<3x4xf32>, %m : memref<3x4xf32>) {
//   CHECK-NOT:   vector.mask
//       CHECK:   vector.transfer_write
  %c0 = arith.constant 0 : index
  %all_true = vector.constant_mask [3, 4] : vector<3x4xi1>
  vector.mask %all_true { vector.transfer_write %a, %m[%c0, %c0] : vector<3x4xf32>, memref<3x4xf32> } : vector<3x4xi1>
  return
}

// -----

// CHECK-LABEL:   func.func @fold_shape_cast_with_mask(
// CHECK-SAME:     %[[VAL_0:.*]]: tensor<1x?xf32>) -> vector<1x4xi1> {
func.func @fold_shape_cast_with_mask(%arg0: tensor<1x?xf32>) -> vector<1x4xi1> {
// CHECK-NOT: vector.shape_cast
// CHECK:     %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:     %[[VAL_2:.*]] = tensor.dim %[[VAL_0]], %[[VAL_1]] : tensor<1x?xf32>
// CHECK:     %[[VAL_3:.*]] = vector.create_mask %[[VAL_1]], %[[VAL_2]] : vector<1x4xi1>
// CHECK:     return %[[VAL_3]] : vector<1x4xi1>
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c1 : tensor<1x?xf32>
  %1 = vector.create_mask %c1, %dim, %c1, %c1 : vector<1x4x1x1xi1>
  %2 = vector.shape_cast %1 : vector<1x4x1x1xi1> to vector<1x4xi1>
  return %2 : vector<1x4xi1>
}

// -----

// CHECK-LABEL:   func.func @fold_shape_cast_with_mask_scalable(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<1x?xf32>) -> vector<1x[4]xi1> {
func.func @fold_shape_cast_with_mask_scalable(%arg0: tensor<1x?xf32>) -> vector<1x[4]xi1> {
// CHECK-NOT: vector.shape_cast
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = tensor.dim %[[VAL_0]], %[[VAL_1]] : tensor<1x?xf32>
// CHECK:           %[[VAL_3:.*]] = vector.create_mask %[[VAL_1]], %[[VAL_2]] : vector<1x[4]xi1>
// CHECK:           return %[[VAL_3]] : vector<1x[4]xi1>
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c1 : tensor<1x?xf32>
  %1 = vector.create_mask %c1, %dim, %c1, %c1 : vector<1x[4]x1x1xi1>
  %2 = vector.shape_cast %1 : vector<1x[4]x1x1xi1> to vector<1x[4]xi1>
  return %2 : vector<1x[4]xi1>
}

// -----

// Check that scalable "1" (i.e. [1]) is not folded
// CHECK-LABEL:   func.func @fold_shape_cast_with_mask_scalable_one(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<1x?xf32>) -> vector<1x[1]xi1> {
func.func @fold_shape_cast_with_mask_scalable_one(%arg0: tensor<1x?xf32>) -> vector<1x[1]xi1>{
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = tensor.dim %[[VAL_0]], %[[VAL_1]] : tensor<1x?xf32>
// CHECK:           %[[VAL_3:.*]] = vector.create_mask %[[VAL_1]], %[[VAL_2]] : vector<1x[1]xi1>
// CHECK:           return %[[VAL_3]] : vector<1x[1]xi1>
  %c1 = arith.constant 1 : index
  %dim = tensor.dim %arg0, %c1 : tensor<1x?xf32>
  %1 = vector.create_mask %c1, %dim, %c1 : vector<1x[1]x1xi1>
  %2 = vector.shape_cast %1 : vector<1x[1]x1xi1> to vector<1x[1]xi1>
  return %2 : vector<1x[1]xi1>
}

// -----

// CHECK-LABEL:   func.func @fold_shape_cast_with_constant_mask() -> vector<4xi1> {
func.func @fold_shape_cast_with_constant_mask() -> vector<4xi1>{
// CHECK-NOT: vector.shape_cast
// CHECK:           %[[VAL_0:.*]] = vector.constant_mask [1] : vector<4xi1>
// CHECK:           return %[[VAL_0]] : vector<4xi1>
  %1 = vector.constant_mask [1, 1, 1] : vector<4x1x1xi1>
  %2 = vector.shape_cast %1 : vector<4x1x1xi1> to vector<4xi1>
  return %2 : vector<4xi1>
}

// -----

// TODO: This IR could be canonicalized but the canonicalization pattern is not
// smart enough. For now, just make sure that we do not crash.

// CHECK-LABEL: func.func @load_store_forwarding_rank_mismatch(
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_read
func.func @load_store_forwarding_rank_mismatch(%v0: vector<4x1x1xf32>, %arg0: tensor<4x4x4xf32>) -> (vector<1x100x4x5xf32>) {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  // d0 is explicitly written.
  %w0 = vector.transfer_write %v0, %arg0[%c0, %c0, %c0]
      {in_bounds = [true, true, true],
      permutation_map = affine_map<(d0, d1, d2) -> (d2, d1, d0)>} :
      vector<4x1x1xf32>, tensor<4x4x4xf32>
  // d0 is implicitly read (rank-reduction of unit dim).
  %r = vector.transfer_read %w0[%c0, %c0, %c0], %cf0
      {in_bounds = [true, true, true, true],
      permutation_map = affine_map<(d0, d1, d2) -> (d1, 0, d2, 0)>} :
      tensor<4x4x4xf32>, vector<1x100x4x5xf32>
  return %r : vector<1x100x4x5xf32>
}

// -----

// CHECK-LABEL: func.func @rank_0_shuffle_to_interleave(
//  CHECK-SAME:     %[[LHS:.*]]: vector<f64>, %[[RHS:.*]]: vector<f64>)
func.func @rank_0_shuffle_to_interleave(%arg0: vector<f64>, %arg1: vector<f64>) -> vector<2xf64> {
  // CHECK: %[[ZIP:.*]] = vector.interleave %[[LHS]], %[[RHS]] : vector<f64> -> vector<2xf64>
  // CHECK: return %[[ZIP]]
  %0 = vector.shuffle %arg0, %arg1 [0, 1] : vector<f64>, vector<f64>
  return %0 : vector<2xf64>
}

// -----

// CHECK-LABEL: func.func @rank_1_shuffle_to_interleave(
//  CHECK-SAME:     %[[LHS:.*]]: vector<6xi32>, %[[RHS:.*]]: vector<6xi32>)
func.func @rank_1_shuffle_to_interleave(%arg0: vector<6xi32>, %arg1: vector<6xi32>) -> vector<12xi32> {
  // CHECK: %[[ZIP:.*]] = vector.interleave %[[LHS]], %[[RHS]] : vector<6xi32> -> vector<12xi32>
  // CHECK: return %[[ZIP]]
  %0 = vector.shuffle %arg0, %arg1 [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11] : vector<6xi32>, vector<6xi32>
  return %0 : vector<12xi32>
}

// -----

// CHECK-LABEL: func @extract_from_splatlike_broadcast(
//  CHECK-SAME:     %[[A:.*]]: f32, %[[B:.*]]: vector<f32>, %[[C:.*]]: vector<2xf32>)
func.func @extract_from_splatlike_broadcast(%a: f32, %b: vector<f32>, %c: vector<2xf32>) -> (f32, f32, f32, f32, vector<6x7xf32>, vector<3xf32>) {
  // Broadcast scalar to 0D and extract scalar.
  %0 = vector.broadcast %a : f32 to vector<f32>
  %1 = vector.extract %0[] : f32 from vector<f32>

  // Broadcast 0D to 3D and extract scalar.
  // CHECK: %[[EXTRACT1:.*]] = vector.extract %[[B]][] : f32 from vector<f32>
  %4 = vector.broadcast %b : vector<f32> to vector<1x2x4xf32>
  %5 = vector.extract %4[0, 0, 1] : f32 from vector<1x2x4xf32>

  // Broadcast scalar to 2D and extract scalar.
  %6 = vector.broadcast %a : f32 to vector<2x3xf32>
  %7 = vector.extract %6[0, 1] : f32 from vector<2x3xf32>

  // Broadcast scalar to 3D and extract scalar.
  %8 = vector.broadcast %a : f32 to vector<5x6x7xf32>
  %9 = vector.extract %8[2, 1, 5] : f32 from vector<5x6x7xf32>

  // Extract 2D from 3D that was broadcasted from a scalar.
  // CHECK: %[[EXTRACT2:.*]] = vector.broadcast %[[A]] : f32 to vector<6x7xf32>
  %10 = vector.extract %8[2] : vector<6x7xf32> from vector<5x6x7xf32>

  // Extract 1D from 2D that was splat'ed from a scalar.
  // CHECK: %[[EXTRACT3:.*]] = vector.broadcast %[[A]] : f32 to vector<3xf32>
  %11 = vector.extract %6[1] : vector<3xf32> from vector<2x3xf32>

  // CHECK:   return %[[A]], %[[EXTRACT1]], %[[A]], %[[A]], %[[EXTRACT2]], %[[EXTRACT3]]
  return %1, %5, %7, %9, %10, %11 : f32, f32, f32, f32, vector<6x7xf32>, vector<3xf32>
}

// -----

// CHECK-LABEL: func @to_elements_from_elements_no_op(
// CHECK-SAME:     %[[A:.*]]: f32, %[[B:.*]]: f32
func.func @to_elements_from_elements_no_op(%a: f32, %b: f32) -> (f32, f32) {
  // CHECK-NOT: vector.from_elements
  // CHECK-NOT: vector.to_elements
  %0 = vector.from_elements %b, %a : vector<2xf32>
  %1:2 = vector.to_elements %0 : vector<2xf32>
  // CHECK: return %[[B]], %[[A]]
  return %1#0, %1#1 : f32, f32
}

// -----

// CHECK-LABEL: func @from_elements_to_elements_no_op(
// CHECK-SAME:     %[[A:.*]]: vector<4x2xf32>
func.func @from_elements_to_elements_no_op(%a: vector<4x2xf32>) -> vector<4x2xf32> {
  // CHECK-NOT: vector.from_elements
  // CHECK-NOT: vector.to_elements
  %0:8 = vector.to_elements %a : vector<4x2xf32>
  %1 = vector.from_elements %0#0, %0#1, %0#2, %0#3, %0#4, %0#5, %0#6, %0#7 : vector<4x2xf32>
  // CHECK: return %[[A]]
  return %1 : vector<4x2xf32>
}

// -----

// CHECK-LABEL: func @from_elements_to_elements_dup_elems(
// CHECK-SAME:     %[[A:.*]]: vector<4xf32>
func.func @from_elements_to_elements_dup_elems(%a: vector<4xf32>) -> vector<4x2xf32> {
  // CHECK: %[[TO_EL:.*]]:4 = vector.to_elements %[[A]]
  // CHECK: %[[FROM_EL:.*]] = vector.from_elements %[[TO_EL]]#0, %[[TO_EL]]#1, %[[TO_EL]]#2
  %0:4 = vector.to_elements %a : vector<4xf32> // 4 elements
  %1 = vector.from_elements %0#0, %0#1, %0#2, %0#3, %0#0, %0#1, %0#2, %0#3 : vector<4x2xf32>
  // CHECK: return %[[FROM_EL]]
  return %1 : vector<4x2xf32>
}

// -----

// CHECK-LABEL: func @from_elements_to_elements_shuffle(
// CHECK-SAME:     %[[A:.*]]: vector<4x2xf32>
func.func @from_elements_to_elements_shuffle(%a: vector<4x2xf32>) -> vector<4x2xf32> {
  // CHECK: %[[TO_EL:.*]]:8 = vector.to_elements %[[A]]
  // CHECK: %[[FROM_EL:.*]] = vector.from_elements %[[TO_EL]]#7, %[[TO_EL]]#0, %[[TO_EL]]#6
  %0:8 = vector.to_elements %a : vector<4x2xf32>
  %1 = vector.from_elements %0#7, %0#0, %0#6, %0#1, %0#5, %0#2, %0#4, %0#3 : vector<4x2xf32>
  // CHECK: return %[[FROM_EL]]
  return %1 : vector<4x2xf32>
}

// -----

// CHECK-LABEL: func @to_elements_of_scalar_broadcast_folds
// CHECK-SAME: (%[[S:.*]]: f32) -> (f32, f32, f32, f32)
func.func @to_elements_of_scalar_broadcast_folds(%s: f32) -> (f32, f32, f32, f32) {
  %v = vector.broadcast %s : f32 to vector<4xf32>
  %e:4 = vector.to_elements %v : vector<4xf32>
  // CHECK-NOT: vector.broadcast
  // CHECK-NOT: vector.to_elements
  // CHECK: return %[[S]], %[[S]], %[[S]], %[[S]]
  return %e#0, %e#1, %e#2, %e#3 : f32, f32, f32, f32
}

// -----

// CHECK-LABEL: func @to_elements_of_vector_broadcast
// CHECK-SAME: (%[[VEC:.*]]: vector<2xf32>) -> (f32, f32, f32, f32, f32, f32)
func.func @to_elements_of_vector_broadcast(%vec: vector<2xf32>) -> (f32, f32, f32, f32, f32, f32) {
  %v = vector.broadcast %vec : vector<2xf32> to vector<3x2xf32>
  %e:6 = vector.to_elements %v : vector<3x2xf32>
  // CHECK-NOT: vector.broadcast
  // CHECK: %[[SRC_ELEMS:.*]]:2 = vector.to_elements %[[VEC]]
  // CHECK: return %[[SRC_ELEMS]]#0, %[[SRC_ELEMS]]#1, %[[SRC_ELEMS]]#0, %[[SRC_ELEMS]]#1, %[[SRC_ELEMS]]#0, %[[SRC_ELEMS]]#1
  return %e#0, %e#1, %e#2, %e#3, %e#4, %e#5 : f32, f32, f32, f32, f32, f32
}

// -----

// CHECK-LABEL: func @to_elements_of_vector_broadcast_inner_dim
// CHECK-SAME: (%[[V:.*]]: vector<2x1x2xf32>) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
func.func @to_elements_of_vector_broadcast_inner_dim(%v: vector<2x1x2xf32>) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
  %b = vector.broadcast %v : vector<2x1x2xf32> to vector<2x3x2xf32>
  %e:12 = vector.to_elements %b : vector<2x3x2xf32>
  // CHECK-NOT: vector.broadcast
  // CHECK: %[[SRC:.*]]:4 = vector.to_elements %[[V]] : vector<2x1x2xf32>
  // CHECK: return %[[SRC]]#0, %[[SRC]]#1, %[[SRC]]#0, %[[SRC]]#1, %[[SRC]]#0, %[[SRC]]#1, %[[SRC]]#2, %[[SRC]]#3, %[[SRC]]#2, %[[SRC]]#3, %[[SRC]]#2, %[[SRC]]#3
  return %e#0, %e#1, %e#2, %e#3, %e#4, %e#5, %e#6, %e#7, %e#8, %e#9, %e#10, %e#11 :
    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32
}

// -----

// +---------------------------------------------------------------------------
// Tests for foldFromElementsToConstant
// +---------------------------------------------------------------------------

// CHECK-LABEL: func @from_elements_to_constant(
func.func @from_elements_to_constant() -> vector<2x2xi32> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c3_i32 = arith.constant 3 : i32
  // CHECK: %[[RES:.*]] = arith.constant dense<{{\[\[0, 1\], \[2, 3\]\]}}> : vector<2x2xi32>
  %res = vector.from_elements %c0_i32, %c1_i32, %c2_i32, %c3_i32 : vector<2x2xi32>
  // CHECK: return %[[RES]]
  return %res : vector<2x2xi32>
}

// -----

// One of the elements is not a constant, the folder should fail.

// CHECK-LABEL: func @negative_from_elements_to_constant(
// CHECK-SAME:     %[[A:.*]]: f32
func.func @negative_from_elements_to_constant(%arg0: f32) -> vector<2xf32> {
  // CHECK: %[[C:.*]] = arith.constant 1.000000e+00 : f32
  %c = arith.constant 1.0 : f32
  // CHECK: %[[RES:.*]] = vector.from_elements %[[A]], %[[C]] : vector<2xf32>
  %res = vector.from_elements %arg0, %c : vector<2xf32>
  // CHECK: return %[[RES]]
  return %res : vector<2xf32>
}

// -----

// While all inputs in this example are constant, we cannot create a
// DenselElemAttr containing llvm.mlir.addressof. Instead,
// `foldFromElementsToConstant` bails out. Note that in this case, a different
// folder is applied (`rewriteFromElementsAsBroadcast`).
llvm.mlir.global constant @my_symbol() : i32

// CHECK-LABEL: func @negative_from_elements_to_constant
//       CHECK:   %[[A:.*]] = llvm.mlir.addressof @my_symbol
//       CHECK:   %[[B:.*]] = vector.broadcast %[[A]] : !llvm.ptr to vector<1x!llvm.ptr>
//       CHECK:   return %[[B]]
func.func @negative_from_elements_to_constant() -> vector<1x!llvm.ptr> {
  %a = llvm.mlir.addressof @my_symbol : !llvm.ptr
  %b = vector.from_elements %a : vector<1x!llvm.ptr>
  return %b : vector<1x!llvm.ptr>
}

// -----

// `foldFromElementsToConstant` does not support `ub.poison`, so it bails out.
// Instead, other folders apply here (e.g. `rewriteFromElementsAsBroadcast`).

// CHECK-LABEL: @negative_from_elements_poison
//       CHECK:   %[[VAL:.*]] = ub.poison : vector<2xf32>
//       CHECK:   return %[[VAL]] : vector<2xf32>
func.func @negative_from_elements_poison_f32() -> vector<2xf32> {
  %0 = ub.poison : f32
  %1 = vector.from_elements %0, %0 : vector<2xf32>
  return %1 : vector<2xf32>
}

// -----

// `foldFromElementsToConstant` does not support `ub.poison`, so it bails out.
// Instead, other folders apply here (e.g. `rewriteFromElementsAsBroadcast`).

// CHECK-LABEL: @negative_from_elements_poison_i32
//       CHECK:   %[[VAL:.*]] = ub.poison : vector<2xi32>
//       CHECK:   return %[[VAL]] : vector<2xi32>
func.func @negative_from_elements_poison_i32() -> vector<2xi32> {
  %0 = ub.poison : i32
  %1 = vector.from_elements %0, %0 : vector<2xi32>
  return %1 : vector<2xi32>
}

// -----

// `foldFromElementsToConstant` does not support `ub.poison`, so it bails out.
// Instead, other folders apply here (e.g. `rewriteFromElementsAsBroadcast`).

// CHECK-LABEL: @negative_from_elements_poison_constant_mix
//       CHECK:   %[[POISON:.*]] = ub.poison : f32
//       CHECK:   %[[CONST:.*]] = arith.constant 1.000000e+00 : f32
//       CHECK:   %[[RES:.*]] = vector.from_elements %[[POISON]], %[[CONST]] : vector<2xf32>
//       CHECK:   return %[[RES]] : vector<2xf32>
func.func @negative_from_elements_poison_constant_mix() -> vector<2xf32> {
  %0 = ub.poison : f32
  %c = arith.constant 1.0 : f32
  %1 = vector.from_elements %0, %c : vector<2xf32>
  return %1 : vector<2xf32>
}

// -----

// CHECK-LABEL: func @from_elements_float8_to_i8_conversion(
// CHECK-NEXT:    %[[CST:.*]] = arith.constant dense<[0, 56, -72, 69, 127, -1]> : vector<6xi8>
// CHECK-NEXT:    return %[[CST]] : vector<6xi8>
func.func @from_elements_float8_to_i8_conversion() -> vector<6xi8> {
  %cst0 = llvm.mlir.constant(0.0 : f8E4M3FN) : i8
  %cst1 = llvm.mlir.constant(1.0 : f8E4M3FN) : i8
  %cst_neg1 = llvm.mlir.constant(-1.0 : f8E4M3FN) : i8
  %cst_pi = llvm.mlir.constant(3.14 : f8E4M3FN) : i8
  %cst_inf = llvm.mlir.constant(0x7F : f8E4M3FN) : i8
  %cst_neg_inf = llvm.mlir.constant(0xFF : f8E4M3FN) : i8
  %v = vector.from_elements %cst0, %cst1, %cst_neg1, %cst_pi, %cst_inf, %cst_neg_inf : vector<6xi8>
  return %v : vector<6xi8>
}

// CHECK-LABEL: func @from_elements_float16_to_i16_conversion(
// CHECK-NEXT:    %[[CST:.*]] = arith.constant dense<[0, 15360, -17408, 16968, 31743, -1025]> : vector<6xi16>
// CHECK-NEXT:    return %[[CST]] : vector<6xi16>
func.func @from_elements_float16_to_i16_conversion() -> vector<6xi16> {
  %cst0 = llvm.mlir.constant(0.0 : f16) : i16
  %cst1 = llvm.mlir.constant(1.0 : f16) : i16
  %cst_neg1 = llvm.mlir.constant(-1.0 : f16) : i16
  %cst_pi = llvm.mlir.constant(3.14 : f16) : i16
  %cst_max = llvm.mlir.constant(65504.0	: f16) : i16
  %cst_min = llvm.mlir.constant(-65504.0 : f16) : i16
  %v = vector.from_elements %cst0, %cst1, %cst_neg1, %cst_pi, %cst_max, %cst_min : vector<6xi16>
  return %v : vector<6xi16>
}

// CHECK-LABEL: func @from_elements_f64_to_i64_conversion(
// CHECK-NEXT:    %[[CST:.*]] = arith.constant dense<[0, 4607182418800017408, -4616189618054758400, 4614253070214989087, 9218868437227405311, -4503599627370497]> : vector<6xi64>
// CHECK-NEXT:    return %[[CST]] : vector<6xi64>
func.func @from_elements_f64_to_i64_conversion() -> vector<6xi64> {
  %cst0 = llvm.mlir.constant(0.0 : f64) : i64
  %cst1 = llvm.mlir.constant(1.0 : f64) : i64
  %cst_neg1 = llvm.mlir.constant(-1.0 : f64) : i64
  %cst_pi = llvm.mlir.constant(3.14 : f64) : i64
  %cst_max = llvm.mlir.constant(1.7976931348623157e+308 : f64) : i64
  %cst_min = llvm.mlir.constant(-1.7976931348623157e+308 : f64) : i64
  %v = vector.from_elements %cst0, %cst1, %cst_neg1, %cst_pi, %cst_max, %cst_min : vector<6xi64>
  return %v : vector<6xi64>
}

// -----

// CHECK-LABEL: func @from_elements_i1_to_i8_conversion(
// CHECK-NEXT:    %[[CST:.*]] = arith.constant dense<0> : vector<1xi8>
// CHECK-NEXT:    return %[[CST]] : vector<1xi8>
func.func @from_elements_i1_to_i8_conversion() -> vector<1xi8> {
  %cst = llvm.mlir.constant(0: i1) : i8
  %v = vector.from_elements %cst : vector<1xi8>
  return %v : vector<1xi8>
}

// -----

// CHECK-LABEL: func @from_elements_index_to_i64_conversion(
// CHECK-NEXT:    %[[CST:.*]] = arith.constant dense<[0, 1, 42]> : vector<3xi64>
// CHECK-NEXT:    return %[[CST]] : vector<3xi64>
func.func @from_elements_index_to_i64_conversion() -> vector<3xi64> {
  %cst0 = llvm.mlir.constant(0 : index) : i64
  %cst1 = llvm.mlir.constant(1 : index) : i64
  %cst42 = llvm.mlir.constant(42 : index) : i64
  %v = vector.from_elements %cst0, %cst1, %cst42 : vector<3xi64>
  return %v : vector<3xi64>
}

// +---------------------------------------------------------------------------
// End of  Tests for foldFromElementsToConstant
// +---------------------------------------------------------------------------

// -----

// Not a DenseElementsAttr, don't fold.

// CHECK-LABEL: func @negative_insert_llvm_undef(
//       CHECK:   llvm.mlir.undef
//       CHECK:   vector.insert
func.func @negative_insert_llvm_undef(%arg0: i8) -> vector<4xi8> {
  %0 = llvm.mlir.undef : vector<4xi8>
  %1 = vector.insert %arg0, %0 [0] : i8 into vector<4xi8>
  return %1 : vector<4xi8>
}

// -----

// Insert a poison value shouldn't be folded as the resulting vector is not
// fully poison.

// CHECK-LABEL: @insert_scalar_poison
func.func @insert_scalar_poison(%a: vector<4x8xf32>)
    -> vector<4x8xf32> {
  //  CHECK-NEXT: %[[UB:.*]] = ub.poison : f32
  //  CHECK-NEXT: %[[RES:.*]] = vector.insert %[[UB]]
  //  CHECK-NEXT: return %[[RES]] : vector<4x8xf32>
  %0 = ub.poison : f32
  %1 = vector.insert %0, %a[2, 3] : f32 into vector<4x8xf32>
  return %1 : vector<4x8xf32>
}

// -----

// Insert a poison value shouldn't be folded as the resulting vector is not
// fully poison.

// CHECK-LABEL: @insert_vector_poison
func.func @insert_vector_poison(%a: vector<4x8xf32>)
    -> vector<4x8xf32> {
  //  CHECK-NEXT: %[[UB:.*]] = ub.poison : vector<8xf32>
  //  CHECK-NEXT: %[[RES:.*]] = vector.insert %[[UB]]
  //  CHECK-NEXT: return %[[RES]] : vector<4x8xf32>
  %0 = ub.poison : vector<8xf32>
  %1 = vector.insert %0, %a[2] : vector<8xf32> into vector<4x8xf32>
  return %1 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: @insert_scalar_poison_idx
func.func @insert_scalar_poison_idx(%a: vector<4x5xf32>, %b: f32)
    -> vector<4x5xf32> {
  // CHECK-NEXT: %[[UB:.*]] = ub.poison : vector<4x5xf32>
  //  CHECK-NOT: vector.insert
  // CHECK-NEXT: return %[[UB]] : vector<4x5xf32>
  %0 = vector.insert %b, %a[-1, 0] : f32 into vector<4x5xf32>
  return %0 : vector<4x5xf32>
}

// -----

// CHECK-LABEL: @insert_vector_poison_idx
func.func @insert_vector_poison_idx(%a: vector<4x5xf32>, %b: vector<5xf32>)
    -> vector<4x5xf32> {
  // CHECK-NEXT: %[[UB:.*]] = ub.poison : vector<4x5xf32>
  //  CHECK-NOT: vector.insert
  // CHECK-NEXT: return %[[UB]] : vector<4x5xf32>
  %0 = vector.insert %b, %a[-1] : vector<5xf32> into vector<4x5xf32>
  return %0 : vector<4x5xf32>
}

// -----

// Similar to the test above, but the index is not a static constant.

// CHECK-LABEL: @insert_vector_poison_idx_non_cst
func.func @insert_vector_poison_idx_non_cst(%a: vector<4x5xf32>, %b: vector<5xf32>)
    -> vector<4x5xf32> {
  // CHECK-NEXT: %[[UB:.*]] = ub.poison : vector<4x5xf32>
  //  CHECK-NOT: vector.insert
  // CHECK-NEXT: return %[[UB]] : vector<4x5xf32>
  %c_neg_1 = arith.constant -1 : index
  %0 = vector.insert %b, %a[%c_neg_1] : vector<5xf32> into vector<4x5xf32>
  return %0 : vector<4x5xf32>
}

// -----

// Similar to test above, but now the index is out-of-bounds.

// CHECK-LABEL: @no_fold_insert_scalar_idx_oob
func.func @no_fold_insert_scalar_idx_oob(%a: vector<4x5xf32>, %b: vector<5xf32>)
    -> vector<4x5xf32> {
  //  CHECK: vector.insert
  %c_neg_2 = arith.constant -2 : index
  %0 = vector.insert %b, %a[%c_neg_2] : vector<5xf32> into vector<4x5xf32>
  return %0 : vector<4x5xf32>
}

// -----

// CHECK-LABEL: @insert_multiple_poison_idx
func.func @insert_multiple_poison_idx(%a: vector<4x5x8xf32>, %b: vector<8xf32>)
    -> vector<4x5x8xf32> {
  // CHECK-NEXT: %[[UB:.*]] = ub.poison : vector<4x5x8xf32>
  //  CHECK-NOT: vector.insert
  // CHECK-NEXT: return %[[UB]] : vector<4x5x8xf32>
  %0 = vector.insert %b, %a[-1, -1] : vector<8xf32> into vector<4x5x8xf32>
  return %0 : vector<4x5x8xf32>
}

// -----

// CHECK-LABEL: @contiguous_extract_strided_slices_to_extract_sizes_and_outer_source_dims_overlap
// CHECK:        %[[EXTRACT:.+]] = vector.extract {{.*}}[0, 0, 0, 0, 0] : vector<4xi32> from vector<8x1x2x1x1x4xi32>
// CHECK-NEXT:   return %[[EXTRACT]] :  vector<4xi32>
func.func @contiguous_extract_strided_slices_to_extract_sizes_and_outer_source_dims_overlap(%arg0 : vector<8x1x2x1x1x4xi32>) -> vector<4xi32> {
  %1 = vector.extract_strided_slice %arg0 {offsets = [0, 0, 0, 0, 0, 0], sizes = [1, 1, 1, 1, 1, 4], strides = [1, 1, 1, 1, 1, 1]} : vector<8x1x2x1x1x4xi32> to vector<1x1x1x1x1x4xi32>
  %2 = vector.shape_cast %1 : vector<1x1x1x1x1x4xi32> to vector<4xi32>
  return %2 : vector<4xi32>
}

// -----

// CHECK-LABEL: @contiguous_extract_strided_slices_to_extract_sizes_and_outer_source_dims_no_overlap
// CHECK:        %[[EXTRACT:.+]] = vector.extract {{.*}}[0, 0] : vector<4xi32> from vector<8x2x4xi32>
// CHECK-NEXT:   return %[[EXTRACT]] :  vector<4xi32>
func.func @contiguous_extract_strided_slices_to_extract_sizes_and_outer_source_dims_no_overlap(%arg0 : vector<8x2x4xi32>) -> vector<4xi32> {
  %1 = vector.extract_strided_slice %arg0 {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]} : vector<8x2x4xi32> to vector<1x1x4xi32>
  %2 = vector.shape_cast %1 : vector<1x1x4xi32> to vector<4xi32>
  return %2 : vector<4xi32>
}

// -----

// CHECK-LABEL: @contiguous_extract_strided_slices_to_extract_shorter_size_list
// CHECK:        %[[EXTRACT:.+]] = vector.extract {{.*}}[0, 0, 0, 0] : vector<1x4xi32> from vector<8x1x2x1x1x4xi32>
// CHECK-NEXT:   return %[[EXTRACT]] :  vector<1x4xi32>
func.func @contiguous_extract_strided_slices_to_extract_shorter_size_list(%arg0 : vector<8x1x2x1x1x4xi32>) -> vector<1x4xi32> {
  %1 = vector.extract_strided_slice %arg0 {offsets = [0, 0, 0, 0, 0], sizes = [1, 1, 1, 1, 1], strides = [1, 1, 1, 1, 1]} : vector<8x1x2x1x1x4xi32> to vector<1x1x1x1x1x4xi32>
  %2 = vector.shape_cast %1 : vector<1x1x1x1x1x4xi32> to vector<1x4xi32>
  return %2 : vector<1x4xi32>
}

// -----

// CHECK-LABEL: @contiguous_extract_strided_slices_to_extract_failure_non_unit_outer_size
// CHECK-NEXT:   vector.extract_strided_slice
func.func @contiguous_extract_strided_slices_to_extract_failure_non_unit_outer_size(%arg0 : vector<8x1x2x1x1x4xi32>) -> vector<8x1x1x1x1x4xi32> {
  %1 = vector.extract_strided_slice %arg0 {offsets = [0, 0, 0, 0, 0, 0], sizes = [8, 1, 1, 1, 1, 4], strides = [1, 1, 1, 1, 1, 1]} : vector<8x1x2x1x1x4xi32> to vector<8x1x1x1x1x4xi32>
  return %1 : vector<8x1x1x1x1x4xi32>
}

// -----

// CHECK-LABEL: @contiguous_extract_strided_slices_to_extract_failure_non_full_size
// CHECK-NEXT:   vector.extract_strided_slice
func.func @contiguous_extract_strided_slices_to_extract_failure_non_full_size(%arg0 : vector<8x1x2x1x1x4xi32>) -> vector<1x1x1x1x1x2xi32> {
  %1 = vector.extract_strided_slice %arg0 {offsets = [0, 0, 0, 0, 0, 0], sizes = [1, 1, 1, 1, 1, 2], strides = [1, 1, 1, 1, 1, 1]} : vector<8x1x2x1x1x4xi32> to vector<1x1x1x1x1x2xi32>
  return %1 : vector<1x1x1x1x1x2xi32>
}

// -----

// CHECK-LABEL: @contiguous_extract_strided_slices_to_extract_failure_non_full_inner_size
// CHECK-NEXT:    vector.extract_strided_slice
func.func @contiguous_extract_strided_slices_to_extract_failure_non_full_inner_size(%arg0 : vector<8x1x2x1x1x4xi32>) -> vector<1x1x2x1x1x1xi32> {
  %1 = vector.extract_strided_slice %arg0 {offsets = [0, 0, 0, 0, 0, 0], sizes = [1, 1, 2, 1, 1, 1], strides = [1, 1, 1, 1, 1, 1]} : vector<8x1x2x1x1x4xi32> to vector<1x1x2x1x1x1xi32>
  return %1 : vector<1x1x2x1x1x1xi32>
}

// -----

// CHECK-LABEL: @contiguous_gather
//  CHECK-SAME:   (%[[BASE:.*]]: memref<?xf32>, %[[MASK:.*]]: vector<16xi1>, %[[PASSTHRU:.*]]: vector<16xf32>)
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[R:.*]] = vector.maskedload %[[BASE]][%[[C0]]], %[[MASK]], %[[PASSTHRU]] : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
//       CHECK:   return %[[R]]
func.func @contiguous_gather(%base: memref<?xf32>,
                             %mask: vector<16xi1>, %passthru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %indices = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
  %1 = vector.gather %base[%c0][%indices], %mask, %passthru :
    memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %1 : vector<16xf32>
}

// -----

// CHECK-LABEL: @contiguous_gather_non_zero_start(
//  TODO: Non-zero start is not supported yet.
//       CHECK:   %[[R:.*]] = vector.gather
//       CHECK:   return %[[R]]
func.func @contiguous_gather_non_zero_start(%base: memref<?xf32>,
                                            %mask: vector<16xi1>,
                                            %passthru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %indices = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : vector<16xi32>
  %1 = vector.gather %base[%c0][%indices], %mask, %passthru :
    memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %1 : vector<16xf32>
}

// -----

// CHECK-LABEL: @contiguous_gather_2d(
// TODO: Only 1D vectors are supported.
//       CHECK:   %[[R:.*]] = vector.gather
//       CHECK:   return %[[R]]
func.func @contiguous_gather_2d(%base: memref<?x?xf32>,
                                %mask: vector<4x4xi1>, %passthru: vector<4x4xf32>) -> vector<4x4xf32> {
  %c0 = arith.constant 0 : index
  %indices = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]> : vector<4x4xi32>
  %1 = vector.gather %base[%c0, %c0][%indices], %mask, %passthru :
    memref<?x?xf32>, vector<4x4xi32>, vector<4x4xi1>, vector<4x4xf32> into vector<4x4xf32>
  return %1 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: @contiguous_gather_const_mask
//  CHECK-SAME:   (%[[BASE:.*]]: memref<?xf32>, %[[PASSTHRU:.*]]: vector<16xf32>)
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[R:.*]] = vector.load %[[BASE]][%[[C0]]] : memref<?xf32>, vector<16xf32>
//       CHECK:   return %[[R]]
func.func @contiguous_gather_const_mask(%base: memref<?xf32>,
                                        %passthru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %indices = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
  %mask = arith.constant dense<true> : vector<16xi1>
  %1 = vector.gather %base[%c0][%indices], %mask, %passthru :
    memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %1 : vector<16xf32>
}

// -----

// CHECK-LABEL: @contiguous_gather_step
//  CHECK-SAME:   (%[[BASE:.*]]: memref<?xf32>, %[[MASK:.*]]: vector<16xi1>, %[[PASSTHRU:.*]]: vector<16xf32>)
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[R:.*]] = vector.maskedload %[[BASE]][%[[C0]]], %[[MASK]], %[[PASSTHRU]] : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
//       CHECK:   return %[[R]]
func.func @contiguous_gather_step(%base: memref<?xf32>,
                                  %mask: vector<16xi1>, %passthru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %indices = vector.step : vector<16xindex>
  %1 = vector.gather %base[%c0][%indices], %mask, %passthru :
    memref<?xf32>, vector<16xindex>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %1 : vector<16xf32>
}

// -----

// CHECK-LABEL: @no_fold_contiguous_gather_tensor
func.func @no_fold_contiguous_gather_tensor(%base: tensor<8xf32>, %mask: vector<4xi1>, %pass_thru: vector<4xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %indices = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
  // CHECK: vector.gather
  // CHECK-NOT: vector.maskedload
  %0 = vector.gather %base[%c0][%indices], %mask, %pass_thru :
    tensor<8xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: @gather_broadcast(
// TODO: Broadcast is not supported yet
//       CHECK:   %[[R:.*]] = vector.gather
//       CHECK:   return %[[R]]
func.func @gather_broadcast(%base: memref<?xf32>,
                             %mask: vector<16xi1>, %passthru: vector<16xf32>) -> vector<16xf32> {
  %c0 = arith.constant 0 : index
  %indices = arith.constant dense<0> : vector<16xi32>
  %1 = vector.gather %base[%c0][%indices], %mask, %passthru :
    memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %1 : vector<16xf32>
}

// -----

// CHECK-LABEL: @contiguous_scatter
//  CHECK-SAME:   (%[[BASE:.*]]: memref<?xf32>, %[[MASK:.*]]: vector<16xi1>, %[[VALUE:.*]]: vector<16xf32>)
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   vector.maskedstore %[[BASE]][%[[C0]]], %[[MASK]], %[[VALUE]] : memref<?xf32>, vector<16xi1>, vector<16xf32>
func.func @contiguous_scatter(%base: memref<?xf32>,
                              %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %indices = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
  vector.scatter %base[%c0][%indices], %mask, %value :
    memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
  return
}

// -----

// CHECK-LABEL: @contiguous_scatter_const_mask
//  CHECK-SAME:   (%[[BASE:.*]]: memref<?xf32>, %[[VALUE:.*]]: vector<16xf32>)
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   vector.store %[[VALUE]], %[[BASE]][%[[C0]]] : memref<?xf32>, vector<16xf32>
func.func @contiguous_scatter_const_mask(%base: memref<?xf32>,
                                         %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %indices = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
  %mask = vector.constant_mask [16] : vector<16xi1>
  vector.scatter %base[%c0][%indices], %mask, %value :
    memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
  return
}

// -----

// CHECK-LABEL: @contiguous_scatter_step
//  CHECK-SAME:   (%[[BASE:.*]]: memref<?xf32>, %[[MASK:.*]]: vector<16xi1>, %[[VALUE:.*]]: vector<16xf32>)
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   vector.maskedstore %[[BASE]][%[[C0]]], %[[MASK]], %[[VALUE]] : memref<?xf32>, vector<16xi1>, vector<16xf32>
func.func @contiguous_scatter_step(%base: memref<?xf32>,
                                   %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %indices = vector.step : vector<16xindex>
  vector.scatter %base[%c0][%indices], %mask, %value :
    memref<?xf32>, vector<16xindex>, vector<16xi1>, vector<16xf32>
  return
}

// -----

// CHECK-LABEL: @fold_extract_constant_indices
//   CHECK-SAME:   %[[ARG:.*]]: vector<32x1xi32>) -> i32 {
//        CHECK:   %[[RES:.*]] = vector.extract %[[ARG]][0, 0] : i32 from vector<32x1xi32>
//        CHECK:   return %[[RES]] : i32
func.func @fold_extract_constant_indices(%arg : vector<32x1xi32>) -> i32 {
  %0 = arith.constant 0 : index
  %1 = vector.extract %arg[%0, %0] : i32 from vector<32x1xi32>
  return %1 : i32
}

// -----

// CHECK-LABEL: @fold_insert_constant_indices
//  CHECK-SAME:   %[[ARG:.*]]: vector<4x1xi32>) -> vector<4x1xi32> {
//       CHECK:   %[[VAL:.*]] = arith.constant 1 : i32
//       CHECK:   %[[RES:.*]] = vector.insert %[[VAL]], %[[ARG]] [0, 0] : i32 into vector<4x1xi32>
//       CHECK:   return %[[RES]] : vector<4x1xi32>
func.func @fold_insert_constant_indices(%arg : vector<4x1xi32>) -> vector<4x1xi32> {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : i32
  %res = vector.insert %1, %arg[%0, %0] : i32 into vector<4x1xi32>
  return %res : vector<4x1xi32>
}

// -----

// CHECK-LABEL: @fold_insert_use_chain(
//  CHECK-SAME:   %[[ARG:.*]]: vector<4x4xf32>,
//  CHECK-SAME:   %[[VAL:.*]]: f32,
//  CHECK-SAME:   %[[POS:.*]]: index) -> vector<4x4xf32> {
//  CHECK-NEXT:   %[[RES:.*]] = vector.insert %[[VAL]], %[[ARG]] {{\[}}%[[POS]], 0] : f32 into vector<4x4xf32>
//  CHECK-NEXT:   return %[[RES]] : vector<4x4xf32>
func.func @fold_insert_use_chain(%arg : vector<4x4xf32>, %val : f32, %pos: index) -> vector<4x4xf32> {
  %v_0 = vector.insert %val, %arg[%pos, 0] : f32 into vector<4x4xf32>
  %v_1 = vector.insert %val, %v_0[%pos, 0] : f32 into vector<4x4xf32>
  %v_2 = vector.insert %val, %v_1[%pos, 0] : f32 into vector<4x4xf32>
  return %v_2 : vector<4x4xf32>
}

// -----

// CHECK-LABEL: @no_fold_insert_use_chain_mismatch_static_position(
//  CHECK-SAME:   %[[ARG:.*]]: vector<4xf32>,
//  CHECK-SAME:   %[[VAL:.*]]: f32) -> vector<4xf32> {
//       CHECK:   %[[V_0:.*]] = vector.insert %[[VAL]], %[[ARG]] [0] : f32 into vector<4xf32>
//       CHECK:   %[[V_1:.*]] = vector.insert %[[VAL]], %[[V_0]] [1] : f32 into vector<4xf32>
//       CHECK:   return %[[V_1]] : vector<4xf32>
func.func @no_fold_insert_use_chain_mismatch_static_position(%arg : vector<4xf32>, %val : f32) -> vector<4xf32> {
  %v_0 = vector.insert %val, %arg[0] : f32 into vector<4xf32>
  %v_1 = vector.insert %val, %v_0[1] : f32 into vector<4xf32>
  return %v_1 : vector<4xf32>
}
