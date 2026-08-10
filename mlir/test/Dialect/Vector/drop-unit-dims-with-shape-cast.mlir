// RUN: mlir-opt %s -test-vector-transfer-flatten-patterns -split-input-file | FileCheck %s

///----------------------------------------------------------------------------------------
/// [Pattern: DropUnitDimFromElementwiseOps]
///----------------------------------------------------------------------------------------

func.func @fold_unit_dim_add_basic(%vec : vector<1x8xi32>) -> vector<1x8xi32> {
   %res = arith.addi %vec, %vec : vector<1x8xi32>
   return %res : vector<1x8xi32>
}
// CHECK-LABEL:   func.func @fold_unit_dim_add_basic(
// CHECK-SAME:      %[[VAL_0:.*]]: vector<1x8xi32>) -> vector<1x8xi32> {
// CHECK:           %[[VAL_1:.*]] = vector.shape_cast %[[VAL_0]] : vector<1x8xi32> to vector<8xi32>
// CHECK:           %[[VAL_2:.*]] = vector.shape_cast %[[VAL_0]] : vector<1x8xi32> to vector<8xi32>
// CHECK:           %[[VAL_3:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : vector<8xi32>
// CHECK:           %[[VAL_4:.*]] = vector.shape_cast %[[VAL_3]] : vector<8xi32> to vector<1x8xi32>
// CHECK:           return %[[VAL_4]] : vector<1x8xi32>

// -----

func.func @fold_unit_dim_add_leading_and_trailing(%vec : vector<1x8x1xi32>) -> vector<1x8x1xi32> {
   %res = arith.addi %vec, %vec : vector<1x8x1xi32>
   return %res : vector<1x8x1xi32>
}
// CHECK-LABEL:   func.func @fold_unit_dim_add_leading_and_trailing(
// CHECK-SAME:      %[[VAL_0:.*]]: vector<1x8x1xi32>) -> vector<1x8x1xi32> {
// CHECK:           %[[VAL_1:.*]] = vector.shape_cast %[[VAL_0]] : vector<1x8x1xi32> to vector<8xi32>
// CHECK:           %[[VAL_2:.*]] = vector.shape_cast %[[VAL_0]] : vector<1x8x1xi32> to vector<8xi32>
// CHECK:           %[[VAL_3:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : vector<8xi32>
// CHECK:           %[[VAL_4:.*]] = vector.shape_cast %[[VAL_3]] : vector<8xi32> to vector<1x8x1xi32>
// CHECK:           return %[[VAL_4]] : vector<1x8x1xi32>

// -----

func.func @fold_unit_dim_add(%vec_0 : vector<8x1xi32>,
                             %vec_1 : vector<1x8xi32>) -> vector<8xi32> {
   %sc_vec_0 = vector.shape_cast %vec_0 : vector<8x1xi32> to vector<1x8xi32>
   %add = arith.addi %sc_vec_0, %vec_1 : vector<1x8xi32>
   %res = vector.shape_cast %add : vector<1x8xi32> to vector<8xi32>
   return %res : vector<8xi32>
}

// CHECK-LABEL:   func.func @fold_unit_dim_add(
// CHECK-SAME:      %[[VAL_0:.*]]: vector<8x1xi32>,
// CHECK-SAME:      %[[VAL_1:.*]]: vector<1x8xi32>) -> vector<8xi32> {
// CHECK:           %[[VAL_2:.*]] = vector.shape_cast %[[VAL_0]] : vector<8x1xi32> to vector<8xi32>
// CHECK:           %[[VAL_3:.*]] = vector.shape_cast %[[VAL_1]] : vector<1x8xi32> to vector<8xi32>
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : vector<8xi32>
// CHECK:           return %[[VAL_4]] : vector<8xi32>

// -----

func.func @fold_unit_dim_mulf(%vec_0 : vector<8x[2]x1xf32>,
                              %vec_1 : vector<1x8x[2]xf32>) -> vector<8x[2]xf32> {
   %sc_vec_0 = vector.shape_cast %vec_0 : vector<8x[2]x1xf32> to vector<1x8x[2]xf32>
   %add = arith.mulf %sc_vec_0, %vec_1 : vector<1x8x[2]xf32>
   %res = vector.shape_cast %add : vector<1x8x[2]xf32> to vector<8x[2]xf32>
   return %res : vector<8x[2]xf32>
}

// CHECK-LABEL:   func.func @fold_unit_dim_mulf(
// CHECK-SAME:      %[[VAL_0:.*]]: vector<8x[2]x1xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: vector<1x8x[2]xf32>) -> vector<8x[2]xf32> {
// CHECK:           %[[VAL_2:.*]] = vector.shape_cast %[[VAL_0]] : vector<8x[2]x1xf32> to vector<8x[2]xf32>
// CHECK:           %[[VAL_3:.*]] = vector.shape_cast %[[VAL_1]] : vector<1x8x[2]xf32> to vector<8x[2]xf32>
// CHECK:           %[[VAL_4:.*]] = arith.mulf %[[VAL_2]], %[[VAL_3]] : vector<8x[2]xf32>
// CHECK:           return %[[VAL_4]] : vector<8x[2]xf32>

// -----

func.func @fold_unit_dim_sitofp(%vec : vector<8x[2]x1xi8>) -> vector<8x[2]xf32> {
   %sc_vec_0 = vector.shape_cast %vec : vector<8x[2]x1xi8> to vector<1x8x[2]xi8>
   %add = arith.sitofp %sc_vec_0 : vector<1x8x[2]xi8> to vector<1x8x[2]xf32>
   %res = vector.shape_cast %add : vector<1x8x[2]xf32> to vector<8x[2]xf32>
   return %res : vector<8x[2]xf32>
}

// CHECK-LABEL:   func.func @fold_unit_dim_sitofp(
// CHECK-SAME:      %[[VAL_0:.*]]: vector<8x[2]x1xi8>) -> vector<8x[2]xf32> {
// CHECK:           %[[VAL_1:.*]] = vector.shape_cast %[[VAL_0]] : vector<8x[2]x1xi8> to vector<8x[2]xi8>
// CHECK:           %[[VAL_2:.*]] = arith.sitofp %[[VAL_1]] : vector<8x[2]xi8> to vector<8x[2]xf32>
// CHECK:           return %[[VAL_2]] : vector<8x[2]xf32>

// -----

// All shape casts are folded away

func.func @fold_unit_dims_entirely(%vec_0 : vector<8xi32>,
                                   %vec_1 : vector<8xi32>,
                                   %vec_2 : vector<8xi32>) -> vector<8xi32> {
   %sc_vec_0 = vector.shape_cast %vec_0 : vector<8xi32> to vector<1x8xi32>
   %sc_vec_1 = vector.shape_cast %vec_1 : vector<8xi32> to vector<1x8xi32>
   %sc_vec_2 = vector.shape_cast %vec_2 : vector<8xi32> to vector<1x8xi32>
   %mul = arith.muli %sc_vec_0, %sc_vec_1 : vector<1x8xi32>
   %add = arith.addi %mul, %sc_vec_2 : vector<1x8xi32>
   %res = vector.shape_cast %add : vector<1x8xi32> to vector<8xi32>
   return %res : vector<8xi32>
}

// CHECK-LABEL:   func.func @fold_unit_dims_entirely(
// CHECK-SAME:      %[[VAL_0:.*]]: vector<8xi32>, %[[VAL_1:.*]]: vector<8xi32>,
// CHECK-SAME:      %[[VAL_2:.*]]: vector<8xi32>) -> vector<8xi32> {
// CHECK:           %[[VAL_3:.*]] = arith.muli %[[VAL_0]], %[[VAL_1]] : vector<8xi32>
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_3]], %[[VAL_2]] : vector<8xi32>
// CHECK:           return %[[VAL_4]] : vector<8xi32>

// -----

func.func @fold_inner_unit_dim(%vec_0 : vector<8x1x3xf128>,
                               %vec_1 : vector<1x8x3xf128>) -> vector<8x3xf128> {
   %sc_vec_1 = vector.shape_cast %vec_1 : vector<1x8x3xf128> to vector<8x1x3xf128>
   %mul = arith.mulf %vec_0, %sc_vec_1 : vector<8x1x3xf128>
   %res = vector.shape_cast %mul : vector<8x1x3xf128> to vector<8x3xf128>
   return %res : vector<8x3xf128>
}

// CHECK-LABEL: func.func @fold_inner_unit_dim(
// CHECK-SAME:    %[[VAL_0:.*]]: vector<8x1x3xf128>,
// CHECK-SAME:    %[[VAL_1:.*]]: vector<1x8x3xf128>) -> vector<8x3xf128> {
// CHECK:         %[[VAL_2:.*]] = vector.shape_cast %[[VAL_0]] : vector<8x1x3xf128> to vector<8x3xf128>
// CHECK:         %[[VAL_3:.*]] = vector.shape_cast %[[VAL_1]] : vector<1x8x3xf128> to vector<8x3xf128>
// CHECK:         %[[VAL_4:.*]] = arith.mulf %[[VAL_2]], %[[VAL_3]] : vector<8x3xf128>
// CHECK:         return %[[VAL_4]] : vector<8x3xf128>

// -----

func.func @fold_inner_unit_dim_scalable(%vec_0 : vector<8x1x[1]x3xf128>,
                                        %vec_1 : vector<1x8x[1]x3xf128>) -> vector<8x[1]x3xf128> {
   %sc_vec_1 = vector.shape_cast %vec_1 : vector<1x8x[1]x3xf128> to vector<8x1x[1]x3xf128>
   %mul = arith.mulf %vec_0, %sc_vec_1 : vector<8x1x[1]x3xf128>
   %res = vector.shape_cast %mul : vector<8x1x[1]x3xf128> to vector<8x[1]x3xf128>
   return %res : vector<8x[1]x3xf128>
}

// CHECK-LABEL: func.func @fold_inner_unit_dim_scalable(
// CHECK-SAME:    %[[VAL_0:.*]]: vector<8x1x[1]x3xf128>,
// CHECK-SAME:    %[[VAL_1:.*]]: vector<1x8x[1]x3xf128>) -> vector<8x[1]x3xf128> {
// CHECK:         %[[VAL_2:.*]] = vector.shape_cast %[[VAL_0]] : vector<8x1x[1]x3xf128> to vector<8x[1]x3xf128>
// CHECK:         %[[VAL_3:.*]] = vector.shape_cast %[[VAL_1]] : vector<1x8x[1]x3xf128> to vector<8x[1]x3xf128>
// CHECK:         %[[VAL_4:.*]] = arith.mulf %[[VAL_2]], %[[VAL_3]] : vector<8x[1]x3xf128>
// CHECK:         return %[[VAL_4]] : vector<8x[1]x3xf128>

// -----

func.func @fold_all_unit_dims(%vec: vector<1x1xf32>) -> vector<1xf32> {
  %0 = arith.mulf %vec, %vec : vector<1x1xf32>
  %res = vector.shape_cast %0 : vector<1x1xf32> to vector<1xf32>
  return %res : vector<1xf32>
}

// CHECK-LABEL: func.func @fold_all_unit_dims(
// CHECK-SAME:    %[[VAL_0:.*]]: vector<1x1xf32>) -> vector<1xf32>
// CHECK:         %[[VAL_1:.*]] = vector.shape_cast %[[VAL_0]] : vector<1x1xf32> to vector<1xf32>
// CHECK:         %[[VAL_2:.*]] = vector.shape_cast %[[VAL_0]] : vector<1x1xf32> to vector<1xf32>
// CHECK:         %[[VAL_3:.*]] = arith.mulf %[[VAL_1]], %[[VAL_2]] : vector<1xf32>
// CHECK:         return %[[VAL_3]] : vector<1xf32>

///----------------------------------------------------------------------------------------
/// [Pattern: DropUnitDimsFromTransposeOp]
///----------------------------------------------------------------------------------------

func.func @transpose_with_internal_unit_dims(%vec: vector<1x1x4x[4]xf32>) -> vector<[4]x1x1x4xf32> {
  %res = vector.transpose %vec, [3, 0, 1, 2] : vector<1x1x4x[4]xf32> to vector<[4]x1x1x4xf32>
  return %res : vector<[4]x1x1x4xf32>
}

// CHECK-LABEL: func.func @transpose_with_internal_unit_dims(
// CHECK-SAME:                                               %[[VEC:.*]]: vector<1x1x4x[4]xf32>)
// CHECK-NEXT:    %[[DROP_DIMS:.*]] = vector.shape_cast %arg0 : vector<1x1x4x[4]xf32> to vector<4x[4]xf32>
// CHECK-NEXT:    %[[TRANSPOSE:.*]] = vector.transpose %0, [1, 0] : vector<4x[4]xf32> to vector<[4]x4xf32>
// CHECK-NEXT:    %[[RESTORE_DIMS:.*]] = vector.shape_cast %1 : vector<[4]x4xf32> to vector<[4]x1x1x4xf32>
// CHECK-NEXT:    return %[[RESTORE_DIMS]] : vector<[4]x1x1x4xf32>

// -----

func.func @transpose_with_scalable_unit_dims(%vec: vector<[1]x1x2x4x1xf32>) -> vector<1x1x4x2x[1]xf32>
{
  %res = vector.transpose %vec, [4, 1, 3, 2, 0]  : vector<[1]x1x2x4x1xf32> to vector<1x1x4x2x[1]xf32>
  return %res: vector<1x1x4x2x[1]xf32>
}

// CHECK-LABEL: func.func @transpose_with_scalable_unit_dims(
// CHECK-SAME:                                               %[[VEC:.*]]: vector<[1]x1x2x4x1xf32>)
// CHECK-NEXT:    %[[DROP_DIMS:.*]] = vector.shape_cast %[[VEC]] : vector<[1]x1x2x4x1xf32> to vector<[1]x2x4xf32>
// CHECK-NEXT:    %[[TRANSPOSE:.*]] = vector.transpose %[[DROP_DIMS]], [2, 1, 0] : vector<[1]x2x4xf32> to vector<4x2x[1]xf32>
// CHECK-NEXT:    %[[RESTORE_DIMS:.*]] = vector.shape_cast %[[TRANSPOSE]] : vector<4x2x[1]xf32> to vector<1x1x4x2x[1]xf32>
// CHECK-NEXT:    return %[[RESTORE_DIMS]] : vector<1x1x4x2x[1]xf32>

// -----

func.func @transpose_with_all_unit_dims(%vec: vector<1x1x1xf32>) -> vector<1x1x1xf32> {
  %res = vector.transpose %vec, [0, 2, 1] : vector<1x1x1xf32> to vector<1x1x1xf32>
  return %res : vector<1x1x1xf32>
}
// The `vec` is returned because there are other flattening patterns that fold
// vector.shape_cast ops away.
// CHECK-LABEL: func.func @transpose_with_all_unit_dims
// CHECK-SAME:      %[[VEC:.[a-zA-Z0-9]+]]
// CHECK-NEXT:    return %[[VEC]]

// -----

func.func @negative_transpose_with_no_unit_dims(%vec: vector<4x2x3xf32>) -> vector<4x3x2xf32> {
  %res = vector.transpose %vec, [0, 2, 1] : vector<4x2x3xf32> to vector<4x3x2xf32>
  return %res : vector<4x3x2xf32>
}

// CHECK-LABEL: func.func @negative_transpose_with_no_unit_dims
// CHECK-NOT: vector.shape_cast

// -----

///----------------------------------------------------------------------------------------
/// [Pattern: DropUnitDimsFromScfForOp]
///----------------------------------------------------------------------------------------

func.func @scf_for_with_internal_unit_dims(%vec: vector<4x1x1x[4]xf32>) -> vector<4x1x1x[4]xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res = scf.for %i = %c0 to %c4 step %c1 iter_args(%iter = %vec) -> vector<4x1x1x[4]xf32> {
    %s = math.sqrt %iter : vector<4x1x1x[4]xf32>
    scf.yield %s : vector<4x1x1x[4]xf32>
  }
  return %res : vector<4x1x1x[4]xf32>
}

// CHECK-LABEL: func.func @scf_for_with_internal_unit_dims
//  CHECK-SAME:   %[[VEC:[A-Za-z0-9]+]]: vector<4x1x1x[4]xf32>
//       CHECK:   %[[CAST:.+]] = vector.shape_cast %[[VEC]] : vector<4x1x1x[4]xf32> to vector<4x[4]xf32>
//       CHECK:   %[[LOOP:.+]] = scf.for {{.*}} iter_args(%[[ITER:.+]] = %[[CAST]])
//       CHECK:     %[[SQRT:.+]] = math.sqrt %[[ITER]] : vector<4x[4]xf32>
//       CHECK:     scf.yield %[[SQRT]]
//       CHECK:   %[[CASTBACK:.+]] = vector.shape_cast %[[LOOP]] : vector<4x[4]xf32> to vector<4x1x1x[4]xf32>
//       CHECK:   return %[[CASTBACK]]

// -----

func.func @scf_for_with_all_unit_dims(%vec: vector<1x1xf32>) -> vector<1x1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res = scf.for %i = %c0 to %c4 step %c1 iter_args(%iter = %vec) -> vector<1x1xf32> {
    %s = math.sqrt %iter : vector<1x1xf32>
    scf.yield %s : vector<1x1xf32>
  }
  return %res : vector<1x1xf32>
}

// CHECK-LABEL: func.func @scf_for_with_all_unit_dims
//  CHECK-SAME:   %[[VEC:[A-Za-z0-9]+]]: vector<1x1xf32>
//       CHECK:   %[[CAST:.+]] = vector.shape_cast %[[VEC]] : vector<1x1xf32> to vector<1xf32>
//       CHECK:   %[[LOOP:.+]] = scf.for {{.*}} iter_args(%[[ITER:.+]] = %[[CAST]])
//       CHECK:     %[[SQRT:.+]] = math.sqrt %[[ITER]] : vector<1xf32>
//       CHECK:     scf.yield %[[SQRT]]
//       CHECK:   %[[CASTBACK:.+]] = vector.shape_cast %[[LOOP]] : vector<1xf32> to vector<1x1xf32>
//       CHECK:   return %[[CASTBACK]]

// -----

func.func @scf_for_with_multiple_operands(%idx: index, %vec0: vector<1x4xf32>, %vec1: vector<1x4xf32>) -> vector<1x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:3 = scf.for %i = %c0 to %c4 step %c1
    iter_args(%id = %idx, %iter0 = %vec0, %iter1 = %vec1) -> (index, vector<1x4xf32>, vector<1x4xf32>) {
    %add = arith.addf %iter0, %iter1 : vector<1x4xf32>
    scf.yield %id, %add, %add : index, vector<1x4xf32>, vector<1x4xf32>
  }
  return %res#1 : vector<1x4xf32>
}

// CHECK-LABEL: func.func @scf_for_with_multiple_operands
//  CHECK-SAME:   %[[IDX:[A-Za-z0-9]+]]: index
//  CHECK-SAME:   %[[VEC0:[A-Za-z0-9]+]]: vector<1x4xf32>
//  CHECK-SAME:   %[[VEC1:[A-Za-z0-9]+]]: vector<1x4xf32>
//   CHECK-DAG:   %[[CAST0:.+]] = vector.shape_cast %[[VEC0]] : vector<1x4xf32> to vector<4xf32>
//   CHECK-DAG:   %[[CAST1:.+]] = vector.shape_cast %[[VEC1]] : vector<1x4xf32> to vector<4xf32>
//       CHECK:   %[[LOOP:.+]]:3 = scf.for
//  CHECK-SAME:     iter_args(%{{.*}} = %[[IDX]], %[[ITER0:.+]] = %[[CAST0]], %[[ITER1:.+]] = %[[CAST1]])
//       CHECK:     %[[ADD:.+]] = arith.addf %[[ITER0]], %[[ITER1]] : vector<4xf32>
//       CHECK:     scf.yield %{{.*}}, %[[ADD]], %[[ADD]]
//       CHECK:   %[[CASTBACK:.+]] = vector.shape_cast %[[LOOP]]#1 : vector<4xf32> to vector<1x4xf32>
//       CHECK:   return %[[CASTBACK]]

// -----

func.func @scf_for_with_scalable_unit_dims(%vec: vector<1x[1]xf32>) -> vector<1x[1]xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res = scf.for %i = %c0 to %c4 step %c1 iter_args(%iter = %vec) -> vector<1x[1]xf32> {
    %s = math.sqrt %iter : vector<1x[1]xf32>
    scf.yield %s : vector<1x[1]xf32>
  }
  return %res : vector<1x[1]xf32>
}

// CHECK-LABEL: func.func @scf_for_with_scalable_unit_dims
//  CHECK-SAME:   %[[VEC:[A-Za-z0-9]+]]: vector<1x[1]xf32>
//       CHECK:   %[[CAST:.+]] = vector.shape_cast %[[VEC]] : vector<1x[1]xf32> to vector<[1]xf32>
//       CHECK:   %[[LOOP:.+]] = scf.for {{.*}} iter_args(%[[ITER:.+]] = %[[CAST]])
//       CHECK:     %[[SQRT:.+]] = math.sqrt %[[ITER]] : vector<[1]xf32>
//       CHECK:     scf.yield %[[SQRT]]
//       CHECK:   %[[CASTBACK:.+]] = vector.shape_cast %[[LOOP]] : vector<[1]xf32> to vector<1x[1]xf32>
//       CHECK:   return %[[CASTBACK]]
