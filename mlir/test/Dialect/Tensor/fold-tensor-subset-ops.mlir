// RUN: mlir-opt -fold-tensor-subset-ops -split-input-file --allow-unregistered-dialect %s | FileCheck %s

func.func @fold_vector_transfer_read_with_rank_reduced_extract_slice(
    %arg0 : tensor<?x?x?xf32>,
    %arg1: index, %arg2 : index, %arg3 : index, %arg4: index, %arg5 : index,
    %arg6 : index) -> vector<4xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.extract_slice %arg0[0, %arg1, %arg2] [1, %arg3, %arg4] [1, 1, 1]
      : tensor<?x?x?xf32> to
        tensor<?x?xf32>
  %1 = vector.transfer_read %0[%arg5, %arg6], %cst {in_bounds = [true]}
      : tensor<?x?xf32>, vector<4xf32>
  return %1 : vector<4xf32>
}
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func @fold_vector_transfer_read_with_rank_reduced_extract_slice
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG5:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG6:[a-zA-Z0-9]+]]: index
//   CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:    %[[IDX0:.+]] = affine.apply #[[$MAP1]]()[%[[ARG1]], %[[ARG5]]]
//   CHECK-DAG:    %[[IDX1:.+]] = affine.apply #[[$MAP1]]()[%[[ARG2]], %[[ARG6]]]
//       CHECK:    vector.transfer_read %[[ARG0]][%[[C0]], %[[IDX0]], %[[IDX1]]], %{{.*}} : tensor<?x?x?xf32

// -----

// CHECK-LABEL: func.func @transfer_read_from_rank_reducing_extract_slice_failure
func.func @transfer_read_from_rank_reducing_extract_slice_failure(
    %src: tensor<1x8x8x8xf32>,
    %i1: index, %i2: index, %i3: index, %i4: index) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %f0 = arith.constant 0.000000e+00 : f32

  // Can't fold this atm since we don' emit the proper vector.extract_strided_slice.
//   CHECK: tensor.extract_slice
  %0 = tensor.extract_slice %src[0, %i1, %i2, %i3] [1, 4, 1, 4] [2, 3, 4, 5] : tensor<1x8x8x8xf32> to tensor<1x4x4xf32>
  %1 = vector.transfer_read %0[%c1, %i4, %c2], %f0 {in_bounds = [true]} : tensor<1x4x4xf32>, vector<4xf32>
  return %1 : vector<4xf32>
}

// -----

//   CHECK-DAG: #[[$ADD_4:.+]] = affine_map<()[s0] -> (s0 + 4)>

// CHECK-LABEL: func @transfer_read_of_extract_slice(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
//   CHECK-DAG:   %[[c8:.*]] = arith.constant 8 : index
//       CHECK:   %[[add:.*]] = affine.apply #[[$ADD_4]]()[%[[s1]]]
//       CHECK:   %[[r:.*]] = vector.transfer_read %[[t]][%[[c8]], %[[add]]], %{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<5x6xf32>
//       CHECK:   return %[[r]]
func.func @transfer_read_of_extract_slice(%t : tensor<?x?xf32>, %s1 : index, %s2 : index) -> vector<5x6xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.extract_slice %t[5, %s1] [10, %s2] [1, 1] : tensor<?x?xf32> to tensor<10x?xf32>
  %1 = vector.transfer_read %0[%c3, %c4], %cst {in_bounds = [true, true]} : tensor<10x?xf32>, vector<5x6xf32>
  return %1 : vector<5x6xf32>
}
// -----

func.func @fold_extract_slice_with_transfer_read_0d(
  %arg0 : tensor<12x32xf32>, %arg1 : index, %arg2 : index, %arg3 : index)
    -> vector<f32> {
  %f1 = arith.constant 1.0 : f32
  %0 = tensor.extract_slice %arg0[%arg1, %arg2][1, 1][1, 1] : tensor<12x32xf32> to tensor<f32>
  %1 = vector.transfer_read %0[], %f1 : tensor<f32>, vector<f32>
  return %1 : vector<f32>
}
//      CHECK: func @fold_extract_slice_with_transfer_read_0d
// CHECK-SAME:   %[[T:[a-zA-Z0-9_]+]]: tensor<12x32xf32>
// CHECK-SAME:   %[[SZ0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[SZ1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ST1:[a-zA-Z0-9_]+]]: index
//      CHECK:   vector.transfer_read %[[T]][%[[SZ0]], %[[SZ1]]]

// -----

//   CHECK-DAG: #[[$ADD_4:.+]] = affine_map<()[s0] -> (s0 + 4)>

// CHECK-LABEL: func @transfer_read_of_extract_slice(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
//   CHECK-DAG:   %[[c8:.*]] = arith.constant 8 : index
//       CHECK:   %[[add:.*]] = affine.apply #[[$ADD_4]]()[%[[s1]]]
//       CHECK:   %[[r:.*]] = vector.transfer_read %[[t]][%[[c8]], %[[add]]], %{{.*}} {in_bounds = [true]} : tensor<?x?xf32>, vector<6xf32>
//       CHECK:   return %[[r]]
func.func @transfer_read_of_extract_slice(%t : tensor<?x?xf32>, %s1 : index, %s2 : index) -> vector<6xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.extract_slice %t[5, %s1] [10, %s2] [1, 1] : tensor<?x?xf32> to tensor<10x?xf32>
  %1 = vector.transfer_read %0[%c3, %c4], %cst {in_bounds = [true]} : tensor<10x?xf32>, vector<6xf32>
  return %1 : vector<6xf32>
}

// -----

//   CHECK-DAG: #[[$ADD_3:.+]] = affine_map<()[s0] -> (s0 + 3)>

// CHECK-LABEL: func @transfer_read_of_extract_slice_rank_reducing(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
//   CHECK-DAG:   %[[c5:.*]] = arith.constant 5 : index
//   CHECK-DAG:   %[[c10:.*]] = arith.constant 10 : index
//       CHECK:   %[[add:.*]] = affine.apply #[[$ADD_3]]()[%[[s1]]]
//       CHECK:   %[[r:.*]] = vector.transfer_read %[[t]][%[[c5]], %[[add]], %[[c10]]], %{{.*}} {in_bounds = [true, true]} : tensor<?x?x?xf32>, vector<5x6xf32>
//       CHECK:   return %[[r]]
func.func @transfer_read_of_extract_slice_rank_reducing(%t : tensor<?x?x?xf32>, %s1 : index, %s2 : index) -> vector<5x6xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.extract_slice %t[5, %s1, 6] [1, %s2, 12] [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x12xf32>
  %1 = vector.transfer_read %0[%c3, %c4], %cst {in_bounds = [true, true]} : tensor<?x12xf32>, vector<5x6xf32>
  return %1 : vector<5x6xf32>
}

// -----

//   CHECK-DAG: #[[$ADD_4:.+]] = affine_map<()[s0] -> (s0 + 4)>
//   CHECK-DAG: #[[$d0d2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL: func @transfer_read_of_extract_slice_swappy_rank_reducing(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
func.func @transfer_read_of_extract_slice_swappy_rank_reducing(%t : tensor<?x?x?xf32>, %s1 : index, %s2 : index) -> vector<5x6xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32

//   CHECK-NOT:   extract_slice
//       CHECK:   %[[c8:.*]] = arith.constant 8 : index
//       CHECK:   %[[add:.*]] = affine.apply #[[$ADD_4]]()[%[[s2]]]
//       CHECK:   %[[r:.*]] = vector.transfer_read %[[t]][%[[c8]], %[[s1]], %[[add]]]
//  CHECK-SAME:     permutation_map = #[[$d0d2]]
//  CHECK-SAME:     tensor<?x?x?xf32>, vector<5x6xf32>
  %0 = tensor.extract_slice %t[5, %s1, %s2] [%s2, 1, 12] [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x12xf32>
  %1 = vector.transfer_read %0[%c3, %c4], %cst {in_bounds = [true, true]} : tensor<?x12xf32>, vector<5x6xf32>

  return %1 : vector<5x6xf32>
}

// -----

//       CHECK: func @fold_vector_transfer_write_with_rank_reduced_insert_slice
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: vector<4xf32>
//  CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG5:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG6:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG7:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG8:[a-zA-Z0-9]+]]: tensor<?x?xf32>
func.func @fold_vector_transfer_write_with_rank_reduced_insert_slice(
    %arg0 : tensor<?x?x?xf32>,
    %arg1 : vector<4xf32>, %arg2: index, %arg3 : index, %arg4 : index,
    %arg5: index, %arg6 : index, %arg7 : index,
    %st : tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  %cst = arith.constant 0.0 : f32

  //   CHECK-DAG:  %[[r1:.*]] = vector.transfer_write %[[ARG1]], %[[ARG8]][%[[ARG6]], %[[ARG7]]] {in_bounds = [true]} : vector<4xf32>, tensor<?x?xf32>
  //   CHECK-DAG:  %[[r2:.*]] = tensor.insert_slice %[[r1]] into %[[ARG0]][0, %[[ARG2]], %[[ARG3]]] [1, %[[ARG4]], %[[ARG5]]] [1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?xf32>
  %0 = vector.transfer_write %arg1, %st[%arg6, %arg7] {in_bounds = [true]}
      : vector<4xf32>, tensor<?x?xf32>
  %1 = tensor.insert_slice %0 into %arg0[0, %arg2, %arg3] [1, %arg4, %arg5] [1, 1, 1]
      : tensor<?x?xf32> into tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}

// -----

//       CHECK: func @fold_vector_transfer_write_with_inner_rank_reduced_insert_slice
//  CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]: vector<4xf32>
//  CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG5:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG6:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG7:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:    %[[ARG8:[a-zA-Z0-9]+]]: tensor<?x?xf32>
func.func @fold_vector_transfer_write_with_inner_rank_reduced_insert_slice(
    %arg0 : tensor<?x?x?xf32>,
    %arg1 : vector<4xf32>, %arg2: index, %arg3 : index, %arg4 : index,
    %arg5: index, %arg6 : index, %arg7 : index,
    %st : tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  %cst = arith.constant 0.0 : f32

  //   CHECK-DAG:  %[[r1:.*]] = vector.transfer_write %[[ARG1]], %[[ARG8]][%[[ARG6]], %[[ARG7]]] {in_bounds = [true]} : vector<4xf32>, tensor<?x?xf32>
  //   CHECK-DAG:  %[[r2:.*]] = tensor.insert_slice %[[r1]] into %[[ARG0]][%[[ARG2]], %[[ARG3]], 0] [%[[ARG4]], %[[ARG5]], 1] [1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?xf32>
  %0 = vector.transfer_write %arg1, %st[%arg6, %arg7] {in_bounds = [true]}
      : vector<4xf32>, tensor<?x?xf32>
  %1 = tensor.insert_slice %0 into %arg0[%arg2, %arg3, 0] [%arg4, %arg5, 1] [1, 1, 1]
      : tensor<?x?xf32> into tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_of_transfer_write(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x12xf32>, %[[v:.*]]: vector<5x6xf32>, %[[s:.*]]: index
func.func @insert_slice_of_transfer_write(%t1 : tensor<?x12xf32>, %v : vector<5x6xf32>, %s : index, %t2 : tensor<5x6xf32>) -> tensor<?x12xf32> {
  %c0 = arith.constant 0 : index

  //   CHECK-NOT: insert_slice
//       CHECK:   %[[c3:.*]] = arith.constant 3 : index
//       CHECK:   %[[r:.*]] = vector.transfer_write %[[v]], %[[t1]][%[[c3]], %[[s]]] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<?x12xf32>
//       CHECK:   return %[[r]]
  %0 = vector.transfer_write %v, %t2[%c0, %c0] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<5x6xf32>
  %1 = tensor.insert_slice %0 into %t1[3, %s] [5, 6] [1, 1] : tensor<5x6xf32> into tensor<?x12xf32>
  return %1 : tensor<?x12xf32>
}

// -----

// This test is negative since `transfer_write` only
// writes to `5x6` of the `100x100` elements of `%arg3`
// CHECK-LABEL: func @insert_slice_of_transfer_write_overwrite_all(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<1000x1000xf32>, %[[arg1:.*]]: vector<5x6xf32>, %[[arg2:.*]]: index, %[[arg3:.*]]: tensor<100x100xf32>
func.func @insert_slice_of_transfer_write_overwrite_all(%arg0: tensor<1000x1000xf32>, %arg1: vector<5x6xf32>, %arg2: index, %arg3: tensor<100x100xf32>) -> tensor<1000x1000xf32> {
  %c0 = arith.constant 0 : index

//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[r1:.*]] = vector.transfer_write %[[arg1]], %[[arg3]][%[[c0]], %[[c0]]] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<100x100xf32>
//       CHECK:   %[[r2:.*]] = tensor.insert_slice %[[r1]] into %[[arg0]][3, %[[arg2]]] [100, 100] [1, 1] : tensor<100x100xf32> into tensor<1000x1000xf32>
//       CHECK:   return %[[r2]] : tensor<1000x1000xf32>
  %0 = vector.transfer_write %arg1, %arg3[%c0, %c0] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<100x100xf32>
  %inserted_slice = tensor.insert_slice %0 into %arg0[3, %arg2] [100, 100] [1, 1] : tensor<100x100xf32> into tensor<1000x1000xf32>
  return %inserted_slice : tensor<1000x1000xf32>
}

// -----

//   CHECK-DAG: #[[$d0d2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL: func @insert_slice_of_transfer_write_swappy_rank_extending(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x?x12xf32>, %[[v:.*]]: vector<5x6xf32>, %[[s:.*]]: index
func.func @insert_slice_of_transfer_write_swappy_rank_extending(
    %t1 : tensor<?x?x12xf32>, %v : vector<5x6xf32>, 
    %s : index, %t2 : tensor<5x6xf32>) -> tensor<?x?x12xf32> {
  %c0 = arith.constant 0 : index

//   CHECK-NOT:   insert_slice
//   CHECK-DAG:   %[[c3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[c4:.*]] = arith.constant 4 : index
//       CHECK:   %[[r:.*]] = vector.transfer_write %[[v]], %[[t1]][%[[c4]], %[[c3]], %[[s]]]
//  CHECK-SAME:    {in_bounds = [true, true], permutation_map = #[[$d0d2]]} : vector<5x6xf32>, tensor<?x?x12xf32>
//       CHECK:   return %[[r]]
  %0 = vector.transfer_write %v, %t2[%c0, %c0] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<5x6xf32>
  %1 = tensor.insert_slice %0 into %t1[4, 3, %s] [5, 1, 6] [1, 1, 1] : tensor<5x6xf32> into tensor<?x?x12xf32>
  return %1 : tensor<?x?x12xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_of_transfer_write_rank_extending(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x?x12xf32>, %[[v:.*]]: vector<5x6xf32>, %[[s:.*]]: index
//   CHECK-DAG:   %[[c3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[c4:.*]] = arith.constant 4 : index
//       CHECK:   %[[r:.*]] = vector.transfer_write %[[v]], %[[t1]][%[[c4]], %[[c3]], %[[s]]] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<?x?x12xf32>
//       CHECK:   return %[[r]]
func.func @insert_slice_of_transfer_write_rank_extending(%t1 : tensor<?x?x12xf32>, %v : vector<5x6xf32>, %s : index, %t2 : tensor<5x6xf32>) -> tensor<?x?x12xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_write %v, %t2[%c0, %c0] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<5x6xf32>
  %1 = tensor.insert_slice %0 into %t1[4, 3, %s] [1, 5, 6] [1, 1, 1] : tensor<5x6xf32> into tensor<?x?x12xf32>
  return %1 : tensor<?x?x12xf32>
}

// -----

//       CHECK: #[[$map:.*]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-LABEL: func @insert_slice_of_insert_slice(
//  CHECK-SAME:     %[[t:[0-9a-z]*]]: tensor<f32>
//  CHECK-SAME:     %[[r1:[0-9a-z]*]]: tensor<1x14xf32>
//  CHECK-SAME:     %[[pos:[0-9a-z]*]]: index
//       CHECK:   %[[add:.*]] = affine.apply #[[$map]]()[%[[pos]]]
//       CHECK:   tensor.insert_slice %[[t]] into %[[r1]][4, %[[add]]] [1, 1] [1, 1] : tensor<f32> into tensor<1x14xf32>
func.func @insert_slice_of_insert_slice(%t: tensor<f32>, %r0: tensor<1x1xf32>, %r1: tensor<1x14xf32>, %pos: index)
    -> tensor<1x14xf32> 
{
  %0 = tensor.insert_slice %t into %r0[1, 2] [1, 1] [1, 1] 
    : tensor<f32> into tensor<1x1xf32>
  %1 = tensor.insert_slice %0 into %r1[3, %pos] [1, 1] [1, 1] 
    : tensor<1x1xf32> into tensor<1x14xf32>
  return %1 : tensor<1x14xf32>
}

// -----

//   CHECK-DAG: #[[$map:.*]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-LABEL: func @insert_slice_of_insert_slice(
//  CHECK-SAME:     %[[t:[0-9a-z]*]]: tensor<f32>
//  CHECK-SAME:     %[[r1:[0-9a-z]*]]: tensor<1x14xf32>
//  CHECK-SAME:     %[[pos:[0-9a-z]*]]: index
//       CHECK:   %[[composed_pos:.+]] = affine.apply #[[$map]]()[%[[pos]]]
//       CHECK:   tensor.insert_slice %[[t]] into %[[r1]][3, %[[composed_pos]]] [1, 1] [1, 1] : tensor<f32> into tensor<1x14xf32>
func.func @insert_slice_of_insert_slice(%t: tensor<f32>, %r0: tensor<1xf32>, %r1: tensor<1x14xf32>, %pos: index)
    -> tensor<1x14xf32> 
{
  %0 = tensor.insert_slice %t into %r0[2] [1] [1] 
    : tensor<f32> into tensor<1xf32>
  %1 = tensor.insert_slice %0 into %r1[3, %pos] [1, 1] [1, 1] 
    : tensor<1xf32> into tensor<1x14xf32>
  return %1 : tensor<1x14xf32>
}

// -----

// This test fails to fold because the size `4` and `%pos` do not match: 
// this requires a copy
// CHECK-LABEL: func @fail_insert_slice_of_insert_slice(
//       CHECK:   tensor.insert_slice
//       CHECK:   tensor.insert_slice
func.func @fail_insert_slice_of_insert_slice(
  %t: tensor<4xf32>, %r0: tensor<?xf32>, %r1: tensor<?x?xf32>, %pos: index)
    -> tensor<?x?xf32> 
{
  %0 = tensor.insert_slice %t into %r0[%pos] [4] [1] 
    : tensor<4xf32> into tensor<?xf32>
  %1 = tensor.insert_slice %0 into %r1[%pos, 423] [%pos, 1] [1, 1] 
    : tensor<?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// Here the sizes are the same and the folding occurs properly.
//       CHECK: #[[$map:.*]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK-LABEL: func @insert_slice_of_insert_slice_dynamic(
//  CHECK-SAME:     %[[t:[0-9a-z]*]]: tensor<?xf32>
//  CHECK-SAME:     %[[r0:[0-9a-z]*]]: tensor<?xf32>
//  CHECK-SAME:     %[[r1:[0-9a-z]*]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[pos:[0-9a-z]*]]: index
//       CHECK:   %[[add:.*]] = affine.apply #[[$map]]()[%[[pos]]]
//       CHECK:   tensor.insert_slice %[[t]] into %[[r1]][%[[add]], 423] [%[[pos]], 1] [1, 1] : tensor<?xf32> into tensor<?x?xf32>
func.func @insert_slice_of_insert_slice_dynamic(
  %t: tensor<?xf32>, %r0: tensor<?xf32>, %r1: tensor<?x?xf32>, %pos: index)
    -> tensor<?x?xf32> 
{
  %0 = tensor.insert_slice %t into %r0[%pos] [%pos] [1] 
    : tensor<?xf32> into tensor<?xf32>
  %1 = tensor.insert_slice %0 into %r1[%pos, 423] [%pos, 1] [1, 1] 
    : tensor<?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// Here the sizes are the same and the folding occurs properly.
//       CHECK: #[[$map:.*]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK-LABEL: func @insert_slice_of_insert_slice_dynamic(
//  CHECK-SAME:     %[[t:[0-9a-z]*]]: tensor<?xf32>
//  CHECK-SAME:     %[[r0:[0-9a-z]*]]: tensor<?xf32>
//  CHECK-SAME:     %[[r1:[0-9a-z]*]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[pos:[0-9a-z]*]]: index
//       CHECK:   %[[add:.*]] = affine.apply #[[$map]]()[%[[pos]]]
//       CHECK:   tensor.insert_slice %[[t]] into %[[r1]][%[[add]], 423] [%[[pos]], 1] [1, 1] : tensor<?xf32> into tensor<?x?xf32>
func.func @insert_slice_of_insert_slice_dynamic(
  %t: tensor<?xf32>, %r0: tensor<?xf32>, %r1: tensor<?x?xf32>, %pos: index)
    -> tensor<?x?xf32> 
{
  %0 = tensor.insert_slice %t into %r0[%pos] [%pos] [1] 
    : tensor<?xf32> into tensor<?xf32>
  %1 = tensor.insert_slice %0 into %r1[%pos, 423] [%pos, 1] [1, 1] 
    : tensor<?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

//       CHECK: #[[$map:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-LABEL: func @parallel_insert_slice_of_insert_slice_dynamic(
//  CHECK-SAME:   %[[t:[0-9a-z]*]]: tensor<12x34xf32>
//  CHECK-SAME:   %[[o0:[0-9a-z]*]]: index
//  CHECK-SAME:   %[[o1:[0-9a-z]*]]: index
//  CHECK-SAME:   %[[sz0:[0-9a-z]*]]: index
//  CHECK-SAME:   %[[sz1:[0-9a-z]*]]: index
func.func @parallel_insert_slice_of_insert_slice_dynamic(
    %t: tensor<12x34xf32>, %o0: index, %o1: index, %sz0: index, %sz1: index) 
      -> tensor<12x34xf32>{

  // CHECK: scf.forall {{.*}} shared_outs(%[[out:.*]] = %[[t]]
  %0 = scf.forall (%arg0, %arg1) in (27, 8) shared_outs(%arg2 = %t) -> (tensor<12x34xf32>) {
    // CHECK: %[[tt:.*]] = "make_me_a_tensor"() : () -> tensor<?x?xf32>
    %tt = "make_me_a_tensor"() : () -> tensor<?x?xf32>
    %tt2 = "make_me_another_tensor"() : () -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %tt into %tt2[%o1, 0] [%sz0, %sz1] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

    //      CHECK: %[[add:.*]] = affine.apply #[[$map]]()[%[[o0]], %[[o1]]]
    //      CHECK: scf.forall.in_parallel
    //      CHECK:   tensor.parallel_insert_slice %[[tt]] into %[[out]][%[[add]], %[[o1]]] [%[[sz0]], %[[sz1]]] [1, 1]
    // CHECK-SAME:     : tensor<?x?xf32> into tensor<12x34xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %inserted_slice into %arg2[%o0, %o1] [%sz0, %sz1] [1, 1]
        : tensor<?x?xf32> into tensor<12x34xf32>
    }
  }
  return %0: tensor<12x34xf32>
}
