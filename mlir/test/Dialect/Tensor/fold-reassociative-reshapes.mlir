// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-reassociative-reshape-folding %s | FileCheck %s

// CHECK-LABEL: func @expand_shape_of_rank_reducing_extract(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?x?x?xf32>
//   CHECK-DAG:   %[[extract1:.*]] = tensor.extract_slice %{{.*}}[0, 0, 0, 0]
//   CHECK-SAME:    [%{{.*}}, 1, 1, 5] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x1x1x5xf32>
//   CHECK-DAG:   %[[extract2:.*]] = tensor.extract_slice %{{.*}}[0, 0, 0, 0]
//   CHECK-SAME:    [%{{.*}}, 1, 1, 5] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x1x1x5xf32>
//       CHECK:   return %[[extract1]], %[[extract2]]
func.func @expand_shape_of_rank_reducing_extract(
    %t: tensor<?x?x?x?xf32>, %idx: index)
  -> (tensor<?x1x1x5xf32>, tensor<?x1x1x5xf32>)
{
  %0 = tensor.extract_slice %t[0, 0, 0, 0][%idx, 1, 1, 5][1, 1, 1, 1]
      : tensor<?x?x?x?xf32> to tensor<?x1x5xf32>
  %c0 = arith.constant 0 : index
  %sz0 = tensor.dim %0, %c0 : tensor<?x1x5xf32>
  %1 = tensor.expand_shape %0 [[0], [1, 2], [3]] output_shape [%sz0, 1, 1, 5]
      : tensor<?x1x5xf32> into tensor<?x1x1x5xf32>
  %2 = tensor.expand_shape %0 [[0, 1], [2], [3]] output_shape [%sz0, 1, 1, 5]
      : tensor<?x1x5xf32> into tensor<?x1x1x5xf32>
  return %1, %2 : tensor<?x1x1x5xf32>, tensor<?x1x1x5xf32>
}

// -----

// CHECK-LABEL: func @unpadding_collapse_of_extract_slice(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[x:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[y:[a-zA-Z0-9_]+]]: index
//       CHECK:   %[[extract:.*]] = tensor.extract_slice %[[t]][%[[x]], %[[y]], 0, 0]
//  CHECK-SAME:     [1, %{{.*}}, 1, %{{.*}}] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?xf32>
//       CHECK:   return %[[extract]]
func.func @unpadding_collapse_of_extract_slice(
    %t: tensor<?x?x?x?xf32>, %x: index, %y: index)
  -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %sz0 = tensor.dim %t, %c1 : tensor<?x?x?x?xf32>
  %sz1 = tensor.dim %t, %c3 : tensor<?x?x?x?xf32>
  %0 = tensor.extract_slice %t[%x, %y, 0, 0] [1, %sz0, 1, %sz1] [1, 1, 1, 1]
      : tensor<?x?x?x?xf32> to tensor<1x?x1x?xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2, 3]]
      : tensor<1x?x1x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @non_unpadding_collapse_of_extract_slice(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[x:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[y:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[sz:[a-zA-Z0-9_]+]]: index
//       CHECK:   %[[extract:.*]] = tensor.extract_slice %[[t]][%[[x]], %[[y]], 0, 0]
//  CHECK-SAME:     [%{{.*}}, %{{.*}}, %[[sz]], 1] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?x?xf32>
//       CHECK:   %[[collapse:.*]] = tensor.collapse_shape %[[extract]] {{\[}}[0], [1, 2]] : tensor<?x?x?xf32> into tensor<?x?xf32>
//       CHECK:   return %[[collapse]]
func.func @non_unpadding_collapse_of_extract_slice(
    %t: tensor<?x?x?x?xf32>, %x: index, %y: index, %sz: index)
  -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sz0 = tensor.dim %t, %c0 : tensor<?x?x?x?xf32>
  %sz1 = tensor.dim %t, %c1 : tensor<?x?x?x?xf32>
  %0 = tensor.extract_slice %t[%x, %y, 0, 0] [%sz0, %sz1, %sz, 1] [1, 1, 1, 1]
      : tensor<?x?x?x?xf32> to tensor<?x?x?xf32>
  %1 = tensor.collapse_shape %0 [[0], [1, 2]]
      : tensor<?x?x?xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @unpadding_collapse_of_extract_slice_with_multiple_users(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[x:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[y:[a-zA-Z0-9_]+]]: index
//       CHECK:   %[[extract:.*]] = tensor.extract_slice %[[t]][%[[x]], %[[y]], 0, 0]
//  CHECK-SAME:     [1, %{{.*}}, 1, %{{.*}}] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<1x?x1x?xf32>
//       CHECK:   %[[collapse:.*]] = tensor.collapse_shape %[[extract]] {{\[}}[0, 1], [2, 3]] : tensor<1x?x1x?xf32> into tensor<?x?xf32>
//       CHECK:   return %[[extract]], %[[collapse]]
func.func @unpadding_collapse_of_extract_slice_with_multiple_users(
    %t: tensor<?x?x?x?xf32>, %x: index, %y: index)
  -> (tensor<1x?x1x?xf32>, tensor<?x?xf32>) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %sz0 = tensor.dim %t, %c1 : tensor<?x?x?x?xf32>
  %sz1 = tensor.dim %t, %c3 : tensor<?x?x?x?xf32>
  %0 = tensor.extract_slice %t[%x, %y, 0, 0] [1, %sz0, 1, %sz1] [1, 1, 1, 1]
      : tensor<?x?x?x?xf32> to tensor<1x?x1x?xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2, 3]]
      : tensor<1x?x1x?xf32> into tensor<?x?xf32>
  return %0, %1 : tensor<1x?x1x?xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @rank_reducing_insert_of_collapse_shape(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x1x1x5xf32>
//       CHECK:   %[[insert:.*]] = tensor.insert_slice %[[t]] into %{{.*}}[0, 0, 0, 0]
//  CHECK-SAME:     [%{{.*}}, 1, 1, 5] [1, 1, 1, 1] : tensor<?x1x1x5xf32> into tensor<?x?x?x?xf32>
//       CHECK:   return %[[insert]]
func.func @rank_reducing_insert_of_collapse_shape(
    %t: tensor<?x1x1x5xf32>, %d: tensor<?x?x?x?xf32>, %sz: index)
  -> tensor<?x?x?x?xf32> {
  %0 = tensor.collapse_shape %t [[0, 1], [2], [3]]
      : tensor<?x1x1x5xf32> into tensor<?x1x5xf32>
  %1 = tensor.insert_slice %0 into %d[0, 0, 0, 0][%sz, 1, 1, 5][1, 1, 1, 1]
      : tensor<?x1x5xf32> into tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @rank_reducing_parallel_insert_of_collapse_shape(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x1x1x5xf32>
//       CHECK:   tensor.parallel_insert_slice %[[t]] into %{{.*}}[0, 0, 0, 0]
//  CHECK-SAME:     [%{{.*}}, 1, 1, 5] [1, 1, 1, 1] : tensor<?x1x1x5xf32> into tensor<?x?x?x?xf32>
func.func @rank_reducing_parallel_insert_of_collapse_shape(
    %t: tensor<?x1x1x5xf32>, %d: tensor<?x?x?x?xf32>, %sz: index, %thr: index)
  -> tensor<?x?x?x?xf32> {
  %0 = tensor.collapse_shape %t [[0, 1], [2], [3]]
      : tensor<?x1x1x5xf32> into tensor<?x1x5xf32>
  %1 = scf.forall (%iv) in (%thr) shared_outs(%o = %d) -> (tensor<?x?x?x?xf32>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %0 into %o[0, 0, 0, 0][%sz, 1, 1, 5][1, 1, 1, 1]
          : tensor<?x1x5xf32> into tensor<?x?x?x?xf32>
    }
  }
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @insert_of_padding_expand_shape(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[d:.*]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[x:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[y:[a-zA-Z0-9_]+]]: index
//       CHECK:   %[[insert:.*]] = tensor.insert_slice %[[t]] into %[[d]][%[[x]], %[[y]], 0, 0]
//  CHECK-SAME:     [1, %{{.*}}, 1, %{{.*}}] [1, 1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?x?xf32>
//       CHECK:   return %[[insert]]
func.func @insert_of_padding_expand_shape(
    %t: tensor<?x?xf32>, %d: tensor<?x?x?x?xf32>, %x: index, %y: index)
  -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sz0 = tensor.dim %t, %c0 : tensor<?x?xf32>
  %sz1 = tensor.dim %t, %c1 : tensor<?x?xf32>
  %0 = tensor.expand_shape %t [[0, 1], [2, 3]] output_shape [1, %sz0, 1, %sz1]
      : tensor<?x?xf32> into tensor<1x?x1x?xf32>
  %1 = tensor.insert_slice %0 into %d[%x, %y, 0, 0][1, %sz0, 1, %sz1][1, 1, 1, 1]
      : tensor<1x?x1x?xf32> into tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @insert_of_non_padding_expand_shape(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[d:.*]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[x:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[y:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[sz:[a-zA-Z0-9_]+]]: index
//       CHECK:   %[[expand:.*]] = tensor.expand_shape %[[t]] {{\[}}[0, 1], [2]]
//  CHECK-SAME:     output_shape [%[[sz]], %{{.*}}, %{{.*}}] : tensor<?x?xf32> into tensor<?x?x?xf32>
//       CHECK:   %[[insert:.*]] = tensor.insert_slice %[[expand]] into %[[d]][%[[x]], %[[y]], 0, 0]
//  CHECK-SAME:     [%[[sz]], 1, %{{.*}}, %{{.*}}] [1, 1, 1, 1] : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
//       CHECK:   return %[[insert]]
func.func @insert_of_non_padding_expand_shape(
    %t: tensor<?x?xf32>, %d: tensor<?x?x?x?xf32>, %x: index, %y: index, %sz: index)
  -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sz0 = tensor.dim %t, %c0 : tensor<?x?xf32>
  %sz1 = tensor.dim %t, %c1 : tensor<?x?xf32>
  %0 = tensor.expand_shape %t [[0, 1], [2]] output_shape [%sz, %sz0, %sz1]
      : tensor<?x?xf32> into tensor<?x?x?xf32>
  %1 = tensor.insert_slice %0 into %d[%x, %y, 0, 0][%sz, 1, %sz0, %sz1][1, 1, 1, 1]
      : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @parallel_insert_of_padding_expand_shape(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[d:.*]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[x:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[y:[a-zA-Z0-9_]+]]: index
//       CHECK:   tensor.parallel_insert_slice %[[t]] into %{{.*}}[%{{.*}}, %{{.*}}, 0, 0]
//  CHECK-SAME:     [1, %{{.*}}, 1, %{{.*}}] [1, 1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?x?xf32>
func.func @parallel_insert_of_padding_expand_shape(
    %t: tensor<?x?xf32>, %d: tensor<?x?x?x?xf32>, %x: index, %y: index)
  -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sz0 = tensor.dim %t, %c0 : tensor<?x?xf32>
  %sz1 = tensor.dim %t, %c1 : tensor<?x?xf32>
  %0 = tensor.expand_shape %t [[0, 1], [2, 3]] output_shape [1, %sz0, 1, %sz1]
      : tensor<?x?xf32> into tensor<1x?x1x?xf32>
  %1 = scf.forall (%i, %j) in (%x, %y) shared_outs(%o = %d) -> (tensor<?x?x?x?xf32>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %0 into %o[%i, %j, 0, 0][1, %sz0, 1, %sz1][1, 1, 1, 1]
          : tensor<1x?x1x?xf32> into tensor<?x?x?x?xf32>
    }
  }
  return %1 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func @parallel_insert_of_non_padding_expand_shape(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[d:.*]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:     %[[x:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[y:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:     %[[sz:[a-zA-Z0-9_]+]]: index
//       CHECK:   %[[expand:.*]] = tensor.expand_shape %[[t]] {{\[}}[0, 1], [2]]
//  CHECK-SAME:     output_shape [%[[sz]], %{{.*}}, %{{.*}}] : tensor<?x?xf32> into tensor<?x?x?xf32>
//       CHECK:   tensor.parallel_insert_slice %[[expand]] into %{{.*}}[%{{.*}}, %{{.*}}, 0, 0]
//  CHECK-SAME:     [%[[sz]], 1, %{{.*}}, %{{.*}}] [1, 1, 1, 1] : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
func.func @parallel_insert_of_non_padding_expand_shape(
    %t: tensor<?x?xf32>, %d: tensor<?x?x?x?xf32>, %x: index, %y: index, %sz: index)
  -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %sz0 = tensor.dim %t, %c0 : tensor<?x?xf32>
  %sz1 = tensor.dim %t, %c1 : tensor<?x?xf32>
  %0 = tensor.expand_shape %t [[0, 1], [2]] output_shape [%sz, %sz0, %sz1]
      : tensor<?x?xf32> into tensor<?x?x?xf32>
  %1 = scf.forall (%i, %j) in (%x, %y) shared_outs(%o = %d) -> (tensor<?x?x?x?xf32>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %0 into %o[%i, %j, 0, 0][%sz, 1, %sz0, %sz1][1, 1, 1, 1]
          : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
    }
  }
  return %1 : tensor<?x?x?x?xf32>
}
