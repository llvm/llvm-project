// RUN: mlir-opt -split-input-file -test-linalg-transform-patterns=test-convert-to-destination-style-patterns -canonicalize %s | FileCheck %s

// CHECK: #[[$map:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @tensor_generate(
//  CHECK-SAME:     %[[s1:.*]]: index, %[[s2:.*]]: index
//       CHECK:   %[[empty:.*]] = tensor.empty(%[[s1]], %[[s2]]) : tensor<?x?xindex>
//       CHECK:   %[[generic:.*]] = linalg.generic
//  CHECK-SAME:       {indexing_maps = [#[[$map]]], iterator_types = ["parallel", "parallel"]}
//  CHECK-SAME:       outs(%[[empty]] : tensor<?x?xindex>) {
//       CHECK:     %[[i0:.*]] = linalg.index 0
//       CHECK:     %[[i1:.*]] = linalg.index 1
//       CHECK:     %[[added:.*]] = arith.addi %[[i0]], %[[i1]]
//       CHECK:     linalg.yield %[[added]]
//       CHECK:   }
//       CHECK:   return %[[generic]]
func.func @tensor_generate(%s1: index, %s2: index) -> tensor<?x?xindex> {
  %0 = tensor.generate %s1, %s2 {
    ^bb0(%arg0: index, %arg1: index):
    %1 = arith.addi %arg0, %arg1 : index
    tensor.yield %1 : index
  } : tensor<?x?xindex>
  return %0 : tensor<?x?xindex>
}

// -----

// CHECK:       #[[$map:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
// CHECK:       #[[$map1:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 10)>
// CHECK:       #[[$map2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @tensor_pad(
//  CHECK-SAME:   %[[t1:.*]]: tensor<?x10xindex>, %[[l2:.*]]: index, %[[h1:.*]]: index, %[[h2:.*]]: index
//   CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[dim0:.*]] = tensor.dim %[[t1]], %[[c0]]
//   CHECK-DAG:   %[[size0:.*]] = affine.apply #[[$map]]()[%[[h1]], %[[dim0]]]
//   CHECK-DAG:   %[[size1:.*]] = affine.apply #[[$map1]]()[%[[l2]], %[[h2]]]
//       CHECK:   %[[empty:.*]] = tensor.empty(%[[size0]], %[[size1]]) : tensor<?x?xindex>
//       CHECK:   %[[generic:.*]] = linalg.generic
//  CHECK-SAME:       {indexing_maps = [#[[$map2]]], iterator_types = ["parallel", "parallel"]}
//  CHECK-SAME:       outs(%[[empty]] : tensor<?x?xindex>) {
//       CHECK:     %[[i0:.*]] = linalg.index 0
//       CHECK:     %[[i1:.*]] = linalg.index 1
//       CHECK:     %[[mul:.*]] = arith.muli %[[i0]], %[[i1]]
//       CHECK:     linalg.yield %[[mul]]
//       CHECK:   }
//   CHECK-DAG:   %[[dim0:.*]] = tensor.dim %[[t1]], %[[c0]]
//       CHECK:   %[[inserted:.*]] = tensor.insert_slice %[[t1]] into %[[generic]][5, %[[l2]]] [%[[dim0]], 10] [1, 1] : tensor<?x10xindex> into tensor<?x?xindex>
//       CHECK:   return %[[inserted]]
func.func @tensor_pad(%t1: tensor<?x10xindex>, %l2: index, %h1: index,
                      %h2: index) -> tensor<?x?xindex> {
  %0 = tensor.pad %t1 low[5, %l2] high[%h1, %h2] {
  ^bb0(%arg0: index, %arg1: index):
    %m = arith.muli %arg0, %arg1 : index
    tensor.yield %m : index
  } : tensor<?x10xindex> to tensor<?x?xindex>
  return %0 : tensor<?x?xindex>
}

// -----

// CHECK:       #[[$map:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
// CHECK:       #[[$map1:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 10)>
// CHECK-LABEL: func @tensor_pad_constant(
//  CHECK-SAME:   %[[t1:.*]]: tensor<?x10xindex>, %[[l2:.*]]: index, %[[h1:.*]]: index, %[[h2:.*]]: index
//   CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[c50:.*]] = arith.constant 50 : index
//   CHECK-DAG:   %[[dim0:.*]] = tensor.dim %[[t1]], %[[c0]]
//   CHECK-DAG:   %[[size0:.*]] = affine.apply #[[$map]]()[%[[h1]], %[[dim0]]]
//   CHECK-DAG:   %[[size1:.*]] = affine.apply #[[$map1]]()[%[[l2]], %[[h2]]]
//       CHECK:   %[[empty:.*]] = tensor.empty(%[[size0]], %[[size1]]) : tensor<?x?xindex>
//       CHECK:   %[[filled:.*]] = linalg.fill ins(%[[c50]] : index) outs(%[[empty]] : tensor<?x?xindex>)
//   CHECK-DAG:   %[[dim0:.*]] = tensor.dim %[[t1]], %[[c0]]
//       CHECK:   %[[inserted:.*]] = tensor.insert_slice %[[t1]] into %[[filled]][5, %[[l2]]] [%[[dim0]], 10] [1, 1] : tensor<?x10xindex> into tensor<?x?xindex>
//       CHECK:   return %[[inserted]]
func.func @tensor_pad_constant(%t1: tensor<?x10xindex>, %l2: index, %h1: index,
                               %h2: index) -> tensor<?x?xindex> {
  %0 = tensor.pad %t1 low[5, %l2] high[%h1, %h2] {
  ^bb0(%arg0: index, %arg1: index):
    %c = arith.constant 50 : index
    tensor.yield %c : index
  } : tensor<?x10xindex> to tensor<?x?xindex>
  return %0 : tensor<?x?xindex>
}

// -----

// CHECK:       #[[$map:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
// CHECK:       #[[$map1:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 10)>
// CHECK-LABEL: func @tensor_pad_invariant(
//  CHECK-SAME:   %[[t1:.*]]: tensor<?x10xindex>, %[[l2:.*]]: index, %[[h1:.*]]: index, %[[h2:.*]]: index, %[[padding:.*]]: index
//   CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[dim0:.*]] = tensor.dim %[[t1]], %[[c0]]
//   CHECK-DAG:   %[[size0:.*]] = affine.apply #[[$map]]()[%[[h1]], %[[dim0]]]
//   CHECK-DAG:   %[[size1:.*]] = affine.apply #[[$map1]]()[%[[l2]], %[[h2]]]
//       CHECK:   %[[empty:.*]] = tensor.empty(%[[size0]], %[[size1]]) : tensor<?x?xindex>
//       CHECK:   %[[filled:.*]] = linalg.fill ins(%[[padding]] : index) outs(%[[empty]] : tensor<?x?xindex>)
//   CHECK-DAG:   %[[dim0:.*]] = tensor.dim %[[t1]], %[[c0]]
//       CHECK:   %[[inserted:.*]] = tensor.insert_slice %[[t1]] into %[[filled]][5, %[[l2]]] [%[[dim0]], 10] [1, 1] : tensor<?x10xindex> into tensor<?x?xindex>
//       CHECK:   return %[[inserted]]
func.func @tensor_pad_invariant(%t1: tensor<?x10xindex>, %l2: index, %h1: index,
                                %h2: index, %padding: index) -> tensor<?x?xindex> {
  %0 = tensor.pad %t1 low[5, %l2] high[%h1, %h2] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %padding : index
  } : tensor<?x10xindex> to tensor<?x?xindex>
  return %0 : tensor<?x?xindex>
}
