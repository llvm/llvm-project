// RUN: mlir-opt  -transform-interpreter --split-input-file -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @tensor_from_elements_0d(
//  CHECK-SAME:     %[[arg0:.*]]: index
//       CHECK:   %[[empty:.*]] = tensor.empty() : tensor<index>
//       CHECK:   %[[insert:.*]] = tensor.insert %[[arg0]] into %[[empty]][]
//       CHECK:   return %[[insert]]
func.func @tensor_from_elements_0d(%arg0: index) -> tensor<index> {
  %0 = tensor.from_elements %arg0 : tensor<index>
  return %0 : tensor<index>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.from_elements"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_in_destination_passing_style %0
      : (!transform.any_op) -> !transform.any_op
      transform.yield
  }
}

// -----

// CHECK-LABEL: func @tensor_from_elements_1d(
//  CHECK-SAME:     %[[arg0:.*]]: index, %[[arg1:.*]]: index
//   CHECK-DAG:   %[[empty:.*]] = tensor.empty() : tensor<2xindex>
//   CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
//       CHECK:   %[[insert:.*]] = tensor.insert %[[arg0]] into %[[empty]][%[[c0]]]
//       CHECK:   %[[insert2:.*]] = tensor.insert %[[arg1]] into %[[insert]][%[[c1]]]
//       CHECK:   return %[[insert2]]
func.func @tensor_from_elements_1d(%arg0: index, %arg1: index) -> tensor<2xindex> {
  %0 = tensor.from_elements %arg0, %arg1 : tensor<2xindex>
  return %0 : tensor<2xindex>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.from_elements"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_in_destination_passing_style %0
      : (!transform.any_op) -> !transform.any_op
      transform.yield
  }
}

// -----

// CHECK-LABEL: func @tensor_from_elements_2d(
//  CHECK-SAME:     %[[arg0:.*]]: index, %[[arg1:.*]]: index
//   CHECK-DAG:   %[[empty:.*]] = tensor.empty() : tensor<3x2xindex>
//   CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[c2:.*]] = arith.constant 2 : index
//       CHECK:   %[[insert0:.*]] = tensor.insert %[[arg0]] into %[[empty]][%[[c0]], %[[c0]]]
//       CHECK:   %[[insert1:.*]] = tensor.insert %[[arg1]] into %[[insert0]][%[[c0]], %[[c1]]]
//       CHECK:   %[[insert2:.*]] = tensor.insert %[[arg0]] into %[[insert1]][%[[c1]], %[[c0]]]
//       CHECK:   %[[insert3:.*]] = tensor.insert %[[arg1]] into %[[insert2]][%[[c1]], %[[c1]]]
//       CHECK:   %[[insert4:.*]] = tensor.insert %[[arg0]] into %[[insert3]][%[[c2]], %[[c0]]]
//       CHECK:   %[[insert5:.*]] = tensor.insert %[[arg1]] into %[[insert4]][%[[c2]], %[[c1]]]
//       CHECK:   return %[[insert5]]
func.func @tensor_from_elements_2d(%arg0: index, %arg1: index) -> tensor<3x2xindex> {
  %0 = tensor.from_elements %arg0, %arg1, %arg0, %arg1, %arg0, %arg1
         : tensor<3x2xindex>
  return %0 : tensor<3x2xindex>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.from_elements"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_in_destination_passing_style %0
      : (!transform.any_op) -> !transform.any_op
      transform.yield
  }
}

// -----

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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.generate"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_in_destination_passing_style %0
      : (!transform.any_op) -> !transform.any_op
      transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_in_destination_passing_style %0
      : (!transform.any_op) -> !transform.any_op
      transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_in_destination_passing_style %0
      : (!transform.any_op) -> !transform.any_op
      transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_in_destination_passing_style %0
      : (!transform.any_op) -> !transform.any_op
      transform.yield
  }
}

// -----

// CHECK-LABEL: func @tensor_pad_nofold(
//  CHECK-SAME:   %[[t1:.*]]: tensor<?x?xindex>, %[[padding:.*]]: index
//   CHECK-NOT:   linalg.fill
//   CHECK-NOT:   generic
//   CHECK-NOT:   insert_slice
//       CHECK:   %[[alloc_tensor:.*]] = bufferization.alloc_tensor(%{{.*}}) : tensor<?x?xindex>
//       CHECK:   %[[copied:.*]] = linalg.copy ins(%[[t1]] : tensor<?x?xindex>) outs(%[[alloc_tensor]] : tensor<?x?xindex>) -> tensor<?x?xindex>
//       CHECK:   return %[[copied]]
func.func @tensor_pad_nofold(%t1: tensor<?x?xindex>, %padding: index)
    -> tensor<?x?xindex> {
  %c0 = arith.constant 0 : index
  %0 = tensor.pad %t1 nofold low[0, %c0] high[%c0, 0] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %padding : index
  } : tensor<?x?xindex> to tensor<?x?xindex>
  return %0: tensor<?x?xindex>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_in_destination_passing_style %0
      : (!transform.any_op) -> !transform.any_op
      transform.yield
  }
}
