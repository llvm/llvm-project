// RUN: mlir-opt -split-input-file -test-linalg-transform-patterns=test-convert-to-destination-style-patterns %s | FileCheck %s

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
