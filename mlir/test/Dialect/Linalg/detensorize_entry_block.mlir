// RUN: mlir-opt %s -allow-unregistered-dialect -pass-pipeline="builtin.module(func.func(linalg-detensorize))" | FileCheck %s

#map = affine_map<() -> ()>
func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%arg0 : tensor<f32>) outs(%0 : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<f32>
  cf.br ^bb1(%1 : tensor<f32>)
^bb1(%2: tensor<f32>):  // pred: ^bb0
  return %2 : tensor<f32>
}

// CHECK-LABEL: @main
// CHECK-SAME:       (%[[ARG0:.+]]: tensor<f32>) -> tensor<f32>
// CHECK:   %[[EXTRACTED:.+]] = tensor.extract %[[ARG0]][] : tensor<f32>
// CHECK: cf.br ^{{.*}}
// CHECK: ^{{.*}}:
// CHECK:   %[[ELEMENTS:.+]] = tensor.from_elements %[[EXTRACTED]] : tensor<f32>
// CHECK:   return %[[ELEMENTS]] : tensor<f32>
