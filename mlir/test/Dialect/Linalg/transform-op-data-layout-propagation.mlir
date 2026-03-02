// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.foreach %funcs : !transform.any_op {
    ^bb0(%func: !transform.any_op):
      transform.apply_patterns to %func {
        transform.apply_patterns.linalg.data_layout_propagation {poison_padding = false}
      } : !transform.any_op
      transform.yield
    }
    transform.yield
  }
}

func.func @no_propagation_without_poison(%arg0: tensor<8x8x4x8xf32>, %dest: tensor<?x64xf32>, %arg1: tensor<?x64xbf16>) -> tensor<?x64xbf16> {
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [4, 8] into %dest : tensor<8x8x4x8xf32> -> tensor<?x64xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%unpack : tensor<?x64xf32>) outs(%arg1 : tensor<?x64xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %1 = arith.truncf %in : f32 to bf16
    linalg.yield %1 : bf16
  } -> tensor<?x64xbf16>
  return %0 : tensor<?x64xbf16>
}
// CHECK-LABEL:  func.func @no_propagation_without_poison
// CHECK:          %[[UNPACK:.+]] = linalg.unpack
// CHECK:          linalg.generic{{.*}}ins(%[[UNPACK]]

// -----

module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.foreach %funcs : !transform.any_op {
    ^bb0(%func: !transform.any_op):
      transform.apply_patterns to %func {
        transform.apply_patterns.linalg.data_layout_propagation {poison_padding = true}
      } : !transform.any_op
      transform.yield
    }
    transform.yield
  }
}

func.func @propagation_with_poison(%arg0: tensor<8x8x4x8xf32>, %dest: tensor<?x64xf32>, %arg1: tensor<?x64xbf16>) -> tensor<?x64xbf16> {
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [4, 8] into %dest : tensor<8x8x4x8xf32> -> tensor<?x64xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%unpack : tensor<?x64xf32>) outs(%arg1 : tensor<?x64xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %1 = arith.truncf %in : f32 to bf16
    linalg.yield %1 : bf16
  } -> tensor<?x64xbf16>
  return %0 : tensor<?x64xbf16>
}
// CHECK-LABEL:  func.func @propagation_with_poison
// CHECK:          %[[GENERIC:.+]] = linalg.generic
// CHECK:          linalg.unpack %[[GENERIC]]
