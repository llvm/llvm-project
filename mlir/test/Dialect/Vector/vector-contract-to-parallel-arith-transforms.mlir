// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func @parallel_contract_lowering
//       CHECK:   %[[E0:.*]] = vector.extract %{{.*}}[0, 0] : vector<4xf32> from vector<1x1x4xf32>
//       CHECK:   %[[E1:.*]] = vector.extract %{{.*}}[0, 0] : vector<4xf32> from vector<1x1x4xf32>
//       CHECK:   %[[F:.*]] = vector.fma %[[E0]], %[[E1]], %{{.*}} : vector<4xf32>
//       CHECK:   return %[[F]] : vector<4xf32>
func.func @parallel_contract_lowering(%arg0: vector<1x1x4xf32>, %arg1: vector<1x1x4xf32>, %arg2: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"], kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<1x1x4xf32>, vector<1x1x4xf32> into vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func @parallel_contract_lowering_broadcast
//       CHECK:   %[[B:.*]] = vector.broadcast %{{.*}} : vector<1x1xf32> to vector<4x1x1xf32>
//       CHECK:   %[[T:.*]] = vector.transpose %[[B]], [1, 2, 0] : vector<4x1x1xf32> to vector<1x1x4xf32>
//       CHECK:   %[[E0:.*]] = vector.extract %[[T]][0, 0] : vector<4xf32> from vector<1x1x4xf32>
//       CHECK:   %[[E1:.*]] = vector.extract %{{.*}}[0, 0] : vector<4xf32> from  vector<1x1x4xf32>
//       CHECK:   %[[F:.*]] = vector.fma %[[E0]], %[[E1]], %{{.*}} : vector<4xf32>
//       CHECK:   return %[[F]] : vector<4xf32>
func.func @parallel_contract_lowering_broadcast(%arg0: vector<1x1xf32>, %arg1: vector<1x1x4xf32>, %arg2: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"], kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<1x1xf32>, vector<1x1x4xf32> into vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func @parallel_contract_lowering
//       CHECK:   %[[B:.*]] = vector.broadcast %{{.*}} : vector<1x1xf32> to vector<4x1x1xf32>
//       CHECK:   %[[T0:.*]] = vector.transpose %[[B]], [1, 2, 0] : vector<4x1x1xf32> to vector<1x1x4xf32>
//       CHECK:   %[[T1:.*]] = vector.transpose %{{.*}}, [0, 2, 1] : vector<1x4x1xf32> to vector<1x1x4xf32>
//       CHECK:   %[[E0:.*]] = vector.extract %[[T0]][0, 0] : vector<4xf32> from vector<1x1x4xf32>
//       CHECK:   %[[E1:.*]] = vector.extract %[[T1]][0, 0] : vector<4xf32> from vector<1x1x4xf32>
//       CHECK:   %[[F:.*]] = vector.fma %[[E0]], %[[E1]], %arg2 : vector<4xf32>
//       CHECK:   return %[[F]] : vector<4xf32>
func.func @parallel_contract_lowering_transpose(%arg0: vector<1x1xf32>, %arg1: vector<1x4x1xf32>, %arg2: vector<4xf32>) -> vector<4xf32> {
  %0 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0, d2)>, affine_map<(d0, d1, d2) -> (d0)>], iterator_types = ["parallel", "reduction", "reduction"], kind = #vector.kind<add>} %arg0, %arg1, %arg2 : vector<1x1xf32>, vector<1x4x1xf32> into vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func @parallel_contract_lowering_scalar
//       CHECK:   %[[E0:.*]] = vector.extract %{{.*}}[0, 0] : f32 from vector<1x1xf32>
//       CHECK:   %[[E1:.*]] = vector.extract %{{.*}}[0, 0] : f32 from vector<1x1xf32>
//       CHECK:   %[[M:.*]] = arith.mulf %[[E0]], %[[E1]] : f32
//       CHECK:   %[[A:.*]] = arith.addf %[[M]], %{{.*}} : f32
//       CHECK:   return %[[A]] : f32
func.func @parallel_contract_lowering_scalar(%arg0: vector<1x1xf32>, %arg1: vector<1x1xf32>, %arg2: f32) -> f32 {
  %0 = vector.contract {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> ()>],
    iterator_types = ["reduction", "reduction"], kind = #vector.kind<add>}
  %arg0, %arg1, %arg2 : vector<1x1xf32>, vector<1x1xf32> into f32
  return %0 : f32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "parallelarith"
    } : !transform.any_op
    transform.yield
  }
}
