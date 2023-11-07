// RUN: mlir-opt %s --transform-interpreter -split-input-file | FileCheck %s

#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
module {
  func.func @foo(%arg0: f32, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%arg0 : f32) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.mulf %in, %out : f32
      linalg.yield %3 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.structured.tile_using_for %0[8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %2 = transform.cast %loops : !transform.any_op to !transform.op<"scf.for">
    %3 = transform.loop.loop_continuous_peel %2 {single_iter_opt = true} : (!transform.op<"scf.for">) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK: #[[MAP:.*]] = affine_map<()[s0, s1, s2] -> (s1 - s1 mod s2)>
// CHECK: #[[MAP1:.*]] = affine_map<() -> (8)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0) -> (d0 - 1)>
// CHECK: #[[MAP3:.*]] = affine_map<(d0) -> ()>
// CHECK: #[[MAP4:.*]] = affine_map<(d0) -> (d0)>

// CHECK: func.func @foo(%[[S:.*]]: f32, %[[INVEC1:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[DIM:.*]] = tensor.dim %[[INVEC1]], %[[C0]] : tensor<?xf32>
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %{{.*}} = arith.constant 8 : index
// CHECK:       %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[IDX0:.*]] = affine.apply #[[MAP]]()[%[[C0]], %[[DIM]], %[[C8]]]
// CHECK:       %[[INS1:.*]] = scf.for %[[IDX:.*]] = %[[C0]] to %[[IDX0]] step %[[C8]] iter_args(%[[AINVEC1:.*]] = %[[INVEC1]]) -> (tensor<?xf32>) {
// CHECK:         %{{.*}} = affine.apply #[[MAP2]](%[[C8]])
// CHECK:         %[[XS8:.*]] = tensor.extract_slice %[[AINVEC1]][%[[IDX]]] [%[[C8]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK:         %[[MUL:.*]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]]], iterator_types = ["parallel"]} ins(%{{.*}} : f32) outs(%[[XS8]] : tensor<?xf32>) {
// CHECK:         ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK:           %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK:           linalg.yield %{{.*}} : f32
// CHECK:         } -> tensor<?xf32>
// CHECK:         %[[INS:.*]] = tensor.insert_slice %[[MUL]] into %[[AINVEC1]][%[[IDX]]] [%[[C8]]] [1] : tensor<?xf32> into tensor<?xf32>
// CHECK:         scf.yield %[[INS]] : tensor<?xf32>
// CHECK:       }
// CHECK:       %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[IDX2:.*]] = affine.apply #[[MAP]]()[%[[IDX0]], %[[DIM]], %[[C4]]]
// CHECK:       %[[CMP3:.*]] = arith.cmpi slt, %[[IDX0]], %[[IDX2]] : index
// CHECK:       %[[INS2:.*]] = scf.if %[[CMP3]] -> (tensor<?xf32>) {
// CHECK:          %{{.*}} = affine.apply #[[MAP2]](%[[C4]])
// CHECK:         %[[XS4:.*]] = tensor.extract_slice %[[INS1]][%[[IDX0]]] [%[[C4]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK:         %[[MUL:.*]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]]], iterator_types = ["parallel"]} ins(%[[S]] : f32) outs(%[[XS4]] : tensor<?xf32>) {
// CHECK:         ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK:           %{{.*}} = arith.mulf  %{{.*}},  %{{.*}} : f32
// CHECK:           linalg.yield  %{{.*}} : f32
// CHECK:         } -> tensor<?xf32>
// CHECK:         %[[INS:.*]] = tensor.insert_slice %[[MUL]] into %[[INS1]][%[[IDX0]]] [%[[C4]]] [1] : tensor<?xf32> into tensor<?xf32>
// CHECK:         scf.yield %[[INS]] : tensor<?xf32>
// CHECK:       } else {
// CHECK:         scf.yield %[[INS1]] : tensor<?xf32>
// CHECK:       }
// CHECK:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[IDX3:.*]] = affine.apply #[[MAP]]()[%[[IDX2]], %[[DIM]], %[[C2]]]
// CHECK:       %[[CMP4:.*]] = arith.cmpi slt, %[[IDX2]], %[[IDX3]] : index
// CHECK:       %[[INS3:.*]] = scf.if %[[CMP4]] -> (tensor<?xf32>) {
// CHECK:         %{{.*}} = affine.apply #[[MAP2]](%[[C2]])
// CHECK:         %[[XS2:.*]] = tensor.extract_slice %[[INS2]][%[[IDX2]]] [%[[C2]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK:         %[[MUL:.*]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]]], iterator_types = ["parallel"]} ins(%[[S]] : f32) outs(%[[XS2]] : tensor<?xf32>) {
// CHECK:         ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK:           %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK:           linalg.yield %{{.*}} : f32
// CHECK:         } -> tensor<?xf32>
// CHECK:         %[[INS:.*]] = tensor.insert_slice %[[MUL]] into %[[INS2]][%[[IDX2]]] [%[[C2]]] [1] : tensor<?xf32> into tensor<?xf32>
// CHECK:         scf.yield %[[INS]] : tensor<?xf32>
// CHECK:       } else {
// CHECK:         scf.yield %[[INS2]] : tensor<?xf32>
// CHECK:       }
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %{{.*}} = affine.apply #[[MAP]]()[%[[IDX3]], %[[DIM]], %[[C1]]]
// CHECK:       %[[CMP5:.*]] = arith.cmpi slt, %[[IDX3]], %[[DIM]] : index
// CHECK:       %[[INS4:.*]] = scf.if %[[CMP5]] -> (tensor<?xf32>) {
// CHECK:         %{{.*}} = affine.apply #[[MAP2]](%[[C1]])
// CHECK:         %[[XS1:.*]] = tensor.extract_slice %[[INS3]][%[[IDX3]]] [%[[C1]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK:         %[[MUL:.*]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]]], iterator_types = ["parallel"]} ins(%[[S]] : f32) outs(%[[XS1]] : tensor<?xf32>) {
// CHECK:         ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// CHECK:           %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK:           linalg.yield %{{.*}} : f32
// CHECK:         } -> tensor<?xf32>
// CHECK:         %[[INS:.*]] = tensor.insert_slice %[[MUL]] into %[[INS3]][%[[IDX3]]] [%[[C1]]] [1] : tensor<?xf32> into tensor<?xf32>
// CHECK:         scf.yield %[[INS]] : tensor<?xf32>
// CHECK:       } else {
// CHECK:         scf.yield %[[INS3]] : tensor<?xf32>
// CHECK:       }
// CHECK:       return %[[INS4]] : tensor<?xf32>
