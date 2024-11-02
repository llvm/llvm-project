// RUN: mlir-opt -split-input-file --transform-interpreter --canonicalize --test-linalg-transform-patterns="test-generalize-tensor-unpack"  %s | FileCheck %s

func.func @KCRSsr_to_KCRS(%arg0: tensor<1x1x4x8x8x32xf32>, %arg1: tensor<1x1x128x64xf32>) -> tensor<1x1x128x64xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x4x8x8x32xf32> -> tensor<1x1x128x64xf32>
  return %0 : tensor<1x1x128x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:4 = transform.structured.tile_using_for %0 [1, 1, 32, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK:       func.func @KCRSsr_to_KCRS
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %{{.+}} = scf.for %[[R:[a-zA-Z0-9]+]] =
// CHECK:           %{{.+}} = scf.for %[[S:[a-zA-Z0-9]+]] =
// CHECK:             %[[IN_R:.+]] = affine.apply #[[MAP0]](%[[R]])
// CHECK:             %[[IN_S:.+]] = affine.apply #[[MAP1]](%[[S]])
// CHECK:             %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME:          [0, 0, %[[IN_R]], %[[IN_S]], 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:             %[[TILE:.+]] = tensor.extract_slice %[[SRC_SLICE]]
// CHECK-SAME:          [0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x8x32xf32> to tensor<8x32xf32>
// CHECK:             %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:             %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:          ins(%[[TILE]]
// CHECK-SAME:          outs(%[[EMPTY]]
// CHECK-SAME:          permutation = [1, 0]
// CHECK:             %{{.+}} = tensor.insert_slice %[[TRANSP]] into %{{.+}}

// -----

func.func @unpack_and_extract_slice(%arg0: tensor<2x8x8x2xf32>, %arg1: tensor<13x15xf32>) -> tensor<13x15xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %arg1 : tensor<2x8x8x2xf32> -> tensor<13x15xf32>
  return %0 : tensor<13x15xf32>
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (-d0 + 13, 8)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (-d0 + 15, 2)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0) -> (d0 floordiv 2)>
// CHECK:       func.func @unpack_and_extract_slice
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %{{.+}} = scf.for %[[I:[a-zA-Z0-9]+]] =
// CHECK:           %{{.+}} = scf.for %[[J:[a-zA-Z0-9]+]] =
// CHECK-DAG:         %[[OUT_I_SZ:.+]] = affine.min #[[MAP0]](%[[I]])
// CHECK-DAG:         %[[OUT_J_SZ:.+]] = affine.min #[[MAP1]](%[[J]])
// CHECK-DAG:         %[[IN_I:.+]] = affine.apply #[[MAP2]](%[[I]])
// CHECK-DAG:         %[[IN_J:.+]] = affine.apply #[[MAP3]](%[[J]])
// CHECK:             %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME:          [%[[IN_I]], %[[IN_J]], 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:             %[[ITER_SLICE:.+]] = tensor.extract_slice %{{[a-zA-Z0-9]+}}
// CHECK-SAME:          [%[[I]], %[[J]]] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]]
// CHECK:             %[[TILE:.+]] = tensor.extract_slice %[[SRC_SLICE]]
// CHECK-SAME:          [0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1] : tensor<1x1x8x2xf32> to tensor<8x2xf32>
// CHECK:             %[[EMPTY:.+]] = tensor.empty() : tensor<8x2xf32>
// CHECK:             %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:          ins(%[[TILE]] : tensor<8x2xf32>)
// CHECK-SAME:          outs(%[[EMPTY]] : tensor<8x2xf32>)
// CHECK-SAME:          permutation = [0, 1]
// CHECK:             %[[UNPACK_TILE:.+]] = tensor.extract_slice %[[TRANSP]]
// CHECK-SAME:          [0, 0] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]
// CHECK:             %[[INSERT1:.+]] = tensor.insert_slice %[[UNPACK_TILE]] into %[[ITER_SLICE]]
// CHECK-SAME:          [0, 0] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]
// CHECK:             %[[INSERT2:.+]] = tensor.insert_slice %[[INSERT1]] into %{{[a-zA-Z0-9]+}}
// CHECK-SAME:          [%[[I]], %[[J]]] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 [8, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @CKkc_to_KC(%arg0: tensor<32x4x32x8xf32>, %arg1: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = tensor.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<32x4x32x8xf32> -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK:       func.func @CKkc_to_KC
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %{{.+}} = scf.for %[[K:[a-zA-Z0-9]+]] =
// CHECK:           %{{.+}} = scf.for %[[C:[a-zA-Z0-9]+]] =
// CHECK:             %[[IN_K:.+]] = affine.apply #[[MAP0]](%[[K]])
// CHECK:             %[[IN_C:.+]] = affine.apply #[[MAP1]](%[[C]])
// CHECK:             %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME:          [%[[IN_C]], %[[IN_K]], 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:             %[[TILE:.+]] = tensor.extract_slice %[[SRC_SLICE]]
// CHECK-SAME:          [0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1] : tensor<1x1x32x8xf32> to tensor<32x8xf32>
// CHECK:             %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:             %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:          ins(%[[TILE]]
// CHECK-SAME:          outs(%[[EMPTY]]
// CHECK-SAME:          permutation = [0, 1]
// CHECK:             %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %{{[a-zA-Z0-9]+}}
// CHECK-SAME:          [%[[K]], %[[C]]] [32, 8] [1, 1]


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 [32, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
