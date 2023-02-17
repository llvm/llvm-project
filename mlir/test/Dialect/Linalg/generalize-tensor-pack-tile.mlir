// RUN: mlir-opt -split-input-file --test-transform-dialect-interpreter --canonicalize --test-linalg-transform-patterns="test-generalize-tensor-pack"  %s | FileCheck %s

func.func @KCRS_to_KCRSsr(%arg0: tensor<1x1x128x64xf32>, %arg1: tensor<1x1x4x8x8x32xf32>) -> tensor<1x1x4x8x8x32xf32> {
  %0 = tensor.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x128x64xf32> -> tensor<1x1x4x8x8x32xf32>
  return %0 : tensor<1x1x4x8x8x32xf32>
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 * -32 + 128, 32)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0) -> (d0 * -8 + 64, 8)>
// CHECK:       func.func @KCRS_to_KCRSsr
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %{{.+}} = scf.for %[[R:[a-zA-Z0-9]+]] =
// CHECK:           %{{.+}} = scf.for %[[S:[a-zA-Z0-9]+]] =
// CHECK:             %[[IN_R:.+]] = affine.apply #[[MAP0]](%[[R]])
// CHECK:             %[[IN_R_SZ:.+]] = affine.min #[[MAP1]](%[[R]])
// CHECK:             %[[IN_S:.+]] = affine.apply #[[MAP2]](%[[S]])
// CHECK:             %[[IN_S_SZ:.+]] = affine.min #[[MAP3]](%[[S]])
// CHECK:             %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME:          [0, 0, %[[IN_R]], %[[IN_S]]] [1, 1, %[[IN_R_SZ]], %[[IN_S_SZ]]] [1, 1, 1, 1]
// CHECK:             %[[TILE:.+]] = tensor.extract_slice %[[SRC_SLICE]]
// CHECK-SAME:          [0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1] : tensor<1x1x?x?xf32> to tensor<32x8xf32>
// CHECK:             %[[EMPTY:.+]] = tensor.empty() : tensor<8x32xf32>
// CHECK:             %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:          ins(%[[TILE]]
// CHECK-SAME:          outs(%[[EMPTY]]
// CHECK-SAME:          permutation = [1, 0]
// CHECK:             %{{.+}} = tensor.insert_slice %[[TRANSP]] into %{{.+}}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %1, %loops:4 = transform.structured.tile_to_scf_for %0 [1, 1, 1, 1]
}

// -----

func.func @pad_and_pack(%arg0: tensor<13x15xf32>, %arg1: tensor<2x8x8x2xf32>, %arg2: f32) -> tensor<2x8x8x2xf32> {
  %0 = tensor.pack %arg0 padding_value(%arg2 : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %arg1 : tensor<13x15xf32> -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}
// CHECK:       func.func @pad_and_pack
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[PAD_VAL:[a-zA-Z0-9]+]]
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             %[[SRC_SLICE]] = tensor.extract_slice %[[SRC]]
// CHECK:             %[[PAD:.+]] = tensor.pad %[[SRC_SLICE]]
// CHECK:               tensor.yield %[[PAD_VAL]]
// CHECK:             } : tensor<?x?xf32> to tensor<8x2xf32>
// CHECK:             %[[EMPTY:.+]] = tensor.empty() : tensor<8x2xf32>
// CHECK:         %[[TRANSP:.+]] = linalg.transpose
// CHECK-SAME:      ins(%[[PAD]] : tensor<8x2xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<8x2xf32>)
// CHECK-SAME:      permutation = [0, 1]
// CHECK:         %{{.+}} = tensor.insert_slice %[[TRANSP]] into %{{.+}}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %1, %loops:2 = transform.structured.tile_to_scf_for %0 [1, 1]
}

// -----


func.func @KC_to_CKkc(%arg0: tensor<128x256xf32>, %arg1: tensor<32x4x32x8xf32>) -> tensor<32x4x32x8xf32> {
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<128x256xf32> -> tensor<32x4x32x8xf32>
  return %0 : tensor<32x4x32x8xf32>
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 * -32 + 128, 32)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0) -> (d0 * -8 + 256, 8)>
// CHECK:       func.func @KC_to_CKkc
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %{{.+}} = scf.for %[[C:[a-zA-Z0-9]+]] =
// CHECK:           %{{.+}} = scf.for %[[K:[a-zA-Z0-9]+]] =
// CHECK-DAG:         %[[IN_K:.+]] = affine.apply #[[MAP0]](%[[K]])
// CHECK-DAG:         %[[IN_K_SZ:.+]] = affine.min #[[MAP1]](%[[K]])
// CHECK-DAG:         %[[IN_C:.+]] = affine.apply #[[MAP2]](%[[C]])
// CHECK-DAG:         %[[IN_C_SZ:.+]] = affine.min #[[MAP3]](%[[C]])
// CHECK:             %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME:          [%[[IN_K]], %[[IN_C]]] [%[[IN_K_SZ]], %[[IN_C_SZ]]] [1, 1]
// CHECK:             %[[TILE:.+]] = tensor.extract_slice %[[SRC_SLICE]]
// CHECK-SAME:          [0, 0] [32, 8] [1, 1] : tensor<?x?xf32> to tensor<32x8xf32>
// CHECK:             %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:             %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:          ins(%[[TILE]]
// CHECK-SAME:          outs(%[[EMPTY]]
// CHECK-SAME:          permutation = [0, 1]
// CHECK:             %[[SUB_ITER:.+]] = tensor.insert_slice %[[TRANSP]] into %{{[a-zA-Z0-9]+}}
// CHECK-SAME:          [0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1] : tensor<32x8xf32> into tensor<1x1x32x8xf32>
// CHECK:             %{{.+}} = tensor.insert_slice %[[SUB_ITER]] into %{{[a-zA-Z0-9]+}}
// CHECK-SAME:          [%[[C]], %[[K]], 0, 0] [1, 1, 32, 8] [1, 1, 1, 1] : tensor<1x1x32x8xf32> into tensor<32x4x32x8xf32>
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %1, %loops:2 = transform.structured.tile_to_scf_for %0 [1, 1]
}
