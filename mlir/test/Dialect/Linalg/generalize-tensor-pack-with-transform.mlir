// RUN: mlir-opt -split-input-file --test-transform-dialect-interpreter --canonicalize --test-linalg-transform-patterns="test-generalize-tensor-pack"  %s | FileCheck %s --check-prefix=CHECK-TRANS

func.func @KCRS_to_KCRSsr(%arg0: tensor<1x1x128x64xf32>, %arg1: tensor<1x1x4x8x8x32xf32>) -> tensor<1x1x4x8x8x32xf32> {
  %0 = tensor.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x128x64xf32> -> tensor<1x1x4x8x8x32xf32>
  return %0 : tensor<1x1x4x8x8x32xf32>
}
// CHECK-TRANS-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-TRANS-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 * -32 + 128, 32)>
// CHECK-TRANS-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-TRANS-DAG:   #[[MAP3:.+]] = affine_map<(d0) -> (d0 * -8 + 64, 8)>
// CHECK-TRANS:       func.func @KCRS_to_KCRSsr
// CHECK-TRANS-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-TRANS-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-TRANS:         %{{.+}} = scf.for %[[R:[a-zA-Z0-9]+]] =
// CHECK-TRANS:           %{{.+}} = scf.for %[[S:[a-zA-Z0-9]+]] =
// CHECK-TRANS:             %[[IN_R:.+]] = affine.apply #[[MAP0]](%[[R]])
// CHECK-TRANS:             %[[IN_R_SZ:.+]] = affine.min #[[MAP1]](%[[R]])
// CHECK-TRANS:             %[[IN_S:.+]] = affine.apply #[[MAP2]](%[[S]])
// CHECK-TRANS:             %[[IN_S_SZ:.+]] = affine.min #[[MAP3]](%[[S]])
// CHECK-TRANS:             %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-TRANS-SAME:          [0, 0, %[[IN_R]], %[[IN_S]]] [1, 1, %[[IN_R_SZ]], %[[IN_S_SZ]]] [1, 1, 1, 1]
// CHECK-TRANS:             %[[TILE:.+]] = tensor.extract_slice %[[SRC_SLICE]]
// CHECK-TRANS-SAME:          [0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1] : tensor<1x1x?x?xf32> to tensor<32x8xf32>
// CHECK-TRANS:             %[[EMPTY:.+]] = tensor.empty() : tensor<8x32xf32>
// CHECK-TRANS:             %[[TRANSP:.+]] =  linalg.transpose
// CHECK-TRANS-SAME:          ins(%[[TILE]]
// CHECK-TRANS-SAME:          outs(%[[EMPTY]]
// CHECK-TRANS-SAME:          permutation = [1, 0]
// CHECK-TRANS:             %{{.+}} = tensor.insert_slice %[[TRANSP]] into %{{.+}}

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
// CHECK-TRANS:       func.func @pad_and_pack
// CHECK-TRANS-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-TRANS-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-TRANS-SAME:    %[[PAD_VAL:[a-zA-Z0-9]+]]
// CHECK-TRANS:         scf.for
// CHECK-TRANS:           scf.for
// CHECK-TRANS:             %[[SRC_SLICE]] = tensor.extract_slice %[[SRC]]
// CHECK-TRANS:             %[[PAD:.+]] = tensor.pad %[[SRC_SLICE]]
// CHECK-TRANS:               tensor.yield %[[PAD_VAL]]
// CHECK-TRANS:             } : tensor<?x?xf32> to tensor<8x2xf32>
// CHECK-TRANS:             %[[EMPTY:.+]] = tensor.empty() : tensor<8x2xf32>
// CHECK-TRANS:         %[[TRANSP:.+]] = linalg.transpose
// CHECK-TRANS-SAME:      ins(%[[PAD]] : tensor<8x2xf32>)
// CHECK-TRANS-SAME:      outs(%[[EMPTY]] : tensor<8x2xf32>)
// CHECK-TRANS-SAME:      permutation = [0, 1]
// CHECK-TRANS:         %{{.+}} = tensor.insert_slice %[[TRANSP]] into %{{.+}}

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
// CHECK-TRANS-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-TRANS-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 * -32 + 128, 32)>
// CHECK-TRANS-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-TRANS-DAG:   #[[MAP3:.+]] = affine_map<(d0) -> (d0 * -8 + 256, 8)>
// CHECK-TRANS:       func.func @KC_to_CKkc
// CHECK-TRANS-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-TRANS-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-TRANS:         %{{.+}} = scf.for %[[C:[a-zA-Z0-9]+]] =
// CHECK-TRANS:           %{{.+}} = scf.for %[[K:[a-zA-Z0-9]+]] =
// CHECK-TRANS-DAG:         %[[IN_K:.+]] = affine.apply #[[MAP0]](%[[K]])
// CHECK-TRANS-DAG:         %[[IN_K_SZ:.+]] = affine.min #[[MAP1]](%[[K]])
// CHECK-TRANS-DAG:         %[[IN_C:.+]] = affine.apply #[[MAP2]](%[[C]])
// CHECK-TRANS-DAG:         %[[IN_C_SZ:.+]] = affine.min #[[MAP3]](%[[C]])
// CHECK-TRANS:             %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-TRANS-SAME:          [%[[IN_K]], %[[IN_C]]] [%[[IN_K_SZ]], %[[IN_C_SZ]]] [1, 1]
// CHECK-TRANS:             %[[TILE:.+]] = tensor.extract_slice %[[SRC_SLICE]]
// CHECK-TRANS-SAME:          [0, 0] [32, 8] [1, 1] : tensor<?x?xf32> to tensor<32x8xf32>
// CHECK-TRANS:             %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK-TRANS:             %[[TRANSP:.+]] =  linalg.transpose
// CHECK-TRANS-SAME:          ins(%[[TILE]]
// CHECK-TRANS-SAME:          outs(%[[EMPTY]]
// CHECK-TRANS-SAME:          permutation = [0, 1]
// CHECK-TRANS:             %[[SUB_ITER:.+]] = tensor.insert_slice %[[TRANSP]] into %{{[a-zA-Z0-9]+}}
// CHECK-TRANS-SAME:          [0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1] : tensor<32x8xf32> into tensor<1x1x32x8xf32>
// CHECK-TRANS:             %{{.+}} = tensor.insert_slice %[[SUB_ITER]] into %{{[a-zA-Z0-9]+}}
// CHECK-TRANS-SAME:          [%[[C]], %[[K]], 0, 0] [1, 1, 32, 8] [1, 1, 1, 1] : tensor<1x1x32x8xf32> into tensor<32x4x32x8xf32>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %1, %loops:2 = transform.structured.tile_to_scf_for %0 [1, 1]
}
