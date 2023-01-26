// RUN: mlir-opt -split-input-file --test-linalg-transform-patterns="test-generalize-tensor-unpack"  %s | FileCheck %s

func.func @simple_KCRSsr_to_KCRS(%arg0: tensor<1x1x1x1x8x32xf32>, %arg1: tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x1x1x8x32xf32> -> tensor<1x1x32x8xf32>
  return %0 : tensor<1x1x32x8xf32>
}
// CHECK-LABEL: func.func @simple_KCRSsr_to_KCRS
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[TILE]] : tensor<8x32xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<32x8xf32>)
// CHECK-SAME:      permutation = [1, 0]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

func.func @simple_unpack_and_extract_slice(%input: tensor<1x1x8x2xf32>, %output: tensor<5x1xf32>) -> tensor<5x1xf32> {
  %0 = tensor.unpack %input inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<1x1x8x2xf32> -> tensor<5x1xf32>
  return %0 : tensor<5x1xf32>
}
// CHECK-LABEL: func.func @simple_unpack_and_extract_slice
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<8x2xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[TILE]] : tensor<8x2xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<8x2xf32>)
// CHECK-SAME:      permutation = [0, 1]
//                They have the same type, so the insert_slice op is folded
//                away.
// CHECK:         %[[SLICE:.+]] = tensor.extract_slice %[[TRANSP]][0, 0] [5, 1] [1, 1]
// CHECK:         return %[[SLICE]]

// -----

func.func @simple_CNnc_to_NC(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<32x8xf32>) -> tensor<32x8xf32>{
  %0 = tensor.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<1x1x32x8xf32> -> tensor<32x8xf32>
  return %0 : tensor<32x8xf32>
}
// CHECK-LABEL: func.func @simple_CNnc_to_NC
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[TILE]] : tensor<32x8xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<32x8xf32>)
// CHECK-SAME:      permutation = [0, 1]
//                They have the same type, so the insert_slice op is folded
//                away.
// CHECK:         return %[[TRANSP]]

// -----

// RUN: mlir-opt -split-input-file --test-transform-dialect-interpreter --canonicalize --test-linalg-transform-patterns="test-generalize-tensor-unpack"  %s | FileCheck %s --check-prefix=CHECK-TRANS

func.func @KCRSsr_to_KCRS(%arg0: tensor<1x1x4x8x8x32xf32>, %arg1: tensor<1x1x128x64xf32>) -> tensor<1x1x128x64xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x4x8x8x32xf32> -> tensor<1x1x128x64xf32>
  return %0 : tensor<1x1x128x64xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.unpack"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %1, %loops:4 = transform.structured.tile_to_scf_for %0 [1, 1, 32, 8]
}
// CHECK-TRANS-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-TRANS-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-TRANS:       func.func @KCRSsr_to_KCRS
// CHECK-TRANS-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-TRANS-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-TRANS:         %{{.+}} = scf.for %[[R:[a-zA-Z0-9]+]] =
// CHECK-TRANS:           %{{.+}} = scf.for %[[S:[a-zA-Z0-9]+]] =
// CHECK-TRANS:             %[[IN_R:.+]] = affine.apply #[[MAP0]](%[[R]])
// CHECK-TRANS:             %[[IN_S:.+]] = affine.apply #[[MAP1]](%[[S]])
// CHECK-TRANS:             %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-TRANS-SAME:          [0, 0, %[[IN_R]], %[[IN_S]], 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK-TRANS:             %[[TILE:.+]] = tensor.extract_slice %[[SRC_SLICE]]
// CHECK-TRANS-SAME:          [0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x8x32xf32> to tensor<8x32xf32>
// CHECK-TRANS:             %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK-TRANS:             %[[TRANSP:.+]] =  linalg.transpose
// CHECK-TRANS-SAME:          ins(%[[TILE]]
// CHECK-TRANS-SAME:          outs(%[[EMPTY]]
// CHECK-TRANS-SAME:          permutation = [1, 0]
// CHECK-TRANS:             %{{.+}} = tensor.insert_slice %[[TRANSP]] into %{{.+}}

// -----

func.func @unpack_and_extract_slice(%arg0: tensor<2x8x8x2xf32>, %arg1: tensor<13x15xf32>) -> tensor<13x15xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %arg1 : tensor<2x8x8x2xf32> -> tensor<13x15xf32>
  return %0 : tensor<13x15xf32>
}
// CHECK-TRANS-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (-d0 + 13, 8)>
// CHECK-TRANS-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (-d0 + 15, 2)>
// CHECK-TRANS-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-TRANS-DAG:   #[[MAP3:.+]] = affine_map<(d0) -> (d0 floordiv 2)>
// CHECK-TRANS:       func.func @unpack_and_extract_slice
// CHECK-TRANS-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-TRANS-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-TRANS:         %{{.+}} = scf.for %[[I:[a-zA-Z0-9]+]] =
// CHECK-TRANS:           %[[OUT_I_SZ:.+]] = affine.min #[[MAP0]](%[[I]])
// CHECK-TRANS:           %{{.+}} = scf.for %[[J:[a-zA-Z0-9]+]] =
// CHECK-TRANS:             %[[OUT_J_SZ:.+]] = affine.min #[[MAP1]](%[[J]])
// CHECK-TRANS:             %[[IN_I:.+]] = affine.apply #[[MAP2]](%[[I]])
// CHECK-TRANS:             %[[IN_J:.+]] = affine.apply #[[MAP3]](%[[J]])
// CHECK-TRANS:             %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-TRANS-SAME:          [%[[IN_I]], %[[IN_J]], 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK-TRANS:             %[[ITER_SLICE:.+]] = tensor.extract_slice %{{[a-zA-Z0-9]+}}
// CHECK-TRANS-SAME:          [%[[I]], %[[J]]] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]]
// CHECK-TRANS:             %[[TILE:.+]] = tensor.extract_slice %[[SRC_SLICE]]
// CHECK-TRANS-SAME:          [0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1] : tensor<1x1x8x2xf32> to tensor<8x2xf32>
// CHECK-TRANS:             %[[EMPTY:.+]] = tensor.empty() : tensor<8x2xf32>
// CHECK-TRANS:             %[[TRANSP:.+]] =  linalg.transpose
// CHECK-TRANS-SAME:          ins(%[[TILE]] : tensor<8x2xf32>)
// CHECK-TRANS-SAME:          outs(%[[EMPTY]] : tensor<8x2xf32>)
// CHECK-TRANS-SAME:          permutation = [0, 1]
// CHECK-TRANS:             %[[UNPACK_TILE:.+]] = tensor.extract_slice %[[TRANSP]]
// CHECK-TRANS-SAME:          [0, 0] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]
// CHECK-TRANS:             %[[INSERT1:.+]] = tensor.insert_slice %[[UNPACK_TILE]] into %[[ITER_SLICE]]
// CHECK-TRANS-SAME:          [0, 0] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]
// CHECK-TRANS:             %[[INSERT2:.+]] = tensor.insert_slice %[[INSERT1]] into %{{[a-zA-Z0-9]+}}
// CHECK-TRANS-SAME:          [%[[I]], %[[J]]] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.unpack"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %1, %loops:2 = transform.structured.tile_to_scf_for %0 [8, 2]
}

// -----

func.func @CKkc_to_KC(%arg0: tensor<32x4x32x8xf32>, %arg1: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = tensor.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<32x4x32x8xf32> -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}
// CHECK-TRANS-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-TRANS-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-TRANS:       func.func @CKkc_to_KC
// CHECK-TRANS-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-TRANS-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-TRANS:         %{{.+}} = scf.for %[[K:[a-zA-Z0-9]+]] =
// CHECK-TRANS:           %{{.+}} = scf.for %[[C:[a-zA-Z0-9]+]] =
// CHECK-TRANS:             %[[IN_K:.+]] = affine.apply #[[MAP0]](%[[K]])
// CHECK-TRANS:             %[[IN_C:.+]] = affine.apply #[[MAP1]](%[[C]])
// CHECK-TRANS:             %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-TRANS-SAME:          [%[[IN_C]], %[[IN_K]], 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK-TRANS:             %[[TILE:.+]] = tensor.extract_slice %[[SRC_SLICE]]
// CHECK-TRANS-SAME:          [0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1] : tensor<1x1x32x8xf32> to tensor<32x8xf32>
// CHECK-TRANS:             %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK-TRANS:             %[[TRANSP:.+]] =  linalg.transpose
// CHECK-TRANS-SAME:          ins(%[[TILE]]
// CHECK-TRANS-SAME:          outs(%[[EMPTY]]
// CHECK-TRANS-SAME:          permutation = [0, 1]
// CHECK-TRANS:             %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %{{[a-zA-Z0-9]+}}
// CHECK-TRANS-SAME:          [%[[K]], %[[C]]] [32, 8] [1, 1]


transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.unpack"]} in %arg1 : (!pdl.operation) -> !pdl.operation
    %1, %loops:2 = transform.structured.tile_to_scf_for %0 [32, 8]
}
