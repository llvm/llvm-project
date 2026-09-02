// RUN: mlir-opt --transform-interpreter --cse -split-input-file %s | FileCheck %s

// Static binary add, tile both dims.

func.func @tile_elementwise(%A: tensor<128x256xf32>, %B: tensor<128x256xf32>,
                            %C: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %r = linalg.elementwise kind=#linalg.elementwise_kind<add>
      ins(%A, %B : tensor<128x256xf32>, tensor<128x256xf32>)
      outs(%C : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %r : tensor<128x256xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.elementwise"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop0, %loop1 = transform.structured.tile_using_for %op tile_sizes [32, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @tile_elementwise(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]: tensor<128x256xf32>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]: tensor<128x256xf32>
// CHECK-SAME:    %[[C:[a-zA-Z0-9]+]]: tensor<128x256xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//  CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//  CHECK-DAG:   %[[C256:.+]] = arith.constant 256 : index
//      CHECK:   scf.for %[[IV0:.+]] = %[[C0]] to %[[C128]] step %[[C32]]
// CHECK-SAME:       iter_args(%[[INIT0:.+]] = %[[C]])
//      CHECK:     scf.for %[[IV1:.+]] = %[[C0]] to %[[C256]] step %[[C64]]
// CHECK-SAME:         iter_args(%[[INIT1:.+]] = %[[INIT0]])
//  CHECK-DAG:       %[[AT:.+]] = tensor.extract_slice %[[A]][%[[IV0]], %[[IV1]]] [32, 64] [1, 1]
//  CHECK-DAG:       %[[BT:.+]] = tensor.extract_slice %[[B]][%[[IV0]], %[[IV1]]] [32, 64] [1, 1]
//  CHECK-DAG:       %[[CT:.+]] = tensor.extract_slice %[[INIT1]][%[[IV0]], %[[IV1]]] [32, 64] [1, 1]
//      CHECK:       %[[TILED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME:           ins(%[[AT]], %[[BT]] :
// CHECK-SAME:           outs(%[[CT]] :
//      CHECK:       %[[INS:.+]] = tensor.insert_slice %[[TILED]] into %[[INIT1]]
// CHECK-SAME:           [%[[IV0]], %[[IV1]]] [32, 64] [1, 1]
//      CHECK:       scf.yield %[[INS]]
//      CHECK:     scf.yield

// -----

// Dynamic binary add.

func.func @tile_elementwise_dynamic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                                    %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %r = linalg.elementwise kind=#linalg.elementwise_kind<add>
      ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.elementwise"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop0, %loop1 = transform.structured.tile_using_for %op tile_sizes [10, 20]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 10)>
//  CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 20)>
// CHECK-LABEL: func.func @tile_elementwise_dynamic(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[C:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[A]], %[[C0]]
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[A]], %[[C1]]
//  CHECK-DAG:   %[[C10:.+]] = arith.constant 10 : index
//  CHECK-DAG:   %[[C20:.+]] = arith.constant 20 : index
//      CHECK:   scf.for %[[IV0:.+]] = %[[C0]] to %[[D0]] step %[[C10]]
// CHECK-SAME:       iter_args(%[[INIT0:.+]] = %[[C]])
//      CHECK:     scf.for %[[IV1:.+]] = %[[C0]] to %[[D1]] step %[[C20]]
// CHECK-SAME:         iter_args(%[[INIT1:.+]] = %[[INIT0]])
//      CHECK:       %[[TS0:.+]] = affine.min #[[$MAP0]](%[[IV0]])[%[[D0]]]
//      CHECK:       %[[TS1:.+]] = affine.min #[[$MAP1]](%[[IV1]])[%[[D1]]]
//  CHECK-DAG:       %[[AT:.+]] = tensor.extract_slice %[[A]][%[[IV0]], %[[IV1]]] [%[[TS0]], %[[TS1]]] [1, 1]
//  CHECK-DAG:       %[[BT:.+]] = tensor.extract_slice %[[B]][%[[IV0]], %[[IV1]]] [%[[TS0]], %[[TS1]]] [1, 1]
//  CHECK-DAG:       %[[CT:.+]] = tensor.extract_slice %[[INIT1]][%[[IV0]], %[[IV1]]] [%[[TS0]], %[[TS1]]] [1, 1]
//      CHECK:       %[[TILED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME:           ins(%[[AT]], %[[BT]] :
// CHECK-SAME:           outs(%[[CT]] :
//      CHECK:       tensor.insert_slice %[[TILED]] into %[[INIT1]]
// CHECK-SAME:           [%[[IV0]], %[[IV1]]] [%[[TS0]], %[[TS1]]] [1, 1]

// -----

// Memref variant: no iter_args, uses memref.subview instead of tensor.extract_slice.

func.func @tile_elementwise_memref(%A: memref<128x256xf32>,
                                   %B: memref<128x256xf32>) {
  linalg.elementwise kind=#linalg.elementwise_kind<negf>
      ins(%A : memref<128x256xf32>)
      outs(%B : memref<128x256xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.elementwise"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop0, %loop1 = transform.structured.tile_using_for %op tile_sizes [32, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @tile_elementwise_memref(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]: memref<128x256xf32>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]: memref<128x256xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//  CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//  CHECK-DAG:   %[[C256:.+]] = arith.constant 256 : index
//      CHECK:   scf.for %[[IV0:.+]] = %[[C0]] to %[[C128]] step %[[C32]]
//  CHECK-NOT:     iter_args
//      CHECK:     scf.for %[[IV1:.+]] = %[[C0]] to %[[C256]] step %[[C64]]
//  CHECK-DAG:       %[[AT:.+]] = memref.subview %[[A]][%[[IV0]], %[[IV1]]] [32, 64] [1, 1]
//  CHECK-DAG:       %[[BT:.+]] = memref.subview %[[B]][%[[IV0]], %[[IV1]]] [32, 64] [1, 1]
//      CHECK:       linalg.elementwise kind=#linalg.elementwise_kind<negf>
// CHECK-SAME:           ins(%[[AT]] :
// CHECK-SAME:           outs(%[[BT]] :

// -----

// Parallel tiling with scf.forall: produces parallel_insert_slice.

func.func @tile_elementwise_forall(%A: tensor<128x256xf32>, %B: tensor<128x256xf32>,
                                   %C: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %r = linalg.elementwise kind=#linalg.elementwise_kind<add>
      ins(%A, %B : tensor<128x256xf32>, tensor<128x256xf32>)
      outs(%C : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %r : tensor<128x256xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.elementwise"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %forall = transform.structured.tile_using_forall %op tile_sizes [32, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

//  CHECK-DAG: #[[$MAPX:.+]] = affine_map<(d0) -> (d0 * 32)>
//  CHECK-DAG: #[[$MAPY:.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-LABEL: func.func @tile_elementwise_forall(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]: tensor<128x256xf32>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]: tensor<128x256xf32>
// CHECK-SAME:    %[[C:[a-zA-Z0-9]+]]: tensor<128x256xf32>
//      CHECK:   %[[RESULT:.+]] = scf.forall (%[[IV0:.+]], %[[IV1:.+]]) in (4, 4)
// CHECK-SAME:       shared_outs(%[[INIT:.+]] = %[[C]])
//  CHECK-DAG:     %[[OFF0:.+]] = affine.apply #[[$MAPX]](%[[IV0]])
//  CHECK-DAG:     %[[OFF1:.+]] = affine.apply #[[$MAPY]](%[[IV1]])
//  CHECK-DAG:     %[[AT:.+]] = tensor.extract_slice %[[A]][%[[OFF0]], %[[OFF1]]] [32, 64] [1, 1]
//  CHECK-DAG:     %[[BT:.+]] = tensor.extract_slice %[[B]][%[[OFF0]], %[[OFF1]]] [32, 64] [1, 1]
//  CHECK-DAG:     %[[CT:.+]] = tensor.extract_slice %[[INIT]][%[[OFF0]], %[[OFF1]]] [32, 64] [1, 1]
//      CHECK:     %[[TILED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME:         ins(%[[AT]], %[[BT]] :
// CHECK-SAME:         outs(%[[CT]] :
//      CHECK:     scf.forall.in_parallel
//      CHECK:       tensor.parallel_insert_slice %[[TILED]] into %[[INIT]]
// CHECK-SAME:           [%[[OFF0]], %[[OFF1]]] [32, 64] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

// Broadcast: non-identity indexing map.  The input has rank 1 (only d1) so
// tiling along d0 does not slice the input at all.

#map_in  = affine_map<(d0, d1) -> (d1)>
#map_out = affine_map<(d0, d1) -> (d0, d1)>

func.func @tile_elementwise_broadcast(%A: tensor<256xf32>,
                                      %B: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %r = linalg.elementwise kind=#linalg.elementwise_kind<exp>
      indexing_maps = [#map_in, #map_out]
      ins(%A : tensor<256xf32>)
      outs(%B : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %r : tensor<128x256xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.elementwise"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop0, %loop1 = transform.structured.tile_using_for %op tile_sizes [32, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @tile_elementwise_broadcast(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]: tensor<256xf32>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]: tensor<128x256xf32>
//      CHECK:   scf.for %[[IV0:[a-zA-Z0-9]+]] =
// CHECK-SAME:       iter_args(%[[INIT0:[a-zA-Z0-9]+]] = %[[B]])
//      CHECK:     scf.for %[[IV1:[a-zA-Z0-9]+]] =
// CHECK-SAME:         iter_args(%[[INIT1:[a-zA-Z0-9]+]] = %[[INIT0]])
// Input is 1-D: sliced only along d1, not d0.
//      CHECK:       %[[AT:.+]] = tensor.extract_slice %[[A]][%[[IV1]]] [64] [1]
//      CHECK:       %[[BT:.+]] = tensor.extract_slice %[[INIT1]][%[[IV0]], %[[IV1]]] [32, 64] [1, 1]
//      CHECK:       %[[TILED:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<exp>
// CHECK-SAME:           ins(%[[AT]] : tensor<64xf32>)
// CHECK-SAME:           outs(%[[BT]] : tensor<32x64xf32>)
//      CHECK:       tensor.insert_slice %[[TILED]] into %[[INIT1]]
// CHECK-SAME:           [%[[IV0]], %[[IV1]]] [32, 64] [1, 1]

// -----

// Tile-and-fuse: exp producer is fused into the tiled add consumer.

func.func @tile_and_fuse_elementwise(%A: tensor<128x256xf32>,
                                     %B: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %empty0 = tensor.empty() : tensor<128x256xf32>
  %exp = linalg.elementwise kind=#linalg.elementwise_kind<exp>
      ins(%A : tensor<128x256xf32>)
      outs(%empty0 : tensor<128x256xf32>) -> tensor<128x256xf32>
  %empty1 = tensor.empty() : tensor<128x256xf32>
  %r = linalg.elementwise kind=#linalg.elementwise_kind<add>
      ins(%exp, %B : tensor<128x256xf32>, tensor<128x256xf32>)
      outs(%empty1 : tensor<128x256xf32>) -> tensor<128x256xf32>
  return %r : tensor<128x256xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %add = transform.structured.match ops{["linalg.elementwise"]}
        attributes{kind = #linalg.elementwise_kind<add>} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop0, %loop1 = transform.structured.fuse %add tile_sizes [32, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @tile_and_fuse_elementwise(
// CHECK-SAME:    %[[A:[a-zA-Z0-9]+]]: tensor<128x256xf32>
// CHECK-SAME:    %[[B:[a-zA-Z0-9]+]]: tensor<128x256xf32>
//      CHECK:   %[[EMPTY:.+]] = tensor.empty()
//      CHECK:   scf.for %[[IV0:[a-zA-Z0-9]+]] =
// CHECK-SAME:       iter_args(%[[INIT0:[a-zA-Z0-9]+]] = %[[EMPTY]])
//      CHECK:     scf.for %[[IV1:[a-zA-Z0-9]+]] =
// CHECK-SAME:         iter_args(%[[INIT1:[a-zA-Z0-9]+]] = %[[INIT0]])
// exp tile is generated from the original input, not the full exp result.
//  CHECK-DAG:       %[[AT:.+]] = tensor.extract_slice %[[A]][%[[IV0]], %[[IV1]]] [32, 64] [1, 1]
//      CHECK:       %[[EXP_TILE:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<exp>
// CHECK-SAME:           ins(%[[AT]] :
//  CHECK-DAG:       %[[BT:.+]] = tensor.extract_slice %[[B]][%[[IV0]], %[[IV1]]] [32, 64] [1, 1]
//      CHECK:       %[[ADD_TILE:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME:           ins(%[[EXP_TILE]], %[[BT]] :
//      CHECK:       tensor.insert_slice %[[ADD_TILE]] into %[[INIT1]]
// CHECK-SAME:           [%[[IV0]], %[[IV1]]] [32, 64] [1, 1]
