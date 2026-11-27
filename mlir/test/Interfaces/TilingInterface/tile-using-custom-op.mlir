// RUN: mlir-opt --transform-interpreter --cse --split-input-file --mlir-print-local-scope %s | FileCheck %s

module {
  func.func @generic_parallel(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%d0, %d1) : tensor<?x?xf32>
    %generic = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%empty : tensor<?x?xf32>) {
      ^bb(%b0 : f32, %b1 : f32, %b2 : f32):
        %add = arith.addf %b0, %b1 : f32
        linalg.yield %add : f32
    } -> tensor<?x?xf32>
    return %generic : tensor<?x?xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops {["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %tiled_op, %loop = transform.test.tile_using_custom_loop %op tile_sizes = [10, 20]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func @generic_parallel
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?xf32>
//   CHECK-DAG:   %[[NITERS0:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 10)>()[%[[D0]]]
//   CHECK-DAG:   %[[NITERS1:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 20)>()[%[[D1]]]
//   CHECK-DAG:   %[[NITERS:.+]] = affine.apply affine_map<()[s0, s1] -> ((s0 ceildiv 10) * (s1 ceildiv 20))>()[%[[D0]], %[[D1]]]
//       CHECK:   %[[FOR:.+]] = scf.for %[[IV:[a-zA-Z0-9]+]] = %[[C0]] to %[[NITERS]] step %[[C1]]
//  CHECK-SAME:       iter_args(%[[INIT:.+]] = %[[EMPTY]])
//       CHECK:     %[[DELINEARIZE:.+]]:2 = affine.delinearize_index %[[IV]] into (%[[NITERS0]], %[[NITERS1]])
//   CHECK-DAG:     %[[SIZE0:.+]] = affine.min affine_map<(d0)[s0] -> (d0 * -10 + s0, 10)>(%[[DELINEARIZE]]#0)[%[[D0]]]
//   CHECK-DAG:     %[[SIZE1:.+]] = affine.min affine_map<(d0)[s0] -> (d0 * -20 + s0, 20)>(%[[DELINEARIZE]]#1)[%[[D1]]]
//   CHECK-DAG:     %[[OFFSET0:.+]] = affine.apply affine_map<(d0) -> (d0 * 10)>(%[[DELINEARIZE]]#0)
//   CHECK-DAG:     %[[OFFSET1:.+]] = affine.apply affine_map<(d0) -> (d0 * 20)>(%[[DELINEARIZE]]#1)
//   CHECK-DAG:     %[[ARG0_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[OFFSET0]], %[[OFFSET1]]] [%[[SIZE0]], %[[SIZE1]]] [1, 1]
//   CHECK-DAG:     %[[ARG1_SLICE:.+]] = tensor.extract_slice %[[ARG1]][%[[OFFSET1]]] [%[[SIZE1]]] [1]
//   CHECK-DAG:     %[[INIT_SLICE:.+]] = tensor.extract_slice %[[INIT]][%[[OFFSET0]], %[[OFFSET1]]] [%[[SIZE0]], %[[SIZE1]]] [1, 1]
//       CHECK:     %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[ARG0_SLICE]], %[[ARG1_SLICE]] :
//  CHECK-SAME:         outs(%[[INIT_SLICE]] :
//       CHECK:     %[[INSERT_SLICE:.+]] = tensor.insert_slice %[[GENERIC]] into %[[INIT]]
//  CHECK-SAME:         [%[[OFFSET0]], %[[OFFSET1]]] [%[[SIZE0]], %[[SIZE1]]] [1, 1]
//       CHECK:     scf.yield %[[INSERT_SLICE]]
//       CHECK:   return %[[FOR]]
