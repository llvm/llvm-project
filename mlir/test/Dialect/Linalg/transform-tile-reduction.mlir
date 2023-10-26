// RUN: mlir-opt %s -transform-interpreter -split-input-file -canonicalize -cse -verify-diagnostics | FileCheck %s

func.func @reduction_tile(%arg0: tensor<?x?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
   iterator_types = ["parallel", "reduction"]}
   ins(%arg0 : tensor<?x?xf32>)
   outs(%out : tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
      %1 = arith.mulf %arg7, %arg7 : f32
      %2 = arith.addf %1, %arg9 : f32
      linalg.yield %2 : f32
    } -> tensor<?xf32>
  return %red : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %2, %3, %loop = transform.structured.tile_reduction_using_for %0
      by tile_sizes = [0, 5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 5)>
//     CHECK: func @reduction_tile(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?xf32>
// CHECK-DAG:   %[[I:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK-DAG:   %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
// CHECK-DAG:   %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?xf32>
//     CHECK:   %[[E:.*]] = tensor.empty(%[[D2]]) : tensor<?x5xf32>
//     CHECK:   %[[F:.*]] = linalg.fill ins(%[[I]] : f32) outs(%[[E]] : tensor<?x5xf32>) -> tensor<?x5xf32>
//     CHECK:   %[[L:.*]] = scf.for %[[K:.*]] = %[[C0]] to %[[D1]] step %[[C5]] iter_args(%[[ARG3:.*]] = %[[F]]) -> (tensor<?x5xf32>) {
//     CHECK:     %[[PS:.*]] = affine.min #[[MAP2]](%[[K]])[%[[D1]]]
//     CHECK:     %[[EXT2:.*]] = tensor.extract_slice %[[ARG0]][0, %[[K:.*]]] [%[[D0]], %[[PS]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//     CHECK:     %[[EXT:.*]] = tensor.extract_slice %[[ARG3]][0, 0] [%[[D0]], %[[PS]]] [1, 1] : tensor<?x5xf32> to tensor<?x?xf32>
//     CHECK:     %[[PR:.*]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP0]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXT2]] : tensor<?x?xf32>) outs(%[[EXT]] : tensor<?x?xf32>) {
//     CHECK:       arith.mulf
//     CHECK:       arith.addf
//     CHECK:       linalg.yield
//     CHECK:     } -> tensor<?x?xf32>
//     CHECK:     %[[INS:.*]] = tensor.insert_slice %[[PR]] into %[[ARG3]][0, 0] [%[[D0]], %[[PS]]] [1, 1] : tensor<?x?xf32> into tensor<?x5xf32>
//     CHECK:     scf.yield %[[INS]] : tensor<?x5xf32>
//     CHECK:   }
//     CHECK:   %[[R:.*]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]]], iterator_types = ["parallel", "reduction"]} ins(%[[L]] : tensor<?x5xf32>) outs(%[[ARG1]] : tensor<?xf32>) {
//     CHECK:     arith.addf
//     CHECK:     linalg.yield
//     CHECK:   } -> tensor<?xf32>
//     CHECK:   return %[[R]] : tensor<?xf32>

// -----

func.func @reduction_tile_transpose(%arg0: tensor<?x?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d1)>],
   iterator_types = ["reduction", "parallel"]}
   ins(%arg0 : tensor<?x?xf32>)
   outs(%out : tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
      %42 = arith.addf %arg7, %arg9 : f32
      linalg.yield %42 : f32
    } -> tensor<?xf32>
  return %red : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %2, %3, %loop = transform.structured.tile_reduction_using_for %0
      by tile_sizes = [5, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 5)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1) -> (d1)>
//     CHECK: func @reduction_tile_transpose
//     CHECK:   tensor.empty(%{{.*}}) : tensor<5x?xf32>
//     CHECK:   linalg.fill {{.*}} : tensor<5x?xf32>) -> tensor<5x?xf32>
//     CHECK:   scf.for
//     CHECK:     %[[EXT:.*]] = tensor.extract_slice %[[ARG3:.*]][0, 0] [%[[D0:.*]], %[[D1:.*]]] [1, 1] : tensor<5x?xf32> to tensor<?x?xf32>
//     CHECK:     %[[R:.*]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[L:.*]] : tensor<?x?xf32>) outs(%[[EXT]] : tensor<?x?xf32>)
//     CHECK:     %[[INS:.*]] = tensor.insert_slice %[[R]] into %[[ARG3]][0, 0] [%[[D0]], %[[D1]]] [1, 1] : tensor<?x?xf32> into tensor<5x?xf32>
//     CHECK:     scf.yield {{.*}} : tensor<5x?xf32>
//     CHECK:   }
//     CHECK:   linalg.generic
//     CHECK:   return

// -----

func.func @reduction_tile_parallel(
  %arg0: tensor<?x?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
   iterator_types = ["parallel", "reduction"]}
   ins(%arg0 : tensor<?x?xf32>)
   outs(%out : tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
      %1 = arith.mulf %arg7, %arg7 : f32
      %2 = arith.addf %1, %arg9 : f32
      linalg.yield %2 : f32
    } -> tensor<?xf32>
  return %red : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %2, %3, %loop = transform.structured.tile_reduction_using_forall %0
      by num_threads = [0, 5], tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0)[s0] -> (-(d0 * (s0 ceildiv 5)) + s0, s0 ceildiv 5)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0)[s0] -> (d0 * (s0 ceildiv 5))>
// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0, d1) -> (d0)>
//     CHECK: func @reduction_tile_parallel(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?xf32>
// CHECK-DAG:   %[[I:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK-DAG:   %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
// CHECK-DAG:   %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?xf32>
//     CHECK:   %[[E:.*]] = tensor.empty(%[[D2]]) : tensor<?x5xf32>
//     CHECK:   %[[F:.*]] = linalg.fill ins(%[[I]] : f32) outs(%[[E]] : tensor<?x5xf32>) -> tensor<?x5xf32>
//     CHECK:   %[[L:.*]] = scf.forall (%[[IV:.+]]) in (5) shared_outs(%[[ARG3:.+]] = %[[F]]) -> (tensor<?x5xf32>) {
// CHECK-DAG:     %[[TS0:.+]] = affine.min #[[MAP0]](%[[IV]])[%[[D1]]]
// CHECK-DAG:     %[[TS1:.+]] = affine.max #[[MAP1]](%[[TS0]])
// CHECK-DAG:     %[[ET:.+]] = tensor.extract_slice %[[ARG3:.+]][0, %[[IV]]] [%[[D0]], 1] [1, 1] : tensor<?x5xf32> to tensor<?xf32>
//     CHECK:     %[[TINDEX:.+]] = affine.apply #[[MAP2]](%[[IV]])[%[[D1]]]
//     CHECK:     %[[INCHUNK:.+]] = tensor.extract_slice %[[ARG0]][0, %[[TINDEX]]] [%[[D0]], %[[TS1]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//     CHECK:     %[[TEMPEXT:.+]] = tensor.extract_slice %[[ET]][0] [%[[D0]]] [1] : tensor<?xf32> to tensor<?xf32>
//     CHECK:     %[[PARTIAL:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]]], iterator_types = ["parallel", "reduction"]} ins(%[[INCHUNK]] : tensor<?x?xf32>) outs(%[[TEMPEXT]] : tensor<?xf32>) {
//     CHECK:       arith.mulf
//     CHECK:       arith.addf
//     CHECK:       linalg.yield
//     CHECK:     } -> tensor<?xf32>
//     CHECK:     scf.forall.in_parallel {
//     CHECK:       tensor.parallel_insert_slice %[[PARTIAL]] into %[[ARG3]][0, %[[IV]]] [%[[D0]], 1] [1, 1] : tensor<?xf32> into tensor<?x5xf32>
//     CHECK:     }
//     CHECK:   }
//     CHECK:   %[[R:.*]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]]], iterator_types = ["parallel", "reduction"]} ins(%[[L]] : tensor<?x5xf32>) outs(%[[ARG1]] : tensor<?xf32>) {
//     CHECK:     arith.addf
//     CHECK:     linalg.yield
//     CHECK:   } -> tensor<?xf32>
//     CHECK:   return %[[R]] : tensor<?xf32>

// -----

func.func @matmul_tile_parallel(
  %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %matmul = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%out: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %matmul : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %2, %3, %loop = transform.structured.tile_reduction_using_forall %0
      by num_threads = [0, 0, 5], tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0)[s0] -> (-(d0 * (s0 ceildiv 5)) + s0, s0 ceildiv 5)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0)[s0] -> (d0 * (s0 ceildiv 5))>
// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//     CHECK: func @matmul_tile_parallel(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
// CHECK-DAG:   %[[I:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK-DAG:   %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
// CHECK-DAG:   %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?xf32>
// CHECK-DAG:   %[[D3:.*]] = tensor.dim %[[ARG2]], %[[C0]] : tensor<?x?xf32>
// CHECK-DAG:   %[[D4:.*]] = tensor.dim %[[ARG2]], %[[C1]] : tensor<?x?xf32>
//     CHECK:   %[[E:.*]] = tensor.empty(%[[D3]], %[[D4]]) : tensor<?x?x5xf32>
//     CHECK:   %[[F:.*]] = linalg.fill ins(%[[I]] : f32) outs(%[[E]] : tensor<?x?x5xf32>) -> tensor<?x?x5xf32>
//     CHECK:   %[[L:.*]] = scf.forall (%[[IV:.+]]) in (5) shared_outs(%[[ARG3:.+]] = %[[F]]) -> (tensor<?x?x5xf32>) {
// CHECK-DAG:     %[[TS0:.+]] = affine.min #[[MAP0]](%[[IV]])[%[[D1]]]
// CHECK-DAG:     %[[TS1:.+]] = affine.max #[[MAP1]](%[[TS0]])
// CHECK-DAG:     %[[ET:.+]] = tensor.extract_slice %[[ARG3:.+]][0, 0, %[[IV]]] [%[[D0]], %[[D2]], 1] [1, 1, 1] : tensor<?x?x5xf32> to tensor<?x?xf32>
//     CHECK:     %[[TINDEX:.+]] = affine.apply #[[MAP2]](%[[IV]])[%[[D1]]]
//     CHECK:     %[[INCHUNKA:.+]] = tensor.extract_slice %[[ARG0]][0, %[[TINDEX]]] [%[[D0]], %[[TS1]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//     CHECK:     %[[INCHUNKB:.+]] = tensor.extract_slice %[[ARG1]][%[[TINDEX]], 0] [%[[TS1]], %[[D2]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//     CHECK:     %[[TEMPEXT:.+]] = tensor.extract_slice %[[ET]][0, 0] [%[[D0]], %[[D2]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//     CHECK:     %[[PARTIAL:.+]] = linalg.matmul ins(%[[INCHUNKA]], %[[INCHUNKB]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[TEMPEXT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
//     CHECK:     scf.forall.in_parallel {
//     CHECK:       tensor.parallel_insert_slice %[[PARTIAL]] into %[[ARG3]][0, 0, %[[IV]]] [%[[D0]], %[[D2]], 1] [1, 1, 1] : tensor<?x?xf32> into tensor<?x?x5xf32>
//     CHECK:     }
//     CHECK:   }
//     CHECK:   %[[R:.*]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[L]] : tensor<?x?x5xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) {
//     CHECK:     arith.addf
//     CHECK:     linalg.yield
//     CHECK:   } -> tensor<?x?xf32>
//     CHECK:   return %[[R]] : tensor<?x?xf32>

// -----

func.func @reduction_tile_parallel_cyclic_dist(
  %arg0: tensor<?x?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
   iterator_types = ["parallel", "reduction"]}
   ins(%arg0 : tensor<?x?xf32>)
   outs(%out : tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
      %1 = arith.mulf %arg7, %arg7 : f32
      %2 = arith.addf %1, %arg9 : f32
      linalg.yield %2 : f32
    } -> tensor<?xf32>
  return %red : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %2, %3, %loop = transform.structured.tile_reduction_using_forall %0
      by num_threads = [0, 5], tile_sizes = [0, 3], mapping = [#gpu.thread<x>] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 3)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 3)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1) -> (d0)>

//     CHECK: func @reduction_tile_parallel_cyclic_dist(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?xf32>
// CHECK-DAG:   %[[I:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C15:.*]] = arith.constant 15 : index
// CHECK-DAG:   %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK-DAG:   %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?xf32>
//     CHECK:   %[[E:.*]] = tensor.empty(%[[D2]]) : tensor<?x5xf32>
//     CHECK:   %[[F:.*]] = linalg.fill ins(%[[I]] : f32) outs(%[[E]] : tensor<?x5xf32>) -> tensor<?x5xf32>
//     CHECK:   %[[L:.*]] = scf.forall (%[[IV:.+]]) in (5) shared_outs(%[[ARG3:.+]] = %[[F]]) -> (tensor<?x5xf32>) {
//     CHECK:     %[[ET:.+]] = tensor.extract_slice %[[ARG3:.+]][0, %[[IV]]] [%[[D0]], 1] [1, 1] : tensor<?x5xf32> to tensor<?xf32>
//     CHECK:     %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
//     CHECK:     %[[LB:.+]] = affine.apply #[[MAP0]]()[%[[IV]]]
//     CHECK:     %[[CARRY:.+]] = scf.for %[[IV1:.+]] = %[[LB]] to %[[D1]] step %[[C15]] iter_args(%[[ACC:.+]] = %[[ET]]) -> (tensor<?xf32>) {
//     CHECK:       %[[TS0:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[D1]]]
//     CHECK:       %[[D3:.+]] = tensor.dim %[[ACC]], %[[C0]] : tensor<?xf32>
//     CHECK:       %[[INCHUNK:.+]] = tensor.extract_slice %[[ARG0]][0, %[[IV1]]] [%[[D0]], %[[TS0]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//     CHECK:       %[[TEMPEXT:.+]] = tensor.extract_slice %[[ACC]][0] [%[[D3]]] [1] : tensor<?xf32> to tensor<?xf32>
//     CHECK:       %[[PARTIAL:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP3]]], iterator_types = ["parallel", "reduction"]} ins(%[[INCHUNK]] : tensor<?x?xf32>) outs(%[[TEMPEXT]] : tensor<?xf32>) {
//     CHECK:         arith.mulf
//     CHECK:         arith.addf
//     CHECK:         linalg.yield
//     CHECK:       } -> tensor<?xf32>
//     CHECK:       %[[INS:.+]] = tensor.insert_slice %[[PARTIAL]] into %[[ACC]][0] [%[[D3]]] [1] : tensor<?xf32> into tensor<?xf32>
//     CHECK:       scf.yield %[[INS]] : tensor<?xf32>
//     CHECK:     }
//     CHECK:     scf.forall.in_parallel {
//     CHECK:       tensor.parallel_insert_slice %[[CARRY]] into %[[ARG3]][0, %[[IV]]] [%[[D0]], 1] [1, 1] : tensor<?xf32> into tensor<?x5xf32>
//     CHECK:     }
//     CHECK:   }
//     CHECK:   %[[R:.*]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP3]]], iterator_types = ["parallel", "reduction"]} ins(%[[L]] : tensor<?x5xf32>) outs(%[[ARG1]] : tensor<?xf32>) {
//     CHECK:     arith.addf
//     CHECK:     linalg.yield
//     CHECK:   } -> tensor<?xf32>
//     CHECK:   return %[[R]] : tensor<?xf32>

// -----

func.func @reduction_tile_parallel_cyclic_dist(
  %arg0: tensor<?x?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
   iterator_types = ["parallel", "reduction"]}
   ins(%arg0 : tensor<?x?xf32>)
   outs(%out : tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
      %1 = arith.mulf %arg7, %arg7 : f32
      %2 = arith.addf %1, %arg9 : f32
      linalg.yield %2 : f32
    } -> tensor<?xf32>
  return %red : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %2, %3, %loop = transform.structured.tile_reduction_using_forall %0
      by num_threads = [0, 5], tile_sizes = [0, 3], mapping = [#gpu.thread<x>] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    //      CHECK:     expecting fill
    // CHECK-NEXT:     linalg.fill
    transform.print %1 {name = "expecting fill"} : !transform.any_op
    //      CHECK:     expecting parallel reduction
    // CHECK-NEXT:     linalg.generic
    //      CHECK:     iterator_types = ["parallel", "reduction"]
    transform.print %2 {name = "expecting parallel reduction"} : !transform.any_op
    //      CHECK:     expecting parallel reduction
    // CHECK-NEXT:     linalg.generic
    //      CHECK:     iterator_types = ["parallel", "reduction"]
    transform.print %3 {name = "expecting parallel reduction"} : !transform.any_op
    transform.yield
  }
}

// -----

func.func @reduction_untiled_forall(
  %arg0: tensor<?x?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
  // expected-note @below {{target operation}}
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
   iterator_types = ["parallel", "reduction"]}
   ins(%arg0 : tensor<?x?xf32>)
   outs(%out : tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
      %1 = arith.mulf %arg7, %arg7 : f32
      %2 = arith.addf %1, %arg9 : f32
      linalg.yield %2 : f32
    } -> tensor<?xf32>
  return %red : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{could not tile reduction}}
    %1, %2, %3, %loop = transform.structured.tile_reduction_using_forall %0
      by num_threads = [5], tile_sizes = [3], mapping = [#gpu.thread<x>] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

      transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

module {
  func.func @fail_for_float_neutral(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    // expected-error @below {{'linalg.generic' op Failed to get an identity value for the reduction operation.}}
    // expected-note @below {{when applied to this op}}
    %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = llvm.fmul %in, %in  : f32
      // expected-error @below {{Unknown neutral element for:}}
      %2 = llvm.fadd %1, %out  : f32
      linalg.yield %2 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{transform.structured.tile_reduction_using_for failed to apply}}
      %fill_op, %split_linalg_op, %combining_linalg_op, %for_op = transform.structured.tile_reduction_using_for %0 by tile_sizes = [0, 5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0)>
module {
  func.func @reduction_tile_multiple_reduction(%arg0: tensor<86x128xf32>, %arg1: tensor<4096x86x128xf32>, %arg2: tensor<4096xf32>) -> tensor<4096xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<86x128xf32>, tensor<4096x86x128xf32>) outs(%arg2 : tensor<4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %1, %out : f32
      linalg.yield %2 : f32
    } -> tensor<4096xf32>
    return %0 : tensor<4096xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %fill_op, %split_linalg_op, %combining_linalg_op, %for_op = transform.structured.tile_reduction_using_for %0 by tile_sizes = [0, 2, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// CHECK: func @reduction_tile_multiple_reduction(%[[ARG0:.+]]: tensor<86x128xf32>, %[[ARG1:.+]]: tensor<4096x86x128xf32>, %[[ARG2:.+]]: tensor<4096xf32>
// CHECK:   %[[F:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<4096x2x64xf32>) -> tensor<4096x2x64xf32>
// CHECK:   %[[L0:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG3:.*]] = %[[F]]) -> (tensor<4096x2x64xf32>)
// CHECK:     %[[L1:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG4:.*]] = %[[ARG3]]) -> (tensor<4096x2x64xf32>)
// CHECK:       %[[OUT:.*]] = linalg.generic  {indexing_maps = [{{.*}}, {{.*}}, {{.*}}], iterator_types = ["parallel", "parallel", "parallel"]} ins(%{{.*}}, %{{.*}}: tensor<2x64xf32>, tensor<4096x2x64xf32>) outs(%{{.*}}: tensor<4096x2x64xf32>)
// CHECK:       scf.yield %[[OUT]] : tensor<4096x2x64xf32>
// CHECK:     scf.yield %[[L1]] : tensor<4096x2x64xf32>
// CHECK:   %[[OUT2:.*]] = linalg.generic {indexing_maps = [{{.*}}, {{.*}}], iterator_types = ["parallel", "reduction", "reduction"]} ins(%{{.*}} : tensor<4096x2x64xf32>) outs(%{{.*}} : tensor<4096xf32>)
// CHECK:  return %[[OUT2]] : tensor<4096xf32>
