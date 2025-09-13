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

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 5)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
//     CHECK: func @reduction_tile(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?xf32>
// CHECK-DAG:   %[[I:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK-DAG:   %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
//     CHECK:   %[[E:.*]] = tensor.empty(%[[D0]]) : tensor<?x5xf32>
//     CHECK:   %[[F:.*]] = linalg.fill ins(%[[I]] : f32) outs(%[[E]] : tensor<?x5xf32>) -> tensor<?x5xf32>
//     CHECK:   %[[L:.*]] = scf.for %[[K:.*]] = %[[C0]] to %[[D1]] step %[[C5]] iter_args(%[[ARG3:.*]] = %[[F]]) -> (tensor<?x5xf32>) {
//     CHECK:     %[[PS:.*]] = affine.min #[[MAP0]](%[[K]])[%[[D1]]]
//     CHECK:     %[[EXT2:.*]] = tensor.extract_slice %[[ARG0]][0, %[[K:.*]]] [%[[D0]], %[[PS]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//     CHECK:     %[[EXT:.*]] = tensor.extract_slice %[[ARG3]][0, 0] [%[[D0]], %[[PS]]] [1, 1] : tensor<?x5xf32> to tensor<?x?xf32>
//     CHECK:     %[[PR:.*]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXT2]] : tensor<?x?xf32>) outs(%[[EXT]] : tensor<?x?xf32>) {
//     CHECK:       arith.mulf
//     CHECK:       arith.addf
//     CHECK:       linalg.yield
//     CHECK:     } -> tensor<?x?xf32>
//     CHECK:     %[[INS:.*]] = tensor.insert_slice %[[PR]] into %[[ARG3]][0, 0] [%[[D0]], %[[PS]]] [1, 1] : tensor<?x?xf32> into tensor<?x5xf32>
//     CHECK:     scf.yield %[[INS]] : tensor<?x5xf32>
//     CHECK:   }
//     CHECK:   %[[R:.*]] = linalg.reduce ins(%[[L]] : tensor<?x5xf32>) outs(%[[ARG1]] : tensor<?xf32>) dimensions = [1]
//     CHECK:     arith.addf
//     CHECK:     linalg.yield
//     CHECK:   }
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
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1) -> (d1, d0)>
//     CHECK: func @reduction_tile_transpose
//     CHECK:   tensor.empty(%{{.*}}) : tensor<?x5xf32>
//     CHECK:   linalg.fill {{.*}} : tensor<?x5xf32>) -> tensor<?x5xf32>
//     CHECK:   scf.for
//     CHECK:     %[[EXT:.*]] = tensor.extract_slice %[[ARG3:.*]][0, 0] [%[[D0:.*]], %[[D1:.*]]] [1, 1] : tensor<?x5xf32> to tensor<?x?xf32>
//     CHECK:     %[[R:.*]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%[[L:.*]] : tensor<?x?xf32>) outs(%[[EXT]] : tensor<?x?xf32>)
//     CHECK:     %[[INS:.*]] = tensor.insert_slice %[[R]] into %[[ARG3]][0, 0] [%[[D0]], %[[D1]]] [1, 1] : tensor<?x?xf32> into tensor<?x5xf32>
//     CHECK:     scf.yield {{.*}} : tensor<?x5xf32>
//     CHECK:   }
//     CHECK:   linalg.reduce
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
      by num_threads = [0, 5] tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
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
//     CHECK:   %[[E:.*]] = tensor.empty(%[[D0]]) : tensor<?x5xf32>
//     CHECK:   %[[F:.*]] = linalg.fill ins(%[[I]] : f32) outs(%[[E]] : tensor<?x5xf32>) -> tensor<?x5xf32>
//     CHECK:   %[[L:.*]] = scf.forall (%[[IV:.+]]) in (5) shared_outs(%[[ARG3:.+]] = %[[F]]) -> (tensor<?x5xf32>) {
// CHECK-DAG:     %[[TS0:.+]] = affine.min #[[MAP0]](%[[IV]])[%[[D1]]]
// CHECK-DAG:     %[[TS1:.+]] = affine.max #[[MAP1]](%[[TS0]])
// CHECK-DAG:     %[[ET:.+]] = tensor.extract_slice %[[ARG3:.+]][0, %[[IV]]] [%[[D0]], 1] [1, 1] : tensor<?x5xf32> to tensor<?xf32>
// CHECK-DAG:     %[[TINDEX:.+]] = affine.apply #[[MAP2]](%[[IV]])[%[[D1]]]
// CHECK-DAG:     %[[INCHUNK:.+]] = tensor.extract_slice %[[ARG0]][0, %[[TINDEX]]] [%[[D0]], %[[TS1]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//     CHECK:     %[[PARTIAL:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]]], iterator_types = ["parallel", "reduction"]} ins(%[[INCHUNK]] : tensor<?x?xf32>) outs(%[[ET]] : tensor<?xf32>) {
//     CHECK:       arith.mulf
//     CHECK:       arith.addf
//     CHECK:       linalg.yield
//     CHECK:     } -> tensor<?xf32>
//     CHECK:     scf.forall.in_parallel {
//     CHECK:       tensor.parallel_insert_slice %[[PARTIAL]] into %[[ARG3]][0, %[[IV]]] [%[[D0]], 1] [1, 1] : tensor<?xf32> into tensor<?x5xf32>
//     CHECK:     }
//     CHECK:   }
//     CHECK:   %[[R:.*]] = linalg.reduce ins(%[[L]] : tensor<?x5xf32>) outs(%[[ARG1]] : tensor<?xf32>) dimensions = [1]
//     CHECK:   {
//     CHECK:     arith.addf
//     CHECK:     linalg.yield
//     CHECK:   }
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
      by num_threads = [0, 0, 5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0)[s0] -> (-(d0 * (s0 ceildiv 5)) + s0, s0 ceildiv 5)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0)[s0] -> (d0 * (s0 ceildiv 5))>
//     CHECK: func @matmul_tile_parallel(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
// CHECK-DAG:   %[[I:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK-DAG:   %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
// CHECK-DAG:   %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?xf32>
//     CHECK:   %[[E:.*]] = tensor.empty(%[[D0]], %[[D2]]) : tensor<?x?x5xf32>
//     CHECK:   %[[F:.*]] = linalg.fill ins(%[[I]] : f32) outs(%[[E]] : tensor<?x?x5xf32>) -> tensor<?x?x5xf32>
//     CHECK:   %[[L:.*]] = scf.forall (%[[IV:.+]]) in (5) shared_outs(%[[ARG3:.+]] = %[[F]]) -> (tensor<?x?x5xf32>) {
// CHECK-DAG:     %[[TS0:.+]] = affine.min #[[MAP0]](%[[IV]])[%[[D1]]]
// CHECK-DAG:     %[[TS1:.+]] = affine.max #[[MAP1]](%[[TS0]])
// CHECK-DAG:     %[[ET:.+]] = tensor.extract_slice %[[ARG3:.+]][0, 0, %[[IV]]] [%[[D0]], %[[D2]], 1] [1, 1, 1] : tensor<?x?x5xf32> to tensor<?x?xf32>
// CHECK-DAG:     %[[TINDEX:.+]] = affine.apply #[[MAP2]](%[[IV]])[%[[D1]]]
// CHECK-DAG:     %[[INCHUNKA:.+]] = tensor.extract_slice %[[ARG0]][0, %[[TINDEX]]] [%[[D0]], %[[TS1]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
// CHECK-DAG:     %[[INCHUNKB:.+]] = tensor.extract_slice %[[ARG1]][%[[TINDEX]], 0] [%[[TS1]], %[[D2]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//     CHECK:     %[[PARTIAL:.+]] = linalg.matmul ins(%[[INCHUNKA]], %[[INCHUNKB]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ET]] : tensor<?x?xf32>) -> tensor<?x?xf32>
//     CHECK:     scf.forall.in_parallel {
//     CHECK:       tensor.parallel_insert_slice %[[PARTIAL]] into %[[ARG3]][0, 0, %[[IV]]] [%[[D0]], %[[D2]], 1] [1, 1, 1] : tensor<?x?xf32> into tensor<?x?x5xf32>
//     CHECK:     }
//     CHECK:   }
//     CHECK:   %[[R:.*]] = linalg.reduce ins(%[[L]] : tensor<?x?x5xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) dimensions = [2]
//     CHECK:     arith.addf
//     CHECK:     linalg.yield
//     CHECK:   }
//     CHECK:   return %[[R]] : tensor<?x?xf32>

// -----

func.func @reduction_untiled_forall(
  %arg0: tensor<?x?xf32>, %out: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @below {{tiling parallel dimensions is not supported with partial reduction tiling strategies}}
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
      by num_threads = [5] tile_sizes = [3] mapping = [#gpu.thread<x>] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

module {
  func.func @fail_for_float_neutral(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    // expected-error @below {{'linalg.generic' op Failed to get an identity value for the reduction operation.}}
    %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = llvm.fmul %in, %in  : f32
      %2 = llvm.fadd %1, %out  : f32
      linalg.yield %2 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{failed to tile using partial reduction}}
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
// CHECK:   %[[OUT2:.*]] = linalg.reduce ins(%{{.*}} : tensor<4096x2x64xf32>) outs(%{{.*}} : tensor<4096xf32>)
// CHECK:  return %[[OUT2]] : tensor<4096xf32>

// -----

func.func @reduction_tile_multiple_results(%arg0: tensor<?x?xf32>, %out: tensor<?xf32>, %out2: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  %red:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                            affine_map<(d0, d1) -> (d0)>,
                                            affine_map<(d0, d1) -> (d0)>],
   iterator_types = ["parallel", "reduction"]}
   ins(%arg0 : tensor<?x?xf32>)
   outs(%out, %out2 : tensor<?xf32>, tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32, %arg9_1: f32):
      %1 = arith.mulf %arg7, %arg7 : f32
      %2 = arith.addf %1, %arg9 : f32
      %3 = arith.maximumf %1, %arg9_1 : f32
      linalg.yield %2, %3 : f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>)
  return %red#0, %red#1 : tensor<?xf32>, tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %12, %2, %3, %4, %loop = transform.structured.tile_reduction_using_for %0
      by tile_sizes = [0, 5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// CHECK: func @reduction_tile_multiple_results
// CHECK-DAG:   %[[SUM_ID:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:   %[[MAX_ID:.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:   %[[SUM_INIT:.+]] = linalg.fill ins(%[[SUM_ID]] : f32) outs(%{{.*}} : tensor<?x5xf32>) -> tensor<?x5xf32>
// CHECK-DAG:   %[[MAX_INIT:.+]] = linalg.fill ins(%[[MAX_ID]] : f32) outs(%{{.*}} : tensor<?x5xf32>) -> tensor<?x5xf32>
// CHECK:       %[[OUT:.+]]:2 = scf.for
// CHECK-SAME:            iter_args(%[[SUM:.+]] = %[[SUM_INIT]], %[[MAX:.+]] = %[[MAX_INIT]])
// CHECK:         %[[UPDATED:.*]]:2 = linalg.generic
// CHECK:         arith.mulf
// CHECK:         arith.addf
// CHECK:         arith.maximumf
// CHECK:       %[[INSERT1:.+]] = tensor.insert_slice %[[UPDATED]]#0 into %[[SUM]]
// CHECK:       %[[INSERT2:.+]] = tensor.insert_slice %[[UPDATED]]#1 into %[[MAX]]
// CHECK:       scf.yield %[[INSERT1]], %[[INSERT1]]
// CHECK:       linalg.reduce
// CHECK:         arith.addf
// CHECK:       linalg.reduce
// CHECK:         arith.maximumf

// -----

func.func @reduction_tile_multi_dim_transpose(%arg0: tensor<?x?x?xf32>, %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                          affine_map<(d0, d1, d2) -> (d2, d0)>],
   iterator_types = ["parallel", "reduction", "parallel"]}
   ins(%arg0 : tensor<?x?x?xf32>)
   outs(%out : tensor<?x?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
      %42 = arith.addf %arg7, %arg9 : f32
      linalg.yield %42 : f32
    } -> tensor<?x?xf32>
  return %red : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %2, %3, %loop = transform.structured.tile_reduction_using_for %0
      by tile_sizes = [0, 5, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
//     CHECK: func @reduction_tile_multi_dim_transpose
//     CHECK:   tensor.empty(%{{.*}}) : tensor<?x?x5xf32>
//     CHECK:   linalg.fill {{.*}} : tensor<?x?x5xf32>) -> tensor<?x?x5xf32>
//     CHECK:   scf.for
//     CHECK:     %[[K:.*]] = affine.min
//     CHECK:     %[[EXT:.*]] = tensor.extract_slice %[[ARG3:.*]][0, 0, 0] [%[[D2:.*]], %[[D0:.*]], %[[K]]] [1, 1, 1] : tensor<?x?x5xf32> to tensor<?x?x?xf32>
//     CHECK:     %[[R:.*]] = linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[L:.*]] : tensor<?x?x?xf32>) outs(%[[EXT]] : tensor<?x?x?xf32>)
//     CHECK:     %[[INS:.*]] = tensor.insert_slice %[[R]] into %[[ARG3]][0, 0, 0] [%[[D2]], %[[D0]], %[[K]]] [1, 1, 1] : tensor<?x?x?xf32> into tensor<?x?x5xf32>
//     CHECK:     scf.yield {{.*}} : tensor<?x?x5xf32>
//     CHECK:   }
//     CHECK:   linalg.reduce
//     CHECK:   return

// -----

// Check that only one of the reduction dimension can be tiled (in this case outer).

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0)>
module {
  func.func @reduction_tile_single_of_multiple_reduction_outer(
        %arg0: tensor<86x128xf32>, %arg1: tensor<4096x86x128xf32>, %arg2: tensor<4096xf32>) -> tensor<4096xf32> {
    %0 = linalg.generic {
        indexing_maps = [#map, #map1, #map2],
        iterator_types = ["parallel", "reduction", "reduction"]}
        ins(%arg0, %arg1 : tensor<86x128xf32>, tensor<4096x86x128xf32>) outs(%arg2 : tensor<4096xf32>) {
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
      %fill_op, %split_linalg_op, %combining_linalg_op, %for_op =
          transform.structured.tile_reduction_using_for %0 reduction_dims = [1] by tile_sizes = [0, 2]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}
//      CHECK: #[[INIT_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: @reduction_tile_single_of_multiple_reduction_outer(
// CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: tensor<4096xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C86:.+]] = arith.constant 86 : index
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<4096x2xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   %[[RESULT:.+]] = scf.for %[[IV:[a-zA-Z0-9]+]] = %[[C0]] to %[[C86]] step %[[C2]]
// CHECK-SAME:       iter_args(%[[ITER_ARG:.+]] = %[[FILL]])
//      CHECK:     %[[PARTIAL_RESULT:.+]] = linalg.generic
// CHECK-SAME:         indexing_maps = [#{{.+}}, #{{.+}}, #[[INIT_MAP]]]
// CHECK-SAME:         iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:         outs(%[[ITER_ARG]] :
//      CHECK:     scf.yield %[[PARTIAL_RESULT]]
//      CHECK:   %[[REDUCE:.+]] = linalg.reduce
// CHECK-SAME:       ins(%[[RESULT]] :
// CHECK-SAME:       outs(%[[INIT]] :
// CHECK-SAME:       dimensions = [1]
//      CHECK:   return %[[REDUCE]]

// -----

// Check that only one of the reduction dimension can be tiled (in this case inner).

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0)>
module {
  func.func @reduction_tile_single_of_multiple_reduction_inner(
        %arg0: tensor<86x128xf32>, %arg1: tensor<4096x86x128xf32>, %arg2: tensor<4096xf32>) -> tensor<4096xf32> {
    %0 = linalg.generic {
        indexing_maps = [#map, #map1, #map2],
        iterator_types = ["parallel", "reduction", "reduction"]}
        ins(%arg0, %arg1 : tensor<86x128xf32>, tensor<4096x86x128xf32>) outs(%arg2 : tensor<4096xf32>) {
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
      %fill_op, %split_linalg_op, %combining_linalg_op, %for_op =
          transform.structured.tile_reduction_using_for %0 reduction_dims = [2] by tile_sizes = [0, 0, 64]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}
//      CHECK: #[[INIT_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//      CHECK: @reduction_tile_single_of_multiple_reduction_inner(
// CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: tensor<4096xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<4096x64xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   %[[RESULT:.+]] = scf.for %[[IV:[a-zA-Z0-9]+]] = %[[C0]] to %[[C128]] step %[[C64]]
// CHECK-SAME:       iter_args(%[[ITER_ARG:.+]] = %[[FILL]])
//      CHECK:     %[[PARTIAL_RESULT:.+]] = linalg.generic
// CHECK-SAME:         indexing_maps = [#{{.+}}, #{{.+}}, #[[INIT_MAP]]]
// CHECK-SAME:         iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME:         outs(%[[ITER_ARG]] :
//      CHECK:     scf.yield %[[PARTIAL_RESULT]]
//      CHECK:   %[[REDUCE:.+]] = linalg.reduce
// CHECK-SAME:       ins(%[[RESULT]] :
// CHECK-SAME:       outs(%[[INIT]] :
// CHECK-SAME:       dimensions = [1]
//      CHECK:   return %[[REDUCE]]

// -----

// Check that both the reduction dimensions are tiled but the dimensions in the output are swapped.

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0)>
module {
  func.func @reduction_tile_single_of_multiple_reduction_reversed(
        %arg0: tensor<86x128xf32>, %arg1: tensor<4096x86x128xf32>, %arg2: tensor<4096xf32>) -> tensor<4096xf32> {
    %0 = linalg.generic {
        indexing_maps = [#map, #map1, #map2],
        iterator_types = ["parallel", "reduction", "reduction"]}
        ins(%arg0, %arg1 : tensor<86x128xf32>, tensor<4096x86x128xf32>) outs(%arg2 : tensor<4096xf32>) {
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
      %fill_op, %split_linalg_op, %combining_linalg_op, %for_op =
          transform.structured.tile_reduction_using_for %0 reduction_dims = [2, 1] by tile_sizes = [0, 2, 64]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}
//      CHECK: #[[INIT_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
//      CHECK: @reduction_tile_single_of_multiple_reduction_reversed(
// CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: tensor<4096xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//  CHECK-DAG:   %[[C86:.+]] = arith.constant 86 : index
//  CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<4096x64x2xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:[a-zA-Z0-9]+]] = %[[C0]] to %[[C86]] step %[[C2]]
// CHECK-SAME:       iter_args(%[[ITER_ARG:.+]] = %[[FILL]])
//      CHECK:     %[[RESULT0:.+]] = scf.for %[[IV1:[a-zA-Z0-9]+]] = %[[C0]] to %[[C128]] step %[[C64]]
// CHECK-SAME:         iter_args(%[[ITER_ARG0:.+]] = %[[ITER_ARG]])
//      CHECK:       %[[PARTIAL_RESULT:.+]] = linalg.generic
// CHECK-SAME:           indexing_maps = [#{{.+}}, #{{.+}}, #[[INIT_MAP]]]
// CHECK-SAME:           iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:           outs(%[[ITER_ARG0]] :
//      CHECK:       scf.yield %[[PARTIAL_RESULT]]
//      CHECK      scf.yield %[[RESULT0]]
//      CHECK:   %[[REDUCE:.+]] = linalg.reduce
// CHECK-SAME:       ins(%[[RESULT]] :
// CHECK-SAME:       outs(%[[INIT]] :
// CHECK-SAME:       dimensions = [1, 2]
//      CHECK: return %[[REDUCE]]

// -----

func.func @reduction_tile_parallel_using_tile_sizes(
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
//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 ceildiv 5)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 5)>
//  CHECK-DAG: #[[MAP2:.*]] = affine_map<()[s0] -> (s0 floordiv 5)>
//      CHECK: func @reduction_tile_parallel_using_tile_sizes(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?xf32>
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
//  CHECK-DAG:   %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
//  CHECK-DAG:   %[[PARALLEL_DIM:.+]] = affine.apply #[[MAP0]]()[%[[D1]]]
//      CHECK:   %[[E:.*]] = tensor.empty(%[[D0]], %[[PARALLEL_DIM]]) : tensor<?x?xf32>
//      CHECK:   %[[F:.*]] = linalg.fill
// CHECK-SAME:      outs(%[[E]] :
//      CHECK:   %[[L:.*]] = scf.forall (%[[IV:.+]]) = (0) to (%[[D1]]) step (5) shared_outs(%[[ARG3:.+]] = %[[F]])
//  CHECK-DAG:     %[[TS0:.+]] = affine.min #[[MAP1]](%[[IV]])[%[[D1]]]
//  CHECK-DAG:     %[[INIT_OFFSET:.+]] = affine.apply #[[MAP2]]()[%[[IV]]]
//  CHECK-DAG:     %[[INCHUNK:.+]] = tensor.extract_slice %[[ARG0]][0, %[[IV]]] [%[[D0]], %[[TS0]]] [1, 1]
//  CHECK-DAG:     %[[ET:.+]] = tensor.extract_slice %[[ARG3]][0, %[[INIT_OFFSET]]] [%[[D0]], 1] [1, 1]
//      CHECK:     %[[PARTIAL:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[INCHUNK]] :
// CHECK-SAME:         outs(%[[ET]] :
//      CHECK:     scf.forall.in_parallel {
//      CHECK:       tensor.parallel_insert_slice %[[PARTIAL]] into %[[ARG3]][0, %[[INIT_OFFSET]]] [%[[D0]], 1] [1, 1]
//      CHECK:     }
//      CHECK:   }
//      CHECK:   %[[R:.*]] = linalg.reduce ins(%[[L]]
// CHECK-SAME:       outs(%[[ARG1]] :
//      CHECK:   return %[[R]] : tensor<?xf32>
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %2, %3, %loop = transform.structured.tile_reduction_using_forall %0
      by tile_sizes = [0, 5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// Check that only one of the reduction dimension can be tiled (in this case inner).

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0)>
module {
  func.func @reduction_using_forall_tile_single_of_multiple_reduction_inner(
        %arg0: tensor<86x128xf32>, %arg1: tensor<4096x86x128xf32>, %arg2: tensor<4096xf32>) -> tensor<4096xf32> {
    %0 = linalg.generic {
        indexing_maps = [#map, #map1, #map2],
        iterator_types = ["parallel", "reduction", "reduction"]}
        ins(%arg0, %arg1 : tensor<86x128xf32>, tensor<4096x86x128xf32>) outs(%arg2 : tensor<4096xf32>) {
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
      %fill_op, %split_linalg_op, %combining_linalg_op, %for_op =
          transform.structured.tile_reduction_using_forall %0 reduction_dims = [2] by tile_sizes = [0, 0, 64]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}
//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 floordiv 64)>
//      CHECK: func @reduction_using_forall_tile_single_of_multiple_reduction_inner(%[[ARG0:.+]]: tensor<86x128xf32>, %[[ARG1:.+]]: tensor<4096x86x128xf32>, %[[ARG2:.+]]: tensor<4096xf32>)
//      CHECK:   %[[E:.*]] = tensor.empty() : tensor<4096x2xf32>
//      CHECK:   %[[F:.*]] = linalg.fill
// CHECK-SAME:      outs(%[[E]] :
//      CHECK:   %[[L:.*]] = scf.forall (%[[IV:.+]]) = (0) to (128) step (64) shared_outs(%[[ARG3:.+]] = %[[F]])
//  CHECK-DAG:     %[[INIT_OFFSET:.+]] = affine.apply #[[MAP0]]()[%[[IV]]]
//  CHECK-DAG:     %[[ARG0_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[IV]]] [86, 64] [1, 1]
//  CHECK-DAG:     %[[ARG1_SLICE:.+]] = tensor.extract_slice %[[ARG1]][0, 0, %[[IV]]] [4096, 86, 64] [1, 1, 1]
//  CHECK-DAG:     %[[ET:.+]] = tensor.extract_slice %[[ARG3]][0, %[[INIT_OFFSET]]] [4096, 1] [1, 1]
//      CHECK:     %[[PARTIAL:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[ARG0_SLICE]], %[[ARG1_SLICE]] :
// CHECK-SAME:         outs(%[[ET]] :
//      CHECK:     scf.forall.in_parallel {
//      CHECK:       tensor.parallel_insert_slice %[[PARTIAL]] into %[[ARG3]][0, %[[INIT_OFFSET]]] [4096, 1] [1, 1]
//      CHECK:     }
//      CHECK:   }
//      CHECK:   %[[R:.*]] = linalg.reduce ins(%[[L]]
// CHECK-SAME:       outs(%[[ARG2]] :
//      CHECK:   return %[[R]]

// -----

// Check that specifying both reduction dimensions, but setting tile size to 0 for one of them behaves consistent with specifying single reduction dimension.

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0)>
module {
  func.func @reduction_using_forall_tilesize_0_of_multiple_reduction_inner(
        %arg0: tensor<86x128xf32>, %arg1: tensor<4096x86x128xf32>, %arg2: tensor<4096xf32>) -> tensor<4096xf32> {
    %0 = linalg.generic {
        indexing_maps = [#map, #map1, #map2],
        iterator_types = ["parallel", "reduction", "reduction"]}
        ins(%arg0, %arg1 : tensor<86x128xf32>, tensor<4096x86x128xf32>) outs(%arg2 : tensor<4096xf32>) {
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
      %fill_op, %split_linalg_op, %combining_linalg_op, %for_op =
          transform.structured.tile_reduction_using_forall %0 reduction_dims = [1, 2] by tile_sizes = [0, 0, 64]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}
//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 floordiv 64)>
//      CHECK: func @reduction_using_forall_tilesize_0_of_multiple_reduction_inner(%[[ARG0:.+]]: tensor<86x128xf32>, %[[ARG1:.+]]: tensor<4096x86x128xf32>, %[[ARG2:.+]]: tensor<4096xf32>)
//      CHECK:   %[[E:.*]] = tensor.empty() : tensor<4096x2xf32>
//      CHECK:   %[[F:.*]] = linalg.fill
// CHECK-SAME:      outs(%[[E]] :
//      CHECK:   %[[L:.*]] = scf.forall (%[[IV:.+]]) = (0) to (128) step (64) shared_outs(%[[ARG3:.+]] = %[[F]])
//  CHECK-DAG:     %[[INIT_OFFSET:.+]] = affine.apply #[[MAP0]]()[%[[IV]]]
//  CHECK-DAG:     %[[ARG0_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[IV]]] [86, 64] [1, 1]
//  CHECK-DAG:     %[[ARG1_SLICE:.+]] = tensor.extract_slice %[[ARG1]][0, 0, %[[IV]]] [4096, 86, 64] [1, 1, 1]
//  CHECK-DAG:     %[[ET:.+]] = tensor.extract_slice %[[ARG3]][0, %[[INIT_OFFSET]]] [4096, 1] [1, 1]
//      CHECK:     %[[PARTIAL:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[ARG0_SLICE]], %[[ARG1_SLICE]] :
// CHECK-SAME:         outs(%[[ET]] :
//      CHECK:     scf.forall.in_parallel {
//      CHECK:       tensor.parallel_insert_slice %[[PARTIAL]] into %[[ARG3]][0, %[[INIT_OFFSET]]] [4096, 1] [1, 1]
//      CHECK:     }
//      CHECK:   }
//      CHECK:   %[[R:.*]] = linalg.reduce ins(%[[L]]
// CHECK-SAME:       outs(%[[ARG2]] :
//      CHECK:   return %[[R]]
