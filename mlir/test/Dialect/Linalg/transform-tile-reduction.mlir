// RUN: mlir-opt %s -test-transform-dialect-interpreter -split-input-file -canonicalize | FileCheck %s

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

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
  %1, %2, %3 = transform.structured.tile_reduction_using_scf %0 { tile_sizes = [0, 5] }
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
//     CHECK:     %[[D3:.*]] = tensor.dim %[[PR]], %[[C0]] : tensor<?x?xf32>
//     CHECK:     %[[D4:.*]] = tensor.dim %[[PR]], %[[C1]] : tensor<?x?xf32>
//     CHECK:     %[[INS:.*]] = tensor.insert_slice %[[PR]] into %[[ARG3]][0, 0] [%[[D3]], %[[D4]]] [1, 1] : tensor<?x?xf32> into tensor<?x5xf32>
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

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
  %1, %2, %3 = transform.structured.tile_reduction_using_scf %0 { tile_sizes = [5, 0] }
}

//     CHECK: func @reduction_tile_transpose
//     CHECK:   tensor.empty(%{{.*}}) : tensor<5x?xf32>
//     CHECK:   linalg.fill {{.*}} : tensor<5x?xf32>) -> tensor<5x?xf32>
//     CHECK:   scf.for
//     CHECK:     linalg.generic
//     CHECK:     %[[D3:.*]] = tensor.dim %{{.*}}, %[[C0]] : tensor<?x?xf32>
//     CHECK:     %[[D4:.*]] = tensor.dim %{{.*}}, %[[C1]] : tensor<?x?xf32>
//     CHECK:     %[[INS:.*]] = tensor.insert_slice %[[PR]] into %[[ARG3]][0, 0] [%[[D3]], %[[D4]]] [1, 1] : tensor<?x?xf32> into tensor<5x?xf32>
//     CHECK:     scf.yield {{.*}} : tensor<5x?xf32>
//     CHECK:   }
//     CHECK:   linalg.generic 
//     CHECK:   return
