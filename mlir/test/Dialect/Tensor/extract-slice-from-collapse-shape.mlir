// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-rewrite-extract-slice-from-collapse-shape %s | FileCheck %s
// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns="test-rewrite-extract-slice-from-collapse-shape use-foreach" %s | FileCheck %s --check-prefix=FOREACH

func.func @extract_slice_static(%input: tensor<3x5x7x11xf32>) -> tensor<20x11xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1, 2], [3]] : tensor<3x5x7x11xf32> into tensor<105x11xf32>
  %slice = tensor.extract_slice %collapsed [0, 0] [20, 11] [1, 1] : tensor<105x11xf32> to tensor<20x11xf32>
  return %slice : tensor<20x11xf32>
}

//     CHECK: func.func @extract_slice_static(%[[arg0:.+]]:
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c20:.+]] = arith.constant 20 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[c3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[c5:.+]] = arith.constant 5 : index
// CHECK-DAG: %[[c7:.+]] = arith.constant 7 : index
// CHECK-DAG: %[[init:.+]] = tensor.empty() : tensor<20x11xf32>
// CHECK-DAG: %[[tile:.+]] = scf.for %[[iv:.+]] = %[[c0]] to %[[c20]] step %[[c1]] iter_args(%[[iterArg:.+]] = %[[init]])
//     CHECK:   %[[multiIndex:.+]]:3 = affine.delinearize_index %[[iv]] into (%[[c3]], %[[c5]], %[[c7]]
//     CHECK:   %[[slice:.+]] = tensor.extract_slice %[[arg0]][%[[multiIndex]]#0, %[[multiIndex]]#1, %[[multiIndex]]#2, 0] [1, 1, 1, 11] [1, 1, 1, 1] :
//     CHECK:   %[[sliceFlat:.+]] = tensor.collapse_shape %[[slice]] {{\[}}[0, 1, 2], [3]{{\]}} :
//     CHECK:   %[[update:.+]] = tensor.insert_slice %[[sliceFlat]] into %[[iterArg]][%[[iv]], 0] [1, 11] [1, 1] :
//     CHECK:   scf.yield %[[update]] :
//     CHECK: return %[[tile]]

//     FOREACH: func.func @extract_slice_static(%[[arg0:.+]]:
// FOREACH-DAG: %[[c3:.+]] = arith.constant 3 : index
// FOREACH-DAG: %[[c5:.+]] = arith.constant 5 : index
// FOREACH-DAG: %[[c7:.+]] = arith.constant 7 : index
// FOREACH-DAG: %[[init:.+]] = tensor.empty() : tensor<20x11xf32>
//     FOREACH: %[[tile:.+]] = scf.foreach_thread (%[[iv:.+]]) in (20) shared_outs(%[[dest:.+]] = %[[init]])
//     FOREACH:   %[[multiIndex:.+]]:3 = affine.delinearize_index %[[iv]] into (%[[c3]], %[[c5]], %[[c7]]
//     FOREACH:   %[[slice:.+]] = tensor.extract_slice %[[arg0]][%[[multiIndex]]#0, %[[multiIndex]]#1, %[[multiIndex]]#2, 0] [1, 1, 1, 11] [1, 1, 1, 1] :
//     FOREACH:   %[[sliceFlat:.+]] = tensor.collapse_shape %[[slice]] {{\[}}[0, 1, 2], [3]{{\]}} :
//     FOREACH:   perform_concurrently
// FOREACH-NEXT:   tensor.parallel_insert_slice %[[sliceFlat]] into %[[dest]][%[[iv]], 0] [1, 11] [1, 1] :
//     FOREACH: return %[[tile]]

// -----


func.func @extract_slice_static_strided(%input: tensor<3x5x7x11xf32>) -> tensor<10x5xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1, 2], [3]] : tensor<3x5x7x11xf32> into tensor<105x11xf32>
  %slice = tensor.extract_slice %collapsed [13, 0] [10, 5] [2, 2] : tensor<105x11xf32> to tensor<10x5xf32>
  return %slice : tensor<10x5xf32>
}

//     CHECK: #[[$map0:.+]] = affine_map<(d0) -> (d0 * 2 + 13)>
//     CHECK: func.func @extract_slice_static_strided(%[[arg0:.+]]:
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[c10:.+]] = arith.constant 10 : index
// CHECK-DAG: %[[c3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[c5:.+]] = arith.constant 5 : index
// CHECK-DAG: %[[c7:.+]] = arith.constant 7 : index
//     CHECK: %[[init:.+]] = tensor.empty() : tensor<10x5xf32>
//     CHECK: %[[tile:.+]] = scf.for %[[iv:.+]] = %[[c0]] to %[[c10]] step %[[c1]] iter_args(%[[iterArg:.+]] = %[[init]])
//     CHECK:   %[[inputIv:.+]] = affine.apply #[[$map0]](%[[iv]])
//     CHECK:   %[[multiIndex:.+]]:3 = affine.delinearize_index %[[inputIv]] into (%[[c3]], %[[c5]], %[[c7]]
//     CHECK:   %[[slice:.+]] = tensor.extract_slice %[[arg0]][%[[multiIndex]]#0, %[[multiIndex]]#1, %[[multiIndex]]#2, 0] [1, 1, 1, 5] [1, 1, 1, 2] :
//     CHECK:   %[[sliceFlat:.+]] = tensor.collapse_shape %[[slice]] {{\[}}[0, 1, 2], [3]{{\]}} :
//     CHECK:   %[[update:.+]] = tensor.insert_slice %[[sliceFlat]] into %[[iterArg]][%[[iv]], 0] [1, 5] [1, 1] :
//     CHECK:   scf.yield %[[update]] :
//     CHECK: return %[[tile]]


// -----


func.func @extract_slice_dynamic(%input: tensor<3x?x?x11xf32>, %offt: index, %size: index) -> tensor<?x5xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1, 2], [3]] : tensor<3x?x?x11xf32> into tensor<?x11xf32>
  %slice = tensor.extract_slice %collapsed [%offt, 0] [%size, 5] [2, 2] : tensor<?x11xf32> to tensor<?x5xf32>
  return %slice : tensor<?x5xf32>
}

//     CHECK: #[[map0:.+]] = affine_map<(d0)[s0] -> (d0 * 2 + s0)>
//     CHECK: func.func @extract_slice_dynamic(%[[arg0:.+]]: tensor<{{.*}}>, %[[lb:.+]]: index, %[[sz:.+]]: index)
// CHECK-DAG:   %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[c2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[c3:.+]] = arith.constant 3 : index
//     CHECK:   %[[init:.+]] = tensor.empty(%[[sz]]) : tensor<?x5xf32>
// CHECK-DAG:   %[[d1:.+]] = tensor.dim %arg0, %[[c1]] : tensor<3x?x?x11xf32>
// CHECK-DAG:   %[[d2:.+]] = tensor.dim %arg0, %[[c2]] : tensor<3x?x?x11xf32>
//     CHECK:   %[[tile:.+]] = scf.for %[[iv:.+]] = %[[c0]] to %[[sz]] step %[[c1]] iter_args(%[[iterArg:.+]] = %[[init]])
//     CHECK:     %[[inputIv:.+]] = affine.apply #[[map0]](%[[iv]])[%[[lb]]]
//     CHECK:     %[[multiIndex:.+]]:3 = affine.delinearize_index %[[inputIv]] into (%[[c3]], %[[d1]], %[[d2]]) :
//     CHECK:     %[[slice:.+]] = tensor.extract_slice %[[arg0]][%[[multiIndex]]#0, %[[multiIndex]]#1, %[[multiIndex]]#2, 0] [1, 1, 1, 5] [1, 1, 1, 2] :
//     CHECK:     %[[sliceFlat:.+]] = tensor.collapse_shape %[[slice]] {{\[}}[0, 1, 2], [3]{{\]}} :
//     CHECK:     %[[update:.+]] = tensor.insert_slice %[[sliceFlat]] into %[[iterArg]][%[[iv]], 0] [1, 5] [1, 1] :
//     CHECK:     scf.yield %[[update]] :
//     CHECK:   return %[[tile]] :

// -----


func.func @extract_slice_dynamic_multidim(%input: tensor<3x?x?x11x?xf32>, %offt0: index, %size0: index, %offt1: index, %size1: index) -> tensor<?x?xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1, 2], [3, 4]] : tensor<3x?x?x11x?xf32> into tensor<?x?xf32>
  %slice = tensor.extract_slice %collapsed [%offt0, %offt1] [%size0, %size1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  return %slice : tensor<?x?xf32>
}

//     CHECK: #[[map0:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//     CHECK: func.func @extract_slice_dynamic_multidim(%[[arg0:.+]]: tensor<3x?x?x11x?xf32>, %[[lb1:.+]]: index, %[[sz1:.+]]: index, %[[lb2:.+]]: index, %[[sz2:.+]]: index)
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[c3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[c4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[c11:.+]] = arith.constant 11 : index
//     CHECK: %[[init:.+]] = tensor.empty(%[[sz1]], %[[sz2]]) : tensor<?x?xf32>
// CHECK-DAG: %[[d1:.+]] = tensor.dim %[[arg0]], %[[c1]] :
// CHECK-DAG: %[[d2:.+]] = tensor.dim %[[arg0]], %[[c2]] :
// CHECK-DAG: %[[d4:.+]] = tensor.dim %[[arg0]], %[[c4]] :
//     CHECK: %[[tile1:.+]] = scf.for %[[iv1:.+]] = %[[c0]] to %[[sz1]] step %[[c1]] iter_args(%[[iterArg1:.+]] = %[[init]])
//     CHECK:   %[[tile2:.+]] = scf.for %[[iv2:.+]] = %[[c0]] to %[[sz2]] step %[[c1]] iter_args(%[[iterArg2:.+]] = %[[iterArg1]])
//     CHECK:       %[[inputIv1:.+]] = affine.apply #[[map0:.+]](%[[iv1]])[%[[lb1]]]
//     CHECK:       %[[multiIndex1:.+]]:3 = affine.delinearize_index %[[inputIv1]] into (%[[c3]], %[[d1]], %[[d2]]) :
//     CHECK:       %[[inputIv2:.+]] = affine.apply #[[map0:.+]](%[[iv2]])[%[[lb2]]]
//     CHECK:       %[[multiIndex2:.+]]:2 = affine.delinearize_index %[[inputIv2]] into (%[[c11]], %[[d4]]) :
//     CHECK:       %[[slice:.+]] = tensor.extract_slice %[[arg0]][%[[multiIndex1]]#0, %[[multiIndex1]]#1, %[[multiIndex1]]#2, %[[multiIndex2]]#0, %[[multiIndex2]]#1] [1, 1, 1, 1, 1] [1, 1, 1, 1, 1] :
//     CHECK:       %[[sliceFlat:.+]] = tensor.collapse_shape %[[slice]] {{\[}}[0, 1, 2], [3, 4]{{\]}} :
//     CHECK:       %[[update:.+]] = tensor.insert_slice %[[sliceFlat]] into %[[iterArg2]][%[[iv1]], %[[iv2]]] [1, 1] [1, 1] :
//     CHECK:       scf.yield %[[update]] :
//     CHECK:     scf.yield %[[tile2]] :
//     CHECK:   return %[[tile1]] :

//     FOREACH: #[[map1:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//     FOREACH: func.func @extract_slice_dynamic_multidim(%[[arg0:.+]]: tensor<3x?x?x11x?xf32>, %[[lb1:.+]]: index, %[[sz1:.+]]: index, %[[lb2:.+]]: index, %[[sz2:.+]]: index)
// FOREACH-DAG: %[[c1:.+]] = arith.constant 1 : index
// FOREACH-DAG: %[[c2:.+]] = arith.constant 2 : index
// FOREACH-DAG: %[[c3:.+]] = arith.constant 3 : index
// FOREACH-DAG: %[[c4:.+]] = arith.constant 4 : index
// FOREACH-DAG: %[[c11:.+]] = arith.constant 11 : index
//     FOREACH:     %[[init:.+]] = tensor.empty(%[[sz1]], %[[sz2]]) : tensor<?x?xf32>
// FOREACH-DAG:     %[[d1:.+]] = tensor.dim %[[arg0]], %[[c1]] :
// FOREACH-DAG:     %[[d2:.+]] = tensor.dim %[[arg0]], %[[c2]] :
// FOREACH-DAG:     %[[d4:.+]] = tensor.dim %[[arg0]], %[[c4]] :
//     FOREACH:     %[[tile1:.+]] = scf.foreach_thread (%[[tid1:.+]], %[[tid2:.+]]) in (%[[sz1]], %[[sz2]]) shared_outs(%[[dest:.+]] = %[[init]])
// FOREACH-DAG:       %[[iv1:.+]] = affine.apply #[[map1]](%[[tid1]])[%[[lb1]]]
//     FOREACH:       %[[multiIndex1:.+]]:3 = affine.delinearize_index %[[iv1]] into (%[[c3]], %[[d1]], %[[d2]]) :
// FOREACH-DAG:       %[[iv2:.+]] = affine.apply #[[map1]](%[[tid2]])[%[[lb2]]]
//     FOREACH:       %[[multiIndex2:.+]]:2 = affine.delinearize_index %[[iv2]] into (%[[c11]], %[[d4]]) :
//     FOREACH:       %[[slice:.+]] = tensor.extract_slice %[[arg0]][%[[multiIndex1]]#0, %[[multiIndex1]]#1, %[[multiIndex1]]#2, %[[multiIndex2]]#0, %[[multiIndex2]]#1] [1, 1, 1, 1, 1] [1, 1, 1, 1, 1] :
//     FOREACH:       %[[sliceFlat:.+]] = tensor.collapse_shape %[[slice]] {{\[}}[0, 1, 2], [3, 4]{{\]}} :
//     FOREACH:       perform_concurrently
//FOREACH-NEXT:         tensor.parallel_insert_slice %[[sliceFlat]] into %[[dest]][%[[tid1]], %[[tid2]]] [1, 1] [1, 1] :

// -----

// Verifies that a linearized dimension that is not sliced does not generate a loop. Note that this
// only works for static shapes.

// CHECK: @extract_slice_non_sliced_linearized_dim(%[[arg0:.+]]: tensor<{{.*}}>,
func.func @extract_slice_non_sliced_linearized_dim(%input: tensor<3x?x?x11x2xf32>, %offt: index, %size: index) -> tensor<?x22xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1, 2], [3, 4]] : tensor<3x?x?x11x2xf32> into tensor<?x22xf32>
  %slice = tensor.extract_slice %collapsed [%offt, 0] [%size, 22] [1, 1] : tensor<?x22xf32> to tensor<?x22xf32>
  // CHECK: scf.for
  // CHECK-NOT: scf.for
  // CHECK: %[[multiIndex:.+]]:3 = affine.delinearize_index
  // CHECK: tensor.extract_slice %[[arg0]][%[[multiIndex]]#0, %[[multiIndex]]#1, %[[multiIndex]]#2, 0, 0] [1, 1, 1, 11, 2] [1, 1, 1, 1, 1]
  return %slice : tensor<?x22xf32>
}

// -----

// CHECK: @no_sliced_linearized_dims(%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: index, %[[arg2:.+]]: index
func.func @no_sliced_linearized_dims(%input: tensor<30x11x100xf32>, %offt: index, %size: index) -> tensor<330x?xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1], [2]] : tensor<30x11x100xf32> into tensor<330x100xf32>
  %slice = tensor.extract_slice %collapsed [0, %offt] [330, %size] [1, 1] : tensor<330x100xf32> to tensor<330x?xf32>
  // CHECK-NOT: scf.for
  // CHECK: %[[init:.+]] = tensor.empty(%[[arg2]])
  // CHECK: %[[e:.+]] = tensor.extract_slice %[[arg0]][0, 0, %[[arg1]]] [30, 11, %[[arg2]]] [1, 1, 1]
  // CHECK: %[[c:.+]] = tensor.collapse_shape %[[e]] {{\[}}[0, 1], [2]]
  // CHECK: %[[res:.+]] = tensor.insert_slice %[[c]] into %[[init]]
  // CHECK: return %[[res]]
  return %slice : tensor<330x?xf32>
}

// -----

// The below tests verify that a dimension which is the result of collapsing at
// most one non-unit dim is handled properly.

// CHECK: @collapse_and_slice_unit_dim(%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: index, %[[arg2:.+]]: index
func.func @collapse_and_slice_unit_dim(%input: tensor<1x11x100xf32>, %offt: index, %size: index) -> tensor<?x100xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1], [2]] : tensor<1x11x100xf32> into tensor<11x100xf32>
  %slice = tensor.extract_slice %collapsed [%offt, 0] [%size, 100] [1, 1] : tensor<11x100xf32> to tensor<?x100xf32>
  // CHECK-NOT: scf.for
  // CHECK: %[[e:.+]] = tensor.extract_slice %[[arg0]][0, 0, 0] [1, 11, 100] [1, 1, 1]
  // CHECK-SAME:           tensor<1x11x100xf32> to tensor<11x100xf32>
  // CHECK: %[[e1:.+]] = tensor.extract_slice %[[e]][%[[arg1]], 0] [%[[arg2]], 100] [1, 1]
  // CHECK-SAME:           tensor<11x100xf32> to tensor<?x100xf32>
  return %slice : tensor<?x100xf32>
}

// CHECK: @collapse_and_slice_multiple_unit_dim_dynamic(%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: index, %[[arg2:.+]]: index
func.func @collapse_and_slice_multiple_unit_dim_dynamic(%input: tensor<1x?x1x100xf32>, %offt: index, %size: index) -> tensor<?x100xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1, 2], [3]] : tensor<1x?x1x100xf32> into tensor<?x100xf32>
  %slice = tensor.extract_slice %collapsed [%offt, 0] [%size, 100] [1, 1] : tensor<?x100xf32> to tensor<?x100xf32>
  // CHECK-NOT: scf.for
  // CHECK: %[[c1:.+]] = arith.constant 1 : index
  // CHECK: %[[dim:.+]] = tensor.dim %[[arg0]], %[[c1]] :
  // CHECK: %[[e:.+]] = tensor.extract_slice %[[arg0]][0, 0, 0, 0] [1, %[[dim]], 1, 100] [1, 1, 1, 1]
  // CHECK-SAME:           tensor<1x?x1x100xf32> to tensor<?x100xf32>
  // CHECK: %[[e1:.+]] = tensor.extract_slice %[[e]][%[[arg1]], 0] [%[[arg2]], 100] [1, 1]
  // CHECK-SAME:           tensor<?x100xf32> to tensor<?x100xf32>
  return %slice : tensor<?x100xf32>
}

// CHECK: @collapse_and_slice_multiple_unit_dim_mixed(%[[arg0:.+]]: tensor<{{.*}}>, %[[arg1:.+]]: index, %[[arg2:.+]]: index
func.func @collapse_and_slice_multiple_unit_dim_mixed(%input: tensor<1x?x1x100x10xf32>, %offt: index, %size: index) -> tensor<?x?xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1, 2], [3, 4]] : tensor<1x?x1x100x10xf32> into tensor<?x1000xf32>
  %slice = tensor.extract_slice %collapsed [%offt, %offt] [%size, %size] [1, 1] : tensor<?x1000xf32> to tensor<?x?xf32>
  // CHECK-DAG: %[[c0]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1]] = arith.constant 1 : index
  // CHECK: %[[dim:.+]] = tensor.dim %[[arg0]], %[[c1]]
  // CHECK: %[[rank_reduced:.+]] = tensor.extract_slice %[[arg0]][0, 0, 0, 0, 0] [1, %[[dim]], 1, 100, 10] [1, 1, 1, 1, 1]
  // CHECK: %[[empty:.+]] = tensor.empty
  // CHECK: %[[result:.+]] = scf.for %[[iv:.+]] = %[[c0]] to %[[arg2]] step %[[c1]] iter_args(%[[ia:.+]] = %[[empty]])
  // CHECK:     %[[idx:.+]] = affine.apply
  // CHECK:     %[[multi_index:.+]] = affine.delinearize_index %[[idx]] into
  // CHECK:     %[[collapsed:.+]] = tensor.collapse_shape
  // CHECK:     %[[updated:.+]] = tensor.insert_slice
  // CHECK:     scf.yield %[[updated]]
  // CHECK: return %[[result]]
  return %slice : tensor<?x?xf32>
}

// Edge case where all collapsed dims are unit dims. This pattern can't eliminate the collapse shape,
// that should be handled by `linalg-fold-unit-extent-dims`.

// CHECK: @collapse_and_slice_multiple_all_unit_dim(%[[arg0:.+]]: tensor<{{.*}}>)
func.func @collapse_and_slice_multiple_all_unit_dim(%input: tensor<1x1x1x100xf32>) -> tensor<1x100xf32> {
  %collapsed = tensor.collapse_shape %input [[0, 1, 2], [3]] : tensor<1x1x1x100xf32> into tensor<1x100xf32>
  %slice = tensor.extract_slice %collapsed [0, 0] [1, 100] [1, 1] : tensor<1x100xf32> to tensor<1x100xf32>
  return %slice : tensor<1x100xf32>
  // CHECK: %[[collapse:.+]] = tensor.collapse_shape %[[arg0]] {{\[}}[0, 1, 2], [3]] : tensor<1x1x1x100xf32> into tensor<1x100xf32>
  // CHECK: return %[[collapse]]
}
