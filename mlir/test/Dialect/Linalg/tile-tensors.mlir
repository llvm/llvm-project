// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

// CHECK-LABEL: func @matmul_tensors(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
func.func @matmul_tensors(
  %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<?x?xf32>) {
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<?x?xf32>) {
//      CHECK:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<?x?xf32>) {
//      CHECK:       %[[sTA:.*]] = tensor.extract_slice %[[TA]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTB:.*]] = tensor.extract_slice %[[TB]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTC:.*]] = tensor.extract_slice %[[TC2]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTD:.*]] = linalg.matmul ins(%[[sTA]], %[[sTB]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:                                  outs(%[[sTC]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
//      CHECK:       %[[TD:.*]] = tensor.insert_slice %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<?x?xf32> into tensor<?x?xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<?x?xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<?x?xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<?x?xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>

//      CHECK: return %[[TD0]] : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [2, 3, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK:       #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-NEXT:  #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-NEXT:  #[[$MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
#access_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                affine_map<(d0, d1, d2) -> (d2, d1)>,
                affine_map<(d0, d1, d2) -> (d0, d1)>]

// CHECK-LABEL: func @matmul_as_contract_tensors(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
func.func @matmul_as_contract_tensors(
  %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<?x?xf32>) {
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<?x?xf32>) {
//      CHECK:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<?x?xf32>) {
//      CHECK:       %[[sTA:.*]] = tensor.extract_slice %[[TA]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTB:.*]] = tensor.extract_slice %[[TB]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTC:.*]] = tensor.extract_slice %[[TC2]][{{.*}}] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTD:.*]] = linalg.contract
// CHECK-SAME:          indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
// CHECK-SAME:          ins(%[[sTA]], %[[sTB]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:          outs(%[[sTC]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
//      CHECK:       %[[TD:.*]] = tensor.insert_slice %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<?x?xf32> into tensor<?x?xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<?x?xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<?x?xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<?x?xf32>
  %0 = linalg.contract indexing_maps = #access_maps
                       ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                       outs(%C: tensor<?x?xf32>)
    -> tensor<?x?xf32>

//      CHECK: return %[[TD0]] : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.contract"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [2, 3, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @matmul_tensors_with_size_zeros(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
func.func @matmul_tensors_with_size_zeros(
  %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32> {

//      CHECK:     %[[RES:.*]] = linalg.matmul ins(%[[TA]], %[[TB]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:                                outs(%[[TC]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
//      CHECK:     return %[[RES]]
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.tile_using_for %0 tile_sizes [0, 0, 0] : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

func.func @generic_op_tensors(
  %arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %3 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
  %4 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d2, d1)>,
                      affine_map<(d0, d1, d2) -> (d2, d1, d0)>],
     iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs(%3 : tensor<?x?x?xf32>) {
    ^bb0(%arg2 : f32, %arg3: f32, %arg4: f32):
      %5 = arith.addf %arg2, %arg3 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?x?xf32>
  return %4 : tensor<?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [2, 3, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func @generic_op_tensors
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//       CHECK:   %[[INIT:.+]] = tensor.empty
//       CHECK:   %[[TD0:.+]] = scf.for %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[TC0:.+]] = %[[INIT]]) -> (tensor<?x?x?xf32>) {
//       CHECK:     %[[TD1:.+]] = scf.for %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[TC1:.+]] = %[[TC0]]) -> (tensor<?x?x?xf32>) {
//       CHECK:       %[[TD2:.+]] = scf.for %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[TC2:.+]] = %[[TC1]]) -> (tensor<?x?x?xf32>) {
//       CHECK:       %[[STARG0:.+]] = tensor.extract_slice %[[ARG0]][{{.+}}] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
//       CHECK:       %[[STARG1:.+]] = tensor.extract_slice %[[ARG1]][{{.+}}] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
//       CHECK:       %[[STARG2:.+]] = tensor.extract_slice %[[TC2]][{{.+}}] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
//       CHECK:       %[[STRETURN:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[STARG0]], %[[STARG1]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
//  CHECK-SAME:         outs(%[[STARG2]] : tensor<?x?x?xf32>)
//       CHECK:       %[[TD:.+]] = tensor.insert_slice %[[STRETURN]] into %[[TC2]]
//       CHECK:       scf.yield %[[TD]]
//       CHECK:     }
//       CHECK:     scf.yield %[[TD2]]
//       CHECK:   }
//       CHECK:   scf.yield %[[TD1]]
//       CHECK: }
//       CHECK: return %[[TD0]]

// -----

//  CHECK-DAG:  #[[MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>

//      CHECK:  fold_extract_slice
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<?x128xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<?x42xf32>
func.func @fold_extract_slice(
  %arg0 : tensor<?x128xf32>, %arg1 : tensor<?x42xf32>, %arg2 : tensor<?x42x?xf32>) -> tensor<?x42xf32> {

  //      CHECK:    %[[C0:.*]] = arith.constant 0
  %c0 = arith.constant 0 : index

  //      CHECK:    %[[DIM:.*]] = tensor.dim %[[ARG1]], %[[C0]]
  %0 = tensor.dim %arg1, %c0 : tensor<?x42xf32>
  %1 = tensor.extract_slice %arg0[3, 4] [%0, 42] [1, 1] : tensor<?x128xf32> to tensor<?x42xf32>

  //      CHECK:   %[[E:.*]] = tensor.extract_slice %[[ARG0]][3, 4] [%[[DIM]], 42] [1, 1] : tensor<?x128xf32> to tensor<?x42xf32>

  //      CHECK:    scf.for %[[IV0:[0-9a-zA-Z]*]] =
  //      CHECK:      scf.for %[[IV1:[0-9a-zA-Z]*]] =

  //      CHECK:      %[[SIZE0:.*]] = affine.min #[[MAP0]](%[[IV0]])[%[[DIM]]
  // Fold the existing extract slice op into the one created by the tiling.
  //      CHECK:        %[[T0:.*]] = tensor.extract_slice %[[E]]
  // CHECK-SAME:                                          %[[IV0]], %[[IV1]]
  // CHECK-SAME:                                          %[[SIZE0]], 3
  // CHECK-SAME:                                          1, 1
  //      CHECK:        {{.*}} = linalg.generic {{.*}} ins(%[[T0]]
  %2 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%1, %arg2 : tensor<?x42xf32>, tensor<?x42x?xf32>)
    outs(%arg1 : tensor<?x42xf32>) {
    ^bb0(%arg3 : f32, %arg4: f32, %arg5: f32):
      %5 = arith.addf %arg3, %arg5 : f32
      linalg.yield %5 : f32
    } -> tensor<?x42xf32>
  return %2 : tensor<?x42xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [2, 3, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
