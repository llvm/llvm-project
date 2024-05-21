// RUN: mlir-opt --transform-interpreter --mlir-print-local-scope --split-input-file --verify-diagnostics --cse %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [4, 4, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func @tile_linalg_matmul(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:  -> tensor<128x128xf32> {
func.func @tile_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<128x128xf32>) {
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<128x128xf32>) {
//      CHECK:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<128x128xf32>) {
//      CHECK:       %[[sTA:.*]] = tensor.extract_slice %[[TA]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK:       %[[sTB:.*]] = tensor.extract_slice %[[TB]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK:       %[[sTC:.*]] = tensor.extract_slice %[[TC2]][{{.*}}] : tensor<128x128xf32> to tensor<4x4xf32>
//      CHECK:       %[[sTD:.*]] = linalg.matmul ins(%[[sTA]], %[[sTB]] : tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-SAME:                                   outs(%[[sTC]] : tensor<4x4xf32>)  -> tensor<4x4xf32>
//      CHECK:       %[[TD:.*]] = tensor.insert_slice %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<4x4xf32> into tensor<128x128xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<128x128xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<128x128xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<128x128xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

//      CHECK: return %[[TD0]] : tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["func.call"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [%1, %1, 4] : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

func.func private @get_dynamic_tile_size() -> index

// CHECK-LABEL: func @tile_linalg_matmul_dynamic(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:  -> tensor<128x128xf32> {
func.func @tile_linalg_matmul_dynamic(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<128x128xf32>) {
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<128x128xf32>) {
//      CHECK:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<128x128xf32>) {
//      CHECK:       %[[sTA:.*]] = tensor.extract_slice %[[TA]][{{.*}}] : tensor<128x128xf32> to tensor<?x4xf32>
//      CHECK:       %[[sTB:.*]] = tensor.extract_slice %[[TB]][{{.*}}] : tensor<128x128xf32> to tensor<4x?xf32>
//      CHECK:       %[[sTC:.*]] = tensor.extract_slice %[[TC2]][{{.*}}] : tensor<128x128xf32> to tensor<?x?xf32>
//      CHECK:       %[[sTD:.*]] = linalg.matmul ins(%[[sTA]], %[[sTB]] : tensor<?x4xf32>, tensor<4x?xf32>)
// CHECK-SAME:                                   outs(%[[sTC]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
//      CHECK:       %[[TD:.*]] = tensor.insert_slice %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<?x?xf32> into tensor<128x128xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<128x128xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<128x128xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<128x128xf32>
  %sz = func.call @get_dynamic_tile_size() : () -> index
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

//      CHECK: return %[[TD0]] : tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{for this parameter}}
    %1 = transform.test_produce_param (0 : i64) : !transform.param<i64>
    // expected-error @below {{expected as many parameter values (0) as target ops (2)}}
    transform.structured.tile_using_for %0 tile_sizes [%1, %1, %1]
      : (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
      -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

func.func @tile_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  %1 = linalg.matmul  ins(%0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0, %1 : tensor<128x128xf32>, tensor<128x128xf32>
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{for this handle}}
    %1 = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected as many dynamic size-producing operations (0) as target ops (2)}}
    transform.structured.tile_using_for %0 tile_sizes [%1, %1, 1]
      : (!transform.any_op, !transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

func.func @tile_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> (tensor<128x128xf32>, tensor<128x128xf32>) {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  %1 = linalg.matmul  ins(%0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0, %1 : tensor<128x128xf32>, tensor<128x128xf32>
}

// -----

// CHECK-LABEL: tile_tensor_pad
func.func @tile_tensor_pad(
  %arg0 : tensor<?x?xf32>, %cst : f32, %low: index, %high: index)
    -> tensor<20x40xf32>
{
  // CHECK: scf.forall
  // CHECK:   scf.if
  // CHECK:     tensor.generate
  // CHECK:   else
  // CHECK:     tensor.pad {{.*}} nofold
  %0 = tensor.pad %arg0 nofold low[%low, %low] high[%high, %high] {
        ^bb0(%arg9: index, %arg10: index):
          tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<20x40xf32>
  return %0 : tensor<20x40xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.tile_using_forall %0 tile_sizes[1, 1]
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
           transform.yield
  }
}

// -----

#map = affine_map<(d0) -> (d0)>

module {
  func.func @scalable_tile(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>, %arg3: f32) -> tensor<?xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%arg2 : tensor<?xf32>) {
    ^bb0(%in_1: f32, %in_2: f32, %out: f32):
      %1 = arith.addf %in_1, %in_2 : f32
      %2 = arith.mulf %arg3, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// CHECK-LABEL:   func.func @scalable_tile(
// CHECK-SAME:      %[[ARG_0:.*]]: tensor<?xf32>, %[[ARG_1:.*]]: tensor<?xf32>, %[[ARG_2:.*]]: tensor<?xf32>,
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[DIM:.*]] = tensor.dim %[[ARG_0]], %[[C0]] : tensor<?xf32>
// CHECK:           %[[VEC_SIZE:.*]] = arith.constant 4 : index
// CHECK:           %[[VS:.*]] = vector.vscale
// CHECK:           %[[STEP:.*]] = arith.muli %[[VEC_SIZE]], %[[VS]] : index
// CHECK:           scf.for %[[IV:.*]] = %[[C0]] to %[[DIM]] step %[[STEP]] iter_args(%[[VAL:.*]] = %[[ARG_2]]) -> (tensor<?xf32>) {
// CHECK:             %[[SIZE:.*]] = affine.min affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>(%[[IV]])[%[[STEP]], %[[DIM]]]
// CHECK:             %[[SLICE_ARG0:.*]] = tensor.extract_slice %[[ARG_0]][%[[IV]]] [%[[SIZE]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK:             %[[SLICE_ARG1:.*]] = tensor.extract_slice %[[ARG_1]][%[[IV]]] [%[[SIZE]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK:             %[[SLICE_ARG2:.*]] = tensor.extract_slice %[[VAL]][%[[IV]]] [%[[SIZE]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK:             linalg.generic {indexing_maps = {{.*}}, iterator_types = ["parallel"]} ins(%[[SLICE_ARG0]], %[[SLICE_ARG1]] : tensor<?xf32>, tensor<?xf32>) outs(%[[SLICE_ARG2]] : tensor<?xf32>) {

  module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loop = transform.structured.tile_using_for %0 tile_sizes [[4]] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
  }
  }

// -----

// CHECK-LABEL:   func.func @scalable_and_fixed_length_tile
//   CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
//   CHECK-DAG:     %[[VS:.*]] = vector.vscale
//   CHECK-DAG:     %[[STEP_2:.*]] = arith.muli %[[C4]], %[[VS]] : index
//   CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:     %[[C128:.*]] = arith.constant 128 : index
//       CHECK:     scf.for %[[VAL_11:.*]] = %[[C0]] to %[[C128]] step %[[C4]]
//       CHECK:       scf.for %[[VAL_16:.*]] = %[[C0]] to %[[C128]] step %[[C4]]
//       CHECK:         scf.for %{{.*}} = %[[C0]] to %[[C128]] step %[[STEP_2]]

func.func @scalable_and_fixed_length_tile(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  return %0 : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [4, 4, [4]] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @too_many_tiles(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>,
                          %arg2: tensor<128x128xf32>) ->  tensor<128x128xf32> {
  // expected-note @below {{target op}}
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{too many tiles provided, expected at most 3 found 4}}
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{op expected number of loops to tile (3) to match number of `loops` results (1)}}
    %1, %loops = transform.structured.tile_using_for %0 tile_sizes [4, 4, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

func.func @tile_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
