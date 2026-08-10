// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics | FileCheck %s

//       CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @interchange_generic
func.func @interchange_generic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  //      CHECK:   linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$MAP]], #[[$MAP]]
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %1 = math.exp %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.interchange %0 iterator_interchange = [1, 0] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @interchange_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-note @below {{when applied to this op}}
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{transform applied to the wrong op kind}}
    transform.structured.interchange %0 iterator_interchange = [1, 0] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @too_many_iters(%0: tensor<?x?xf32>, %1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%0: tensor<?x?xf32>) outs(%1: tensor<?x?xf32>) {
  ^bb0(%2: f32, %3: f32):
    %4 = arith.mulf %2, %2 : f32
    linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{"iterator_interchange" has length (3) different from the number of loops in the target operation (2)}}
    transform.structured.interchange %0 iterator_interchange = [2,1,0] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
