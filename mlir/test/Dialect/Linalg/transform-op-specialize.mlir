// RUN: mlir-opt --transform-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>

func.func @not_a_copy_expect_no_match(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) {
  // expected-note @below {{when applied to this op}}
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<?x?xf32>) outs(%arg1 : memref<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
  }
  return
}

func.func @transpose_op_expect_no_match(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) {
  // expected-note @below {{when applied to this op}}
  linalg.generic {
    indexing_maps = [#map, #map2], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0 : memref<?x?xf32>) outs(%arg1 : memref<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}

func.func @copy_with_up_cast(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf32>) {
  // expected-note @below {{when applied to this op}}
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0 : memref<?x?xf16>) outs(%arg1 : memref<?x?xf32>) {
    ^bb0(%in: f16, %out: f32):
      %0 = arith.extf %in : f16 to f32
      linalg.yield %0 : f32
  }
  return
}

func.func @copy_with_down_cast(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf16>) {
  // expected-note @below {{when applied to this op}}
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0 : memref<?x?xf32>) outs(%arg1 : memref<?x?xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{failed to apply}}
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @specialize_trivial_copy_memref(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0 : memref<?x?xf32>) outs(%arg1 : memref<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}

// CHECK-LABEL: specialize_trivial_copy_memref
// CHECK-SAME: %[[ARG0:.+]]: memref<?x?xf32>, %[[ARG1:.+]]: memref<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.copy ins(%[[ARG0]] : memref<?x?xf32>) outs(%[[ARG1]] : memref<?x?xf32>)

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @specialize_trivial_copy_tensor(%arg0: tensor<?x?x?xf32>, 
    %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map1, #map1], 
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: specialize_trivial_copy_tensor
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: %{{.+}} = linalg.copy ins(%[[ARG0]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>)

func.func @already_trivial_copy_memref(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) {
  linalg.copy ins(%arg0: memref<?x?xf32>) outs(%arg1: memref<?x?xf32>)
  return
}

// CHECK-LABEL: already_trivial_copy_memref
// CHECK-SAME: %[[ARG0:.+]]: memref<?x?xf32>, %[[ARG1:.+]]: memref<?x?xf32>
// CHECK: linalg.copy ins(%[[ARG0]] : memref<?x?xf32>) outs(%[[ARG1]] : memref<?x?xf32>)

func.func @already_trivial_copy_tensor(%arg0: tensor<?x?x?xf32>,
    %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.copy ins(%arg0: tensor<?x?x?xf32>) outs(%arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: already_trivial_copy_tensor
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?x?xf32>
// CHECK: %{{.+}} = linalg.copy ins(%[[ARG0]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @linalg_generic_fill(%arg0: tensor<7x7xf32>) -> tensor<7x7xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : f32) outs(%arg0 : tensor<7x7xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<7x7xf32>
  return %0 : tensor<7x7xf32>
}
// CHECK-LABEL: linalg_generic_fill
// CHECK-SAME: %[[ARG0:.+]]: tensor<7x7xf32>) -> tensor<7x7xf32>
// CHECK:  %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %{{.*}} = linalg.fill ins(%[[CST]] : f32) outs(%[[ARG0]] : tensor<7x7xf32>) -> tensor<7x7xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
