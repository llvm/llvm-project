// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics

func.func @unsupported_non_elementwise(%arg0: memref<2x3xf32>, %arg1: memref<3x4xf32>, %arg2: memref<2x4xf32>) {
  // expected-error @below {{only elementwise flattening is supported}}
  linalg.matmul ins(%arg0, %arg1 : memref<2x3xf32>, memref<3x4xf32>) outs(%arg2: memref<2x4xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %flattened = transform.structured.flatten_elementwise %0
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @unsupported_memref(%arg0: memref<32x7xf32, strided<[7, 2]>>, %arg1: memref<32x7xf32, strided<[7, 2]>>, %arg2: memref<32x7xf32, strided<[7, 2]>>) {
  // expected-error @below {{attempted to flatten, but failed}}
  linalg.map {arith.addf} ins(%arg0, %arg1: memref<32x7xf32, strided<[7, 2]>>, memref<32x7xf32, strided<[7, 2]>>) outs(%arg2: memref<32x7xf32, strided<[7, 2]>>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %flattened = transform.structured.flatten_elementwise %0
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----
#map1 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @unsupported_broadcasting_elementwise(%arg0: tensor<1x2x1xi32>, %arg1: tensor<32x2x2xi32>) -> tensor<32x2x2xi32> {
  // expected-error @below {{broadcasting of non scalar operands is not supported}}
  %0 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x2x1xi32>) outs(%arg1 : tensor<32x2x2xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
  } -> tensor<32x2x2xi32>
  return %0 : tensor<32x2x2xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %flattened = transform.structured.flatten_elementwise %0
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @unsupported_dim_expanding_broadcast(%arg0: tensor<64xi16>, %arg1: tensor<32x64xi16>) -> tensor<32x64xi16> {
  // expected-error @below {{broadcasting of non scalar operands is not supported}}
  %broadcasted = linalg.broadcast ins(%arg0 : tensor<64xi16>) outs(%arg1 : tensor<32x64xi16>) dimensions = [0] 
  return %broadcasted : tensor<32x64xi16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %flattened = transform.structured.flatten_elementwise %0
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
