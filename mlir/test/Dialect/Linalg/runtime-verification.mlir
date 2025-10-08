// RUN: mlir-opt %s -generate-runtime-verification | FileCheck %s
// RUN: mlir-opt %s --generate-runtime-verification="verbose-level=0" | FileCheck %s --check-prefix=VERBOSE0

// Most of the tests for linalg runtime-verification are implemented as integration tests.

#identity = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @static_dims
// VERBOSE0-LABEL: @static_dims
func.func @static_dims(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> (tensor<5xf32>) {
    // CHECK: %[[TRUE:.*]] = index.bool.constant true
    // CHECK: cf.assert %[[TRUE]]
    // VERBOSE0: %[[TRUE:.*]] = index.bool.constant true
    // VERBOSE0: cf.assert %[[TRUE]]
    // VERBOSE0-SAME: ERROR: Runtime op verification failed\0A^\0ALocation: loc(
    %result = tensor.empty() : tensor<5xf32> 
    %0 = linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel"]
    } ins(%arg0, %arg1 : tensor<5xf32>, tensor<5xf32>)
      outs(%result : tensor<5xf32>) {
      ^bb0(%gen_arg1: f32, %gen_arg2: f32, %out: f32) :
        %tmp1 = arith.addf %gen_arg1, %gen_arg2 : f32
        linalg.yield %tmp1 : f32
    } -> tensor<5xf32>
    return %0 : tensor<5xf32>
}

// -----

#map = affine_map<() -> ()>

// CHECK-LABEL: @scalars
// VERBOSE1-LABEL: @scalars
func.func @scalars(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>) {
    // No runtime checks are required if the operands are all scalars
    // CHECK-NOT: cf.assert
    // VERBOSE1-NOT: cf.assert
    %result = tensor.empty() : tensor<f32> 
    %0 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = []
    } ins(%arg0, %arg1 : tensor<f32>, tensor<f32>)
      outs(%result : tensor<f32>) {
      ^bb0(%gen_arg1: f32, %gen_arg2: f32, %out: f32) :
        %tmp1 = arith.addf %gen_arg1, %gen_arg2 : f32
        linalg.yield %tmp1 : f32
    } -> tensor<f32>
    return %0 : tensor<f32>
}
