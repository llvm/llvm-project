// RUN: mlir-opt --split-input-file --verify-diagnostics --tosa-infer-shapes="fold-shape-expressions" %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: mlir-opt --split-input-file --verify-diagnostics --tosa-infer-shapes="convert-function-boundaries fold-shape-expressions" %s | FileCheck %s --check-prefixes=CHECK,FUNCBOUND

// -----

// CHECK-LABEL: test_simple_shape_expression
func.func @test_simple_shape_expression(%arg0: tensor<7x12xi32>, %arg1: tensor<80xi32>, %arg2: tensor<4xi32>) -> tensor<?xi32> {
  // CHECK-NOT: tosa.dim
  // CHECK-NOT: tosa.add_shape
  // CHECK: %[[SHAPE:.+]] = tosa.const_shape {values = dense<84> : tensor<1xindex>} : () -> !tosa.shape<1>
  // CHECK: %[[RESHAPE:.+]] = tosa.reshape %arg0, %[[SHAPE]] : (tensor<7x12xi32>, !tosa.shape<1>) -> tensor<84xi32>
  // CHECK: %[[TILE:.+]] = tosa.tile %[[RESHAPE]], %[[SHAPE]] : (tensor<84xi32>, !tosa.shape<1>) -> tensor<7056xi32>
  // DEFAULT: %[[CAST:.+]] = tensor.cast %[[TILE]] : tensor<7056xi32> to tensor<?xi32>
  // DEFAULT: return %[[CAST]] : tensor<?xi32>
  // FUNCBOUND-NOT: tensor.cast
  // FUNCBOUND: return %[[TILE]] : tensor<7056xi32>
  %a = tosa.dim %arg1 {axis = 0: i32} : (tensor<80xi32>) -> !tosa.shape<1>
  %b = tosa.dim %arg2 {axis = 0: i32} : (tensor<4xi32>) -> !tosa.shape<1>
  %c = tosa.add_shape %a, %b : (!tosa.shape<1>, !tosa.shape<1>) -> !tosa.shape<1>
  %d = tosa.reshape %arg0, %c : (tensor<7x12xi32>, !tosa.shape<1>) -> tensor<?xi32>
  %e = tosa.dim %d {axis = 0: i32} : (tensor<?xi32>) -> !tosa.shape<1>
  %f = tosa.tile %d, %e : (tensor<?xi32>, !tosa.shape<1>) -> tensor<?xi32>
  return %f : tensor<?xi32>
}

// -----

// CHECK-LABEL: test_cond_if_with_shape_expressions
func.func @test_cond_if_with_shape_expressions(%arg0 : tensor<3xf32>, %arg1 : tensor<3xf32>, %arg2 : tensor<i1>) -> () {
  // CHECK: %[[CONST_SHAPE:.*]] = tosa.const_shape {values = dense<3> : tensor<1xindex>} : () -> !tosa.shape<1>
  // CHECK: tosa.cond_if %arg2 (%arg3 = %arg0, %arg4 = %arg1) : tensor<i1> (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32> {
  %0 = tosa.cond_if %arg2 (%arg3 = %arg0, %arg4 = %arg1) : tensor<i1> (tensor<3xf32>, tensor<3xf32>) -> tensor<?xf32> {
    // CHECK: ^bb0(%arg3: tensor<3xf32>, %arg4: tensor<3xf32>)
    ^bb0(%arg3 : tensor<?xf32>, %arg4 : tensor<?xf32>):
      // CHECK-NOT: tosa.dim
      %0 = tosa.dim %arg3 {axis = 0 : i32} : (tensor<?xf32>) -> !tosa.shape<1>
      // CHECK: %[[RESHAPE:.*]] = tosa.reshape %arg3, %[[CONST_SHAPE]] : (tensor<3xf32>, !tosa.shape<1>) -> tensor<3xf32>
      %1 = tosa.reshape %arg3, %0 : (tensor<?xf32>, !tosa.shape<1>) -> tensor<?xf32>
      // CHECK: tosa.yield %[[RESHAPE]] : tensor<3xf32>
      tosa.yield %1 : tensor<?xf32>
  } else {
    // CHECK: ^bb0(%arg3: tensor<3xf32>, %arg4: tensor<3xf32>)
    ^bb0(%arg3 : tensor<?xf32>, %arg4 : tensor<?xf32>):
      // CHECK: tosa.yield %arg4 : tensor<3xf32>
      tosa.yield %arg4 : tensor<?xf32>
  }
  return
}

// -----

// CHECK-LABEL: test_no_fold_shape_expression
func.func @test_no_fold_shape_expression(%arg0: tensor<1x?x3xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: tosa.dim
  %0 = tosa.dim %arg0 {axis = 1: i32} : (tensor<1x?x3xf32>) -> !tosa.shape<1>
  // CHECK: tosa.tile
  %1 = tosa.tile %arg1, %0 : (tensor<?xf32>, !tosa.shape<1>) -> tensor<?xf32>
  // CHECK: return %{{.*}} : tensor<?xf32>
  return %1 : tensor<?xf32>
}
