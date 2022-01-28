// RUN: mlir-opt %s -inline | FileCheck %s

// These tests verify that regions with operations from TOSA dialect
// can be inlined.

// CHECK-LABEL: func @inlined_if_fn
// Check that both the calls and the functions are eliminated after inlining:
// CHECK-NOT: @add
// CHECK-NOT: @sub
func @inlined_if_fn(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) -> tensor<f32> {
  %0 = "tosa.cond_if"(%arg2, %arg0, %arg1) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  
    %1 = call @add(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "tosa.yield"(%1) : (tensor<f32>) -> ()
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  
    %1 = call @sub(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "tosa.yield"(%1) : (tensor<f32>) -> ()
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
func private @add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
func private @sub(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "tosa.sub"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @inlined_while_fn
func @inlined_while_fn(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<10xi32>) -> tensor<10xi32> {
  // Check that calls are inlined and functions eliminated:
  // CHECK-NOT: @while
  %1:4 = "tosa.while_loop"(%arg0, %arg1, %arg2, %arg3) ({
  ^bb0(%arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>, %arg7: tensor<10xi32>):  
    %2 = call @while_cond_40(%arg4, %arg5, %arg6, %arg7) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<10xi32>) -> tensor<i1>
    "tosa.yield"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>, %arg7: tensor<10xi32>):  
    %2:4 = call @while_body_50(%arg4, %arg5, %arg6, %arg7) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<10xi32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<10xi32>)
    "tosa.yield"(%2#0, %2#1, %2#2, %2#3) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<10xi32>) -> ()
  }) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<10xi32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<10xi32>)
  return %1#3 : tensor<10xi32>
}
func private @while_body_50(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<10xi32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<10xi32>) {
  %1 = "tosa.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tosa.add"(%arg3, %1) : (tensor<10xi32>, tensor<i32>) -> tensor<10xi32>
  return %1, %arg1, %arg2, %2: tensor<i32>, tensor<i32>, tensor<i32>, tensor<10xi32>
}
func private @while_cond_40(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<10xi32>) -> tensor<i1> {
  %0 = "tosa.greater_equal"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %1 = "tosa.logical_not"(%0) : (tensor<i1>) -> tensor<i1>
  return %1 : tensor<i1>
}
