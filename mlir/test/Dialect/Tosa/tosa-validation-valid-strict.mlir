// RUN: mlir-opt %s -split-input-file -verify-diagnostics --tosa-attach-target="profiles=pro_int,pro_fp extensions=int16,int4,bf16,fp8e4m3,fp8e5m2,fft,variable,controlflow,doubleround,inexactround" --tosa-validate="strict-op-spec-alignment" | FileCheck %s

// -----

// CHECK-LABEL: test_cond_if_isolated_from_above
func.func @test_cond_if_isolated_from_above(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) -> tensor<f32> {
  %0 = "tosa.cond_if"(%arg2, %arg0, %arg1) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      tosa.yield %arg3 : tensor<f32>
    },  {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      tosa.yield %arg4 : tensor<f32>
    }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: test_while_loop_isolated_from_above
func.func @test_while_loop_isolated_from_above(%arg0: tensor<f32>, %arg1: tensor<i32>) {
  %0 = "tosa.const"() {values = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1:3 = "tosa.while_loop"(%0, %arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<f32>, %arg5: tensor<i32>):
    %2 = "tosa.greater_equal"(%arg3, %arg5) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = "tosa.logical_not"(%2) : (tensor<i1>) -> tensor<i1>
    "tosa.yield"(%3) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<f32>, %arg5: tensor<i32>):
    %2 = "tosa.const"() {values = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.add"(%arg3, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%3, %arg4, %arg5) : (tensor<i32>, tensor<f32>, tensor<i32>) -> ()
  }) : (tensor<i32>, tensor<f32>, tensor<i32>) -> (tensor<i32>, tensor<f32>, tensor<i32>)
  return
}
