// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-experimental-input-shape="args=arg0:2x16,arg2:64x9" %s | FileCheck %s

// CHECK-LABEL: test_empty_func
func.func @test_empty_func(
        // CHECK: %arg0: tensor<2x16xi32>
        %arg0: tensor<2x?xi32>,
        // CHECK: %arg1: tensor<?x256xf32>
        %arg1: tensor<?x256xf32>,
        // CHECK: %arg2: tensor<64x9xf32>
        %arg2: tensor<?x9xf32>) -> (tensor<2x?xi32>, tensor<?x256xf32>, tensor<?x9xf32>) {
    // CHECK: %arg0, %arg1, %arg2 : tensor<2x16xi32>, tensor<?x256xf32>, tensor<64x9xf32>
    return %arg0, %arg1, %arg2 : tensor<2x?xi32>, tensor<?x256xf32>, tensor<?x9xf32>
}

// -----

// CHECK-LABEL: test_func_with_ops
func.func @test_func_with_ops(
        // CHECK: %arg0: tensor<2x16xi32>
        %arg0: tensor<2x?xi32>,
        // CHECK: %arg1: tensor<?x256xf32>
        %arg1: tensor<?x256xf32>,
        // CHECK: %arg2: tensor<64x9xf32>
        %arg2: tensor<?x9xf32>) -> (tensor<2x?xi32>, tensor<?x256xf32>, tensor<?x9xf32>) {
    // CHECK: %[[ADD:.*]] = tosa.add %arg0, %arg0 : (tensor<2x16xi32>, tensor<2x16xi32>)
    %0 = tosa.add %arg0, %arg0 : (tensor<2x?xi32>, tensor<2x?xi32>) -> tensor<2x?xi32>
    // CHECK: %[[RECIP:.*]] =  tosa.reciprocal %arg1 : (tensor<?x256xf32>)
    %1 = tosa.reciprocal %arg1 : (tensor<?x256xf32>) -> tensor<?x256xf32>
    // CHECK: %[[SUB:.*]] = tosa.sub %arg2, %arg2 : (tensor<64x9xf32>, tensor<64x9xf32>)
    %2 = tosa.sub %arg2, %arg2 : (tensor<?x9xf32>, tensor<?x9xf32>) -> tensor<?x9xf32>
    return %0, %1, %2 : tensor<2x?xi32>, tensor<?x256xf32>, tensor<?x9xf32>
}

// -----

// CHECK-LABEL: test_controlflow
func.func @test_controlflow(
        // CHECK: %arg0: tensor<2x16xi32>
        %arg0: tensor<2x?xi32>,
        // CHECK: %arg1: tensor<?x256xf32>
        %arg1: tensor<?x256xf32>,
        // CHECK: %arg2: tensor<64x9xf32>
        %arg2: tensor<?x9xf32>,
        // CHECK: %arg3: tensor<i1>
        %arg3: tensor<i1>) -> (tensor<2x?xi32>, tensor<?x256xf32>, tensor<?x9xf32>) {
    // CHECK: %[[IF:.*]]:3 = tosa.cond_if %arg3 (%arg4 = %arg0, %arg5 = %arg1, %arg6 = %arg2) : tensor<i1> (tensor<2x16xi32>, tensor<?x256xf32>, tensor<64x9xf32>) -> (tensor<2x?xi32>, tensor<?x256xf32>, tensor<?x9xf32>) {
    %0:3 = tosa.cond_if %arg3 (%arg4 = %arg0, %arg5 = %arg1, %arg6 = %arg2) : tensor<i1> (tensor<2x?xi32>, tensor<?x256xf32>, tensor<?x9xf32>) -> (tensor<2x?xi32>, tensor<?x256xf32>, tensor<?x9xf32>) {
        // CHECK: ^bb0(%arg4: tensor<2x?xi32>, %arg5: tensor<?x256xf32>, %arg6: tensor<?x9xf32>):
        ^bb0(%arg4: tensor<2x?xi32>, %arg5: tensor<?x256xf32>, %arg6: tensor<?x9xf32>):
            tosa.yield %arg4, %arg5, %arg6 : tensor<2x?xi32>, tensor<?x256xf32>, tensor<?x9xf32>
    } else {
        // CHECK: ^bb0(%arg4: tensor<2x?xi32>, %arg5: tensor<?x256xf32>, %arg6: tensor<?x9xf32>):
        ^bb0(%arg4: tensor<2x?xi32>, %arg5: tensor<?x256xf32>, %arg6: tensor<?x9xf32>):
            tosa.yield %arg4, %arg5, %arg6 : tensor<2x?xi32>, tensor<?x256xf32>, tensor<?x9xf32>
    }
    // CHECK: return %[[IF]]#0, %[[IF]]#1, %[[IF]]#2 : tensor<2x?xi32>, tensor<?x256xf32>, tensor<?x9xf32>
    return %0#0, %0#1, %0#2 : tensor<2x?xi32>, tensor<?x256xf32>, tensor<?x9xf32>
}

// -----

func.func @test_wrong_number_input_args(%arg0: tensor<2x?xf32>) -> tensor<2x?xf32> {
    // expected-error@-1 {{provided arg index 2 is larger than number of inputs 1}}
    return %arg0 : tensor<2x?xf32>
}

// -----

func.func @test_incompatible_input_shape(%arg0: tensor<1x?xf32>, %arg1: tensor<1x?xf32>, %arg2: tensor<1x?xf32>) -> tensor<1x?xf32> {
    // expected-error@-1 {{arg0 has incompatible shape with requested input shape (2, 16), got 'tensor<1x?xf32>'}}
    return %arg0 : tensor<1x?xf32>
}
