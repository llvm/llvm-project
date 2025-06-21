// RUN: mlir-opt --split-input-file --experimental-tosa-input-shape="args=arg0:2x16,arg3:64x9" %s | FileCheck %s

func.func @test_input_shape(
    // CHECK: %arg0: tensor<2x16xi32>
    %arg0: tensor<2x?xi32>,
    // CHECK: %arg1: tensor<?x256xf32>
    %arg1: tensor<?x256xf32>,
    // CHECK: %arg2: tensor<2x?xi32>
    %arg2: tensor<2x?xi32>,
    // CHECK: %arg3: tensor<64x9xf32>
    %arg3: tensor<?x9xf32>) -> (tensor<2x?xi32>, tensor<?x9xf32>) {

    // CHECK: %arg0, %arg3 : tensor<2x16xi32>, tensor<64x9xf32>
    return %arg0, %arg3 : tensor<2x?xi32>, tensor<?x9xf32>
}
