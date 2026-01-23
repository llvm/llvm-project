// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.TOSA.ArgMax - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @argmax_int(%arg0: !spirv.arm.tensor<3x28x17x17xi8>) -> (!spirv.arm.tensor<3x28x17xi32>) {
  // CHECK: {{%.*}} = spirv.Tosa.ArgMax axis = 3, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28x17xi32>
  %2 = spirv.Tosa.ArgMax axis = 3, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28x17xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<3x28x17xi32>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<3x28x17xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ArgMax - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @argmax_fp(%arg0: !spirv.arm.tensor<2x2x7x14xf32>) -> (!spirv.arm.tensor<2x2x14xi32>) {
  // CHECK: {{%.*}} = spirv.Tosa.ArgMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x2x7x14xf32> -> !spirv.arm.tensor<2x2x14xi32>
  %2 = spirv.Tosa.ArgMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x2x7x14xf32> -> !spirv.arm.tensor<2x2x14xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x2x14xi32>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<2x2x14xi32>
}
