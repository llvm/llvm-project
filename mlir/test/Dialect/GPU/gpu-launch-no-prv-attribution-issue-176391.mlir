// RUN: mlir-opt %s -test-ir-visitors | FileCheck %s

// CHECK: module {
module {
  func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>, %arg3: tensor<2x3xi32>, %arg4: tensor<2x3xi32>) {
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1)
               threads(%arg8, %arg9, %arg10) in (%arg14 = %c1, %arg15 = %c1, %arg16 = %c1) {
      gpu.terminator
    }
    return
  }
}
