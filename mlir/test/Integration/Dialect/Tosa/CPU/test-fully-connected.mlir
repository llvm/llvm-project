// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith))" | \
// RUN: mlir-opt -one-shot-bufferize -func-bufferize -test-lower-to-llvm | \
// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN: | FileCheck %s

func.func private @printMemrefF32(tensor<*xf32>)

func.func @main() {
  %A = arith.constant dense<[
    [8.0, 1.0, 6.0],
    [3.0, 5.0, 7.0],
    [4.0, 9.0, 2.0]
  ]> : tensor<3x3xf32>

  %B = arith.constant dense<[
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0]
  ]> : tensor<3x3xf32>

  %C = arith.constant dense<[0.0, 1.0, 2.0]> : tensor<3xf32>

  %result = tosa.fully_connected %A, %B, %C : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<3xf32>) -> tensor<3x3xf32>

  %result_unranked = tensor.cast %result : tensor<3x3xf32> to tensor<*xf32>
  call @printMemrefF32(%result_unranked) : (tensor<*xf32>) -> ()
  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [3, 3] strides = [3, 1] data =
// CHECK-NEXT:      [
// CHECK-SAME:  [15, 16, 17]
// CHECK-NEXT:  [15, 16, 17]
// CHECK-NEXT:  [15, 16, 17]
// CHECK-SAME: ]
