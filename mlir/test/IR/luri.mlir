// RUN: cd %S && mlir-opt --canonicalize %s | FileCheck %s

func.func @foo() -> (tensor<10xf32>) {
  %0 = arith.constant luri<"luri.raw", byte_offset = 80, byte_size = 80> : tensor<10xf32>
  %1 = arith.constant luri<"luri.raw", byte_offset =  0, byte_size = 80> : tensor<10xf32>
  %c = arith.constant dense<1.0e5> : tensor<10xf32>
  %mul = arith.mulf %0, %c : tensor<10xf32>
  // CHECK: arith.constant{{.*}}785.0708
  %add = arith.addf %1, %mul: tensor<10xf32>
  return %add : tensor<10xf32>
}
