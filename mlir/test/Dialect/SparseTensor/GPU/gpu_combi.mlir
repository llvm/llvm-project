// RUN: mlir-opt %s --linalg-generalize-named-ops \
// RUN:             --pre-sparsification-rewrite \
// RUN:             --sparsification="parallelization-strategy=dense-outer-loop" \
// RUN:             --sparse-gpu-codegen | FileCheck %s

#CSR = #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>

//
// CHECK-LABEL: gpu.module @sparse_kernels
// CHECK-DAG:   gpu.func @kernel0
// CHECK-DAG:   gpu.func @kernel1
//
// CHECK-LABEL: func.func @matmuls
// CHECK-DAG:   gpu.launch_func @sparse_kernels::@kernel0 blocks
// CHECK-DAG:   gpu.launch_func @sparse_kernels::@kernel1 blocks
//
func.func @matmuls(%A: tensor<1024x8xf64>,
                   %B: tensor<8x1024xf64, #CSR>,
		   %C: tensor<1024x1024xf64, #CSR>) -> tensor<1024x1024xf64> {
  %Z = arith.constant dense<0.0> : tensor<1024x1024xf64>
  %T = linalg.matmul
      ins(%A, %B: tensor<1024x8xf64>, tensor<8x1024xf64, #CSR>)
      outs(%Z: tensor<1024x1024xf64>) -> tensor<1024x1024xf64>
  %D = linalg.matmul
      ins(%T, %C: tensor<1024x1024xf64>, tensor<1024x1024xf64, #CSR>)
      outs(%Z: tensor<1024x1024xf64>) -> tensor<1024x1024xf64>
  return %D : tensor<1024x1024xf64>
}

