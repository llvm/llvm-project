// RUN: mlir-opt %s --linalg-generalize-named-ops \
// RUN:             --pre-sparsification-rewrite \
// RUN:             --sparse-reinterpret-map \
// RUN:             --sparsification="parallelization-strategy=dense-outer-loop" \
// RUN:             --sparse-gpu-codegen | FileCheck %s

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

//
// CHECK-LABEL: gpu.module @sparse_kernels
// CHECK:       gpu.func @kernel1
// CHECK:       gpu.func @kernel0
//
// CHECK-LABEL: func.func @matmuls
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       %[[T1:.*]] = gpu.launch_func async @sparse_kernels::@kernel1 blocks
// CHECK:       gpu.memcpy async [%[[T1]]]
// CHECK:       gpu.dealloc async
// CHECK:       gpu.dealloc async
// CHECK:       gpu.dealloc async
// CHECK:       gpu.dealloc async
// CHECK:       gpu.dealloc async
// CHECK:       gpu.wait
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       %[[T0:.*]] = gpu.launch_func async @sparse_kernels::@kernel0 blocks
// CHECK:       gpu.memcpy async [%[[T0]]]
// CHECK:       gpu.dealloc async
// CHECK:       gpu.dealloc async
// CHECK:       gpu.dealloc async
// CHECK:       gpu.dealloc async
// CHECK:       gpu.dealloc async
// CHECK:       gpu.wait
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
