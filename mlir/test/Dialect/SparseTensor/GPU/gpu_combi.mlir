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
// CHECK-LABEL: func.func @matmuls(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<1024x8xf64>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<8x1024xf64, #sparse>,
// CHECK-SAME:    %[[ARG2:.*]]: tensor<1024x1024xf64, #sparse>)
// CHECK-SAME:    -> tensor<1024x1024xf64> {
// CHECK:       %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<1024x1024xf64>
// CHECK:       %[[OUT_BUF0:.*]] = bufferization.to_buffer %[[ZERO]]
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       %[[GPU_OUT_BUF0:.*]], %[[T0:.*]] = gpu.alloc async
// CHECK:       gpu.memcpy async [%[[T0]]] %[[GPU_OUT_BUF0]], %[[OUT_BUF0]]
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       %[[T1:.*]] = gpu.launch_func async @sparse_kernels::@kernel1 blocks
// CHECK:       gpu.dealloc async
// CHECK:       gpu.dealloc async
// CHECK:       gpu.dealloc async
// CHECK:       %[[T2:.*]] = gpu.memcpy async [%[[T1]]] %[[OUT_BUF0]], %[[GPU_OUT_BUF0]]
// CHECK:       gpu.dealloc async [%[[T2]]] %[[GPU_OUT_BUF0]]
// CHECK:       gpu.dealloc async
// CHECK:       gpu.wait
// CHECK:       %[[OUT_BUF1:.*]] = bufferization.to_buffer %[[ZERO]]
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       %[[GPU_OUT_BUF1:.*]], %[[T4:.*]] = gpu.alloc async
// CHECK:       gpu.memcpy async [%[[T4]]] %[[GPU_OUT_BUF1]], %[[OUT_BUF1]]
// CHECK:       gpu.alloc async
// CHECK:       gpu.memcpy async
// CHECK:       %[[T0:.*]] = gpu.launch_func async @sparse_kernels::@kernel0 blocks
// CHECK:       gpu.memcpy async [%[[T0]]]
// CHECK:       gpu.dealloc async
// CHECK:       gpu.wait async
// CHECK:       gpu.dealloc async
// CHECK:       gpu.wait async
// CHECK:       gpu.dealloc async
// CHECK:       %[[T5:.*]] = gpu.memcpy async [%[[T0]]] %[[OUT_BUF1]], %[[GPU_OUT_BUF1]]
// CHECK:       gpu.dealloc async [%[[T5]]] %[[GPU_OUT_BUF1]]
// CHECK:       gpu.wait async
// CHECK:       gpu.dealloc async
// CHECK:       gpu.wait
// CHECK:       %[[OUT_TENSOR:.*]] = bufferization.to_tensor %[[OUT_BUF1]]
// CHECK:       return %[[OUT_TENSOR]]
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
