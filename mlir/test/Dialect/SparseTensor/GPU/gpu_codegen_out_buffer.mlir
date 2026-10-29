// RUN: mlir-opt %s --linalg-generalize-named-ops \
// RUN:             --pre-sparsification-rewrite \
// RUN:             --sparse-reinterpret-map \
// RUN:             --sparsification="parallelization-strategy=dense-outer-loop" \
// RUN:             --sparse-gpu-codegen | FileCheck %s

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

// CHECK-LABEL: func.func @tensor_add
// CHECK:         %[[TENSOR_EMPTY:.*]] = tensor.empty()
// CHECK:         %[[OUT_BUF:.*]] = bufferization.to_buffer %[[TENSOR_EMPTY]]
// CHECK:         %[[GPU_OUT_BUF:.*]], %[[T0:.*]] = gpu.alloc async [{{.*}}] ()
// CHECK:         gpu.memcpy async [%[[T0]]] %[[GPU_OUT_BUF]], %[[OUT_BUF]]
// CHECK:         %[[T1:.*]] = gpu.launch_func async @sparse_kernels::@kernel0 blocks
// CHECK:         %[[M0:.*]] = gpu.memcpy async [%[[T1]]] %[[OUT_BUF]], %[[GPU_OUT_BUF]]
// CHECK:         gpu.dealloc async [%[[M0]]] %[[GPU_OUT_BUF]]

func.func @tensor_add(%arg0: tensor<32x32xf32, #CSR>,
                      %arg1: tensor<32x32xf32, #CSR>) -> tensor<32x32xf32> {
  %empty = tensor.empty() : tensor<32x32xf32>
  %res = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<32x32xf32, #CSR>, tensor<32x32xf32, #CSR>)
    outs(%empty : tensor<32x32xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %sum = arith.addf %in1, %in2 : f32
    linalg.yield %sum : f32
  } -> tensor<32x32xf32>
  return %res : tensor<32x32xf32>
}
