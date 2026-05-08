// RUN: mlir-opt %s -sparsification-and-bufferization | FileCheck %s

// Tests that mlir-opt does not crash when parsing sparse tensor encodings with symbols.

// CHECK-DAG: #[[$SPARSE_0:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : dense, d1 : dense, d2 : compressed) }>
// CHECK-DAG: #[[$SPARSE_1:.*]] = #sparse_tensor.encoding<{ map = [s0](d0, d1) -> (d0 * (s0 * 3) : dense, d0 : dense, d1 : compressed) }>

#Sparse = #sparse_tensor.encoding<{
  map = [c](i, j) -> (c * 3 * i : dense, i : dense, j : compressed)
}>

// CHECK-LABEL: func.func @tensor_add(
// CHECK-SAME:      %{{.*}}: memref<?xindex>, %{{.*}}: memref<?xindex>, %{{.*}}: memref<?xf32>,
// CHECK-SAME:      %{{.*}}: !sparse_tensor.storage_specifier<#[[$SPARSE_0]]>) -> memref<8x8xf32> {
func.func @tensor_add(%arg0: tensor<8x8xf32, #Sparse>) -> tensor<8x8xf32> {
  %result_out = tensor.empty() : tensor<8x8xf32>

  // CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x8xf32>
  // CHECK: %[[RES:.*]] = linalg.add ins(%{{.*}}, %{{.*}} : tensor<8x8xf32, #[[$SPARSE_1]]>, tensor<8x8xf32, #[[$SPARSE_1]]>)
  %result = linalg.add
    ins(%arg0, %arg0 : tensor<8x8xf32, #Sparse>, tensor<8x8xf32, #Sparse>)
    outs(%result_out : tensor<8x8xf32>) -> tensor<8x8xf32>

  // CHECK: return %{{.*}} : memref<8x8xf32>
  return %result : tensor<8x8xf32>
}
