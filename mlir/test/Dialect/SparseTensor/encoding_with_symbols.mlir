// RUN: mlir-opt %s -split-input-file -sparsification-and-bufferization -verify-diagnostics | FileCheck %s

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

// -----

// This section makes sure that using the following encoding does not result in
// an assertion error, but instead the expected error. Ultimately, we want to
// make this section pass without any expected errors.

#Sparse = #sparse_tensor.encoding<{
  map = [c](i, j) -> (c * 3 * i : dense, i : dense, j : compressed)
}>

func.func @tensor_convert() -> memref<?xindex> {
  %I = tensor.generate {
  ^bb0(%i: index, %j: index):
    %is_diag = arith.cmpi eq, %i, %j : index
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32
    %val = arith.select %is_diag, %f1, %f0 : f32
    tensor.yield %val : f32
  } : tensor<32x32xf32>

  // expected-error@+1 {{Level size mismatch between source/dest tensors}}
  %J = sparse_tensor.convert %I : tensor<32x32xf32> to tensor<32x32xf32, #Sparse>

  %result = sparse_tensor.positions %J { level = 0 : index }
    : tensor<32x32xf32, #Sparse> to memref<?xindex>

  return %result : memref<?xindex>
}
