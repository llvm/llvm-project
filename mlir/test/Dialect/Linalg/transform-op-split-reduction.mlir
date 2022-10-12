// RUN: mlir-opt --test-transform-dialect-interpreter %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_split
func.func @matmul_split(%A : tensor<16x256xf32>, %B: tensor<256x32xf32>, %C: tensor<16x32xf32>) -> tensor<16x32xf32> {

  //      CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME: ins(%{{[a-zA-Z0-9_]*}}, %{{[a-zA-Z0-9_]*}} : tensor<16x4x64xf32>, tensor<4x64x32xf32>)
  // CHECK-SAME: outs(%{{[a-zA-Z0-9_]*}} : tensor<16x32x4xf32>) {

  //      CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-SAME: ins(%{{[a-zA-Z0-9_]*}} : tensor<16x32x4xf32>)
  // CHECK-SAME: outs(%{{[a-zA-Z0-9_]*}} : tensor<16x32xf32>) {
  %0 = linalg.matmul ins(%A, %B: tensor<16x256xf32>, tensor<256x32xf32>)
                    outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0: tensor<16x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
  %1:4 = transform.structured.split_reduction %0 { split_factor = 4, insert_split_dimension = 2}
}
