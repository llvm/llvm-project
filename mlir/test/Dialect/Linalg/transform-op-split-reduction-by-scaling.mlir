// RUN: mlir-opt --test-transform-dialect-interpreter %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_split
func.func @matmul_split(%A : tensor<?x256xf32>, %B: tensor<256x32xf32>, %C: tensor<?x32xf32>) -> tensor<?x32xf32> {

  //      CHECK: bufferization.alloc_tensor({{.*}}) : tensor<?x32x64xf32>
  //      CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  // CHECK-SAME: ins(%{{[a-zA-Z0-9]*}}, %{{[a-zA-Z0-9]*}}, %{{[a-zA-Z0-9]*}} : tensor<?x256xf32>, tensor<256x32xf32>, tensor<64x4xi1>)
  // CHECK-SAME: outs(%{{[a-zA-Z0-9]*}} : tensor<?x32x64xf32>) {

  //      CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
  // CHECK-SAME: ins(%{{[a-zA-Z0-9]*}} : tensor<?x32x64xf32>)
  // CHECK-SAME: outs(%{{[a-zA-Z0-9]*}} : tensor<?x32xf32>) {
  %0 = linalg.matmul ins(%A, %B: tensor<?x256xf32>, tensor<256x32xf32>)
                    outs(%C: tensor<?x32xf32>) -> tensor<?x32xf32>
  return %0: tensor<?x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
  %1:4 = transform.structured.split_reduction %0
    { split_factor = 4, insert_split_dimension = 2, use_scaling_algorithm, use_alloc}
}
