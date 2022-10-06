// RUN: mlir-opt -test-transform-dialect-interpreter %s | FileCheck %s

func.func @scalarize(%arg0: tensor<24x12xf32>,
                     %arg1: tensor<12x25xf32>,
                     %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  // The op is first tiled by 10 in the first dimension, which creates a
  // dynamic size, and then scalarized, which brings the dimension to static 1.
  // CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<1x12
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1, %loops = transform.structured.tile %0 [10, 0, 0]
    %2 = transform.structured.scalarize %1
  }
}
