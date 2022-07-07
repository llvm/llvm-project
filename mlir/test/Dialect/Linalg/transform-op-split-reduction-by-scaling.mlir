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

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_target : benefit(1) {
    %args = operands
    %results = types
    %0 = pdl.operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @pdl_target in %arg1
    %1:4 = transform.structured.split_reduction %0 
      { split_factor = 4, insert_split_dimension = 2, use_scaling_algorithm, use_alloc}
  }
}
