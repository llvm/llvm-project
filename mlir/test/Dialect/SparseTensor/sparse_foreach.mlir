// RUN: mlir-opt %s --post-sparsification-rewrite="enable-runtime-library=false enable-foreach=true" | FileCheck %s

// CHECK-LABEL: func.func @sparse_foreach_constant
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[V1:.*]] = arith.constant 5.000000e+00 : f32
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[V3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:   %[[V4:.*]] = arith.constant 6.000000e+00 : f32
//               (1, 1) -> (2, 1) -> (2, 2)
// CHECK-NEXT:  "test.use"(%[[C1]], %[[C1]], %[[V1]])
// CHECK-NEXT:  "test.use"(%[[C2]], %[[C1]], %[[V3]])
// CHECK-NEXT:  "test.use"(%[[C1]], %[[C2]], %[[V4]])
//               (1, 1) -> (1, 2) -> (2, 1)
// CHECK-NEXT:  "test.use"(%[[C1]], %[[C1]], %[[V1]])
// CHECK-NEXT:  "test.use"(%[[C1]], %[[C2]], %[[V4]])
// CHECK-NEXT:  "test.use"(%[[C2]], %[[C1]], %[[V3]])
func.func @sparse_foreach_constant() -> () {
  %cst = arith.constant sparse<[[2, 1], [1, 1], [1, 2]], [1.0, 5.0, 6.0]> : tensor<8x7xf32>
  // Make use the sparse constant are properly sorted based on the requested order.
  sparse_tensor.foreach in %cst { order = affine_map<(d0, d1) -> (d1, d0)> } : tensor<8x7xf32> do {
  ^bb0(%arg0: index, %arg1: index, %arg2: f32):
    "test.use" (%arg0, %arg1, %arg2): (index,index,f32)->()
  }
  sparse_tensor.foreach in %cst : tensor<8x7xf32> do {
  ^bb0(%arg0: index, %arg1: index, %arg2: f32):
    "test.use" (%arg0, %arg1, %arg2): (index,index,f32)->()
  }
  return
}
