// RUN: mlir-opt -test-transform-dialect-interpreter %s | FileCheck %s

func.func @scalarize(%arg0: tensor<24x12xf32>,
                     %arg1: tensor<12x25xf32>,
                     %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  // The op is first tiled by 10 in the first dimension, which creates a
  // dynamic size, and then scalarized, which brings the dimension to static 1.
  // CHECK: %[[RES_LOOP_1:.*]] = scf.for {{.*}} -> (tensor<24x25xf32>)
  // CHECK:   %[[RES_LOOP_2:.*]] = scf.for {{.*}} -> (tensor<?x25xf32>)
  // CHECK:     %[[MM:.*]] = linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<1x12
  // CHECK:     %[[INS_2:.*]] = tensor.insert_slice %[[MM]] into %{{.*}} [1, 25] [1, 1] : tensor<1x25xf32> into tensor<?x25xf32>
  // CHECK:     scf.yield %[[INS_2]] : tensor<?x25xf32>
  // CHECK:   %[[INS_1:.*]] = tensor.insert_slice %[[RES_LOOP_2]] into %{{.*}}, 25] [1, 1] : tensor<?x25xf32> into tensor<24x25xf32>
  // CHECK:   scf.yield %[[INS_1]] : tensor<24x25xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>

  // CHECK: return %[[RES_LOOP_1]] : tensor<24x25xf32>
  func.return %0 : tensor<24x25xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1, %loops = transform.structured.tile_using_for %0 [10, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %2 = transform.structured.scalarize %1 : (!transform.any_op) -> !transform.any_op
}
