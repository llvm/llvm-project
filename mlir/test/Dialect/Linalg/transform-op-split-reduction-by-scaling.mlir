// RUN: mlir-opt --split-input-file --transform-interpreter %s | FileCheck %s

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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.split_reduction %0
      <split_factor = 4, insert_split_dimension = 2, use_scaling_algorithm, use_alloc>
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// The reduction loop `k` (position 2) is split into a parallel loop at
// position 2 and the remaining reduction loop at position 3, so the parallel
// loop `l` shifts from position 3 to 4. In the body, `linalg.index 2` (`k`)
// must become `p * 4 + r`, `linalg.index 3` (`l`) must become
// `linalg.index 4`, and `linalg.index 0` (`i`) must stay as it is.
func.func @generic_split_index(%A: tensor<2x8xi32>, %B: tensor<8x2x3xi32>, %C: tensor<2x2x3xi32>) -> tensor<2x2x3xi32> {
  %r = linalg.generic {indexing_maps = [affine_map<(i, j, k, l) -> (i, k)>,
                                        affine_map<(i, j, k, l) -> (k, j, l)>,
                                        affine_map<(i, j, k, l) -> (i, j, l)>],
                       iterator_types = ["parallel", "parallel", "reduction", "parallel"]}
    ins(%A, %B : tensor<2x8xi32>, tensor<8x2x3xi32>) outs(%C : tensor<2x2x3xi32>) {
  ^bb0(%a: i32, %b: i32, %acc: i32):
    %i = linalg.index 0 : index
    %k = linalg.index 2 : index
    %l = linalg.index 3 : index
    %ik = arith.addi %i, %k : index
    %ikl = arith.addi %ik, %l : index
    %c = arith.index_cast %ikl : index to i32
    %s = arith.addi %acc, %c : i32
    linalg.yield %s : i32
  } -> tensor<2x2x3xi32>
  return %r : tensor<2x2x3xi32>
}

//  CHECK-DAG: #[[$MAP_K:.*]] = affine_map<()[s0, s1] -> (s0 * 4 + s1)>
// CHECK-LABEL: func.func @generic_split_index
//      CHECK: linalg.generic
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]
//      CHECK: ^bb0
//  CHECK-DAG:   %[[I:.*]] = linalg.index 0 : index
//  CHECK-DAG:   %[[P:.*]] = linalg.index 2 : index
//  CHECK-DAG:   %[[R:.*]] = linalg.index 3 : index
//      CHECK:   %[[K:.*]] = affine.apply #[[$MAP_K]]()[%[[P]], %[[R]]]
//      CHECK:   %[[L:.*]] = linalg.index 4 : index
//      CHECK:   %[[IK:.*]] = arith.addi %[[I]], %[[K]] : index
//      CHECK:   arith.addi %[[IK]], %[[L]] : index
//      CHECK: linalg.generic
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction", "parallel"]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:4 = transform.structured.split_reduction %0
      <split_factor = 4, insert_split_dimension = 2, use_scaling_algorithm>
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}
