// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize="test-analysis-only allow-return-memref always-aliasing-with-dest=0" -split-input-file | FileCheck %s

// This is a test case for alwaysAliasingWithDest = 0. In that case, an OpResult
// may bufferize in-place with an "in" OpOperand or any non-"out" OpOperand.


#accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>
]
#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

// CHECK-LABEL: func @linalg_op_same_out_tensors(
func @linalg_op_same_out_tensors(
    %t1: tensor<?xf32> {linalg.inplaceable = true},
// CHECK-SAME:          bufferization.access = "read-write"
    %t2: tensor<?xf32> {linalg.inplaceable = true})
// CHECK-SAME:          bufferization.access = "write"
  -> (tensor<?xf32>, tensor<?xf32>){

  // %1 and %2 are not used in the computation, so the two OpResults do not
  // necessarily have to bufferize in-place with the two "out" OpOperands. They
  // bufferize in-place with the first and second OpOperand (one of which is an
  // "in" OpOperand).
  //      CHECK: linalg.generic
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"]
  %o:2 = linalg.generic #trait ins(%t1 : tensor<?xf32>)
                               outs (%t2, %t2 : tensor<?xf32>, tensor<?xf32>) {
      ^bb(%0: f32, %1: f32, %2 : f32) :
        linalg.yield %0, %0 : f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>)

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0, 1]
  return %o#0, %o#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

#accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>
]
#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

// CHECK-LABEL: func @linalg_op_same_out_tensors_2(
func @linalg_op_same_out_tensors_2(
    %t1: tensor<?xf32> {linalg.inplaceable = true},
// CHECK-SAME:          bufferization.access = "read-write"
    %t2: tensor<?xf32> {linalg.inplaceable = true})
// CHECK-SAME:          bufferization.access = "write"
        -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>){

  // %1, %2 and %3 are not used in the computation, so the three OpResults do
  // not necessarily have to bufferize in-place with the three "out" OpOperands.
  // They bufferize in-place with the first, second and third OpOperand (one of
  // which is an "in" OpOperand).
  // In contrast to the previous test case, two of the chosen OpOperands are the
  // same (aliasing) SSA value, which is why one of them must bufferize
  // out-of-place.
  //      CHECK: linalg.generic
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true", "false"]
  %o:3 = linalg.generic #trait
          ins(%t1 : tensor<?xf32>)
          outs (%t2, %t2, %t2 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
      ^bb(%0: f32, %1: f32, %2 : f32, %3 : f32) :
        linalg.yield %0, %0, %0 : f32, f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0, 1, -1]
  return %o#0, %o#1, %o#2 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
}

