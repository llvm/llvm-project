// RUN: mlir-opt %s --test-transform-dialect-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func @lower_permutation_with_mask(
//       CHECK:   %[[vec:.*]] = arith.constant dense<-2.000000e+00> : vector<7x1xf32>
//       CHECK:   %[[mask:.*]] = arith.constant dense<[true, false, true, false, true, true, true]> : vector<7xi1>
//       CHECK:   %[[b:.*]] = vector.broadcast %[[mask]] : vector<7xi1> to vector<1x7xi1>
//       CHECK:   %[[tp:.*]] = vector.transpose %[[b]], [1, 0] : vector<1x7xi1> to vector<7x1xi1>
//       CHECK:   vector.transfer_write %[[vec]], %{{.*}}[%{{.*}}, %{{.*}}], %[[tp]] {in_bounds = [false, true]} : vector<7x1xf32>, memref<?x?xf32>
func.func @lower_permutation_with_mask(%A : memref<?x?xf32>, %base1 : index,
                                       %base2 : index) {
  %fn1 = arith.constant -2.0 : f32
  %vf0 = vector.splat %fn1 : vector<7xf32>
  %mask = arith.constant dense<[1, 0, 1, 0, 1, 1, 1]> : vector<7xi1>
  vector.transfer_write %vf0, %A[%base1, %base2], %mask
    {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [false]}
    : vector<7xf32>, memref<?x?xf32>
  return
}

transform.sequence failures(propagate) {
^bb1(%module_op: !transform.any_op):
  %f = transform.structured.match ops{["func.func"]} in %module_op
    : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %f {
    transform.apply_patterns.vector.transfer_permutation_patterns
  } : !transform.any_op
}
