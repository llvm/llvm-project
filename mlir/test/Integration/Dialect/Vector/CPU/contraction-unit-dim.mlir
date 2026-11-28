/// Run once without applying the patterns and check the source of truth.
// RUN: mlir-opt %s --test-transform-dialect-erase-schedule -test-lower-to-llvm | \
// RUN: mlir-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

/// Run once with the patterns and compare.
// RUN: mlir-opt %s -transform-interpreter -test-transform-dialect-erase-schedule -test-lower-to-llvm | \
// RUN: mlir-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

// result[m][0][n] = acc[m][0][n] + sum_k lhs[k][m] * rhs[0][k][n]

// The unit dim d0 is in the middle of the acc.
func.func @unit_dim_d0(%lhs: vector<2x3xf32>, %rhs: vector<1x2x4xf32>, %acc: vector<3x1x4xf32>) -> vector<3x1x4xf32> {
  %0 = vector.contract {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d2)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d2, d0, d3)>],
    iterator_types = ["parallel", "reduction", "parallel", "parallel"],
    kind = #vector.kind<add>} %lhs, %rhs, %acc : vector<2x3xf32>, vector<1x2x4xf32> into vector<3x1x4xf32>
  return %0 : vector<3x1x4xf32>
}

func.func @entry() {
  %lhs = arith.constant dense<[[1., 2., 3.], [4., 5., 6.]]> : vector<2x3xf32>
  %rhs = arith.constant dense<[[[1., 2., 3., 4.], [5., 6., 7., 8.]]]> : vector<1x2x4xf32>
  %acc = arith.constant dense<[[[10., 20., 30., 40.]],
                               [[50., 60., 70., 80.]],
                               [[90., 100., 110., 120.]]]> : vector<3x1x4xf32>

  // CHECK: ( ( ( 31, 46, 61, 76 ) ), ( ( 77, 94, 111, 128 ) ), ( ( 123, 142, 161, 180 ) ) )
  %0 = func.call @unit_dim_d0(%lhs, %rhs, %acc) : (vector<2x3xf32>, vector<1x2x4xf32>, vector<3x1x4xf32>) -> vector<3x1x4xf32>
  vector.print %0 : vector<3x1x4xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
        : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    transform.yield
  }
}
