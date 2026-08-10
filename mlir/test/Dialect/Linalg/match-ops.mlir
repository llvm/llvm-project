// RUN: mlir-opt %s | mlir-opt | FileCheck %s

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
    // Checking the syntax of positional specifications.
    // CHECK: dim %{{.*}}[all]
    transform.match.structured.dim %arg1[all] : !transform.any_op
    // CHECK: dim %{{.*}}[0]
    transform.match.structured.dim %arg1[0] : !transform.any_op
    // CHECK: dim %{{.*}}[0, 1, -2]
    transform.match.structured.dim %arg1[0, 1, -2] : !transform.any_op
    // CHECK: dim %{{.*}}[except(0)]
    transform.match.structured.dim %arg1[except(0)] : !transform.any_op
    // CHECK: dim %{{.*}}[except(0, -1, 2)]
    transform.match.structured.dim %arg1[except(0, -1, 2)] : !transform.any_op

    transform.match.structured.yield
  }

  // Checking the syntax of trailing types.
  // CHECK: structured %{{.*}} : !transform.any_op
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
    transform.match.structured.yield
  }
  // CHECK: structured %{{.*}} : (!transform.any_op) -> !transform.any_op
  transform.match.structured %arg0 : (!transform.any_op) -> !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
    transform.match.structured.yield %arg1 : !transform.any_op
  }
  // CHECK: structured %{{.*}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.match.structured %arg0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op) {
  ^bb1(%arg1: !transform.any_op):
    transform.match.structured.yield %arg1, %arg1 : !transform.any_op, !transform.any_op
  }

  transform.yield
}
