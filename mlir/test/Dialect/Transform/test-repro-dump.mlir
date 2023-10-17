// REQUIRES: asserts
// RUN: mlir-opt %s --test-transform-dialect-interpreter \
// RUN:             --mlir-disable-threading \
// RUN:             --debug-only=transform-dialect-dump-repro 2>&1 \
// RUN: | FileCheck %s

module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.test_print_remark_at_operand %arg0, "remark" : !transform.any_op
  }
}

// Verify that the repro string is dumped.

// CHECK: Transform Interpreter Repro
// CHECK: cat <<EOF | mlir-opt --pass-pipeline="builtin.module(test-transform-dialect-interpreter{debug-payload-root-tag=payload_root debug-transform-root-tag=transform_container})"

// Verify that the IR is dumped with tags.

// CHECK: module
// CHECK-SAME: transform.target_tag = "payload_root"
// CHECK: transform.sequence
// CHECK-SAME: transform.target_tag = "transform_container"
// CHECK: EOF

// Verify that the actual IR after the pass doesn't have the tags.

// CHECK: module
// CHECK-NOT: transform.target_tag = "payload_root"
// CHECK: transform.sequence
// CHECK-NOT: transform.target_tag = "transform_container"
