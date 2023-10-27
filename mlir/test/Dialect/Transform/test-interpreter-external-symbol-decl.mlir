// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-library-paths=%p/test-interpreter-library/definitions-self-contained.mlir})" \
// RUN:             --verify-diagnostics --split-input-file | FileCheck %s

// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-library-paths=%p/test-interpreter-library/definitions-self-contained.mlir}, test-transform-dialect-interpreter)" \
// RUN:             --verify-diagnostics --split-input-file | FileCheck %s

// The definition of the @print_message named sequence is provided in another
// file. It will be included because of the pass option. Subsequent application
// of the same pass works but only without the library file (since the first
// application loads external symbols and loading them again woul make them
// clash).
// Note that the same diagnostic produced twice at the same location only
// needs to be matched once.

// expected-remark @below {{message}}
// expected-remark @below {{unannotated}}
// expected-remark @below {{internal colliding (without suffix)}}
// expected-remark @below {{internal colliding_0}}
// expected-remark @below {{internal colliding_1}}
// expected-remark @below {{internal colliding_3}}
// expected-remark @below {{internal colliding_4}}
// expected-remark @below {{internal colliding_5}}
module attributes {transform.with_named_sequence} {
  // CHECK-DAG: transform.named_sequence @print_message(
  // CHECK-DAG: transform.include @private_helper
  transform.named_sequence private @print_message(!transform.any_op {transform.readonly})

  // These ops collide with ops from the other module before or after renaming.
  transform.named_sequence private @colliding(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %arg0, "internal colliding (without suffix)" : !transform.any_op
    transform.yield
  }
  transform.named_sequence private @colliding_0(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %arg0, "internal colliding_0" : !transform.any_op
    transform.yield
  }
  transform.named_sequence private @colliding_1(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %arg0, "internal colliding_1" : !transform.any_op
    transform.yield
  }
  transform.named_sequence private @colliding_3(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %arg0, "internal colliding_3" : !transform.any_op
    transform.yield
  }
  // This symbol is public and thus can't be renamed.
  // CHECK-DAG: transform.named_sequence @colliding_4(
  transform.named_sequence @colliding_4(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %arg0, "internal colliding_4" : !transform.any_op
    transform.yield
  }
  transform.named_sequence private @colliding_5(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %arg0, "internal colliding_5" : !transform.any_op
    transform.yield
  }

  // CHECK-DAG: transform.named_sequence @unannotated(
  // CHECK-DAG: test_print_remark_at_operand %{{.*}}, "unannotated"
  transform.named_sequence @unannotated(!transform.any_op {transform.readonly})

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    include @print_message failures(propagate) (%arg0) : (!transform.any_op) -> ()
    include @unannotated failures(propagate) (%arg0) : (!transform.any_op) -> ()
    include @colliding failures(propagate) (%arg0) : (!transform.any_op) -> ()
    include @colliding_0 failures(propagate) (%arg0) : (!transform.any_op) -> ()
    include @colliding_1 failures(propagate) (%arg0) : (!transform.any_op) -> ()
    include @colliding_3 failures(propagate) (%arg0) : (!transform.any_op) -> ()
    include @colliding_4 failures(propagate) (%arg0) : (!transform.any_op) -> ()
    include @colliding_5 failures(propagate) (%arg0) : (!transform.any_op) -> ()
  }
}
