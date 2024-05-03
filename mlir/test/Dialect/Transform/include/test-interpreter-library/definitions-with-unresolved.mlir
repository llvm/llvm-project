// RUN: mlir-opt %s
// No need to check anything else than parsing here, this is being used by another test as data.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @print_message(%arg0: !transform.any_op {transform.readonly})

  transform.named_sequence @reference_other_module(%arg0: !transform.any_op) {
    transform.include @print_message failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}
