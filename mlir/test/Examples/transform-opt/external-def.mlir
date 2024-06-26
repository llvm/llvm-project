// RUN: mlir-opt %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @external_def(%root: !transform.any_op {transform.readonly}) {
    transform.print %root { name = "external_def" } : !transform.any_op
    transform.yield
  }
}
