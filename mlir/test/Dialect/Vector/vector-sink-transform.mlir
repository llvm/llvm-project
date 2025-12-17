// RUN: mlir-opt %s

// This is smoke test for `transform.apply_patterns.vector.sink_ops` and this
// file is also used in `vector-sink.mlir`.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %module_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.sink_ops
      transform.apply_patterns.vector.sink_mem_ops
    } : !transform.any_op
    transform.yield
  }
}
