// RUN: mlir-opt %s
// No need to check anything else than parsing here, this is being used by another test as data.

transform.with_pdl_patterns {
^bb0(%arg0: !transform.any_op):
  pdl.pattern @func_return : benefit(1) {
    %0 = pdl.operation "func.return"
    pdl.rewrite %0 with "transform.dialect"
  }

  sequence %arg0 : !transform.any_op failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = pdl_match @func_return in %arg1 : (!transform.any_op) -> !transform.op<"func.return">
    transform.debug.emit_remark_at %0, "matched" : !transform.op<"func.return">
  }
}
