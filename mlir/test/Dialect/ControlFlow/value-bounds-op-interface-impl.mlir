// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-affine-reify-value-bounds))' \
// RUN:     -verify-diagnostics -split-input-file

// Note: unstructured control flow (cf dialect) is not yet supported by the
// ValueBoundsOpInterface. Block arguments from non-entry blocks cannot have
// their bounds computed. The tests below verify that the infrastructure does
// not crash on such inputs and fails gracefully instead.
// See: https://github.com/llvm/llvm-project/issues/119861

// Regression test: ValueBoundsConstraintSet must not crash when asked to
// reify a bound for a non-entry block argument produced by unstructured
// control flow.

func.func @no_crash_non_entry_block_arg(%n: index) -> index {
  %c0 = arith.constant 0 : index
  cf.br ^bb1(%c0 : index)
^bb1(%i: index):
  // expected-error@+1 {{'test.reify_bound' op could not reify bound}}
  %bound = "test.reify_bound"(%i) {type = "UB"} : (index) -> index
  "test.some_use"(%bound) : (index) -> ()
  %cond = arith.cmpi slt, %i, %n : index
  %next = arith.addi %i, %c0 : index
  cf.cond_br %cond, ^bb1(%next : index), ^bb2
^bb2:
  return %i : index
}
