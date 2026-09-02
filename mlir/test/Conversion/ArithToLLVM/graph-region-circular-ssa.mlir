// RUN: mlir-opt --convert-arith-to-llvm %s | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/159675
//
// In a graph region (e.g. the module body), SSA values may be used before
// they are defined.  The arith.addi below uses %1 (a constant zero) as its
// LHS operand, but %1 is defined *after* %0 in the block, creating a
// circular SSA dependency from %0's perspective.
//
// During dialect conversion, OpBuilder::tryFold attempted to fold arith.addi:
//   1. foldCommutative swapped operands so the constant moved to RHS.
//   2. On the next iteration, addi.fold() returned %0 (the op's own result)
//      because the RHS is now a zero constant, signalling an in-place fold.
//   3. foldSingleResultHook called trait folds again; foldCommutative found
//      nothing to swap and returned success with empty fold-results.
//   4. The do-while loop condition was still true, causing an infinite loop.
//
// The fix detects this fixpoint: if a fold returns empty results but leaves
// the op state (operands, attributes, and properties) unchanged, the loop
// stops and tryFold returns failure so that the normal conversion pattern is
// applied instead.

// CHECK: module {
// CHECK: llvm.add
"builtin.module"() ({
  %0 = "arith.addi"(%1, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
  %1 = "arith.constant"() <{value = 0 : index}> : () -> index
}) : () -> ()
