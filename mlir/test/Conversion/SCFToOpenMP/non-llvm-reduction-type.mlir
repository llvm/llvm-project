// RUN: not mlir-opt -convert-scf-to-openmp %s 2>&1 | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/61342
// Verify that -convert-scf-to-openmp does not crash when an scf.parallel
// reduction uses a type that is not an LLVM-compatible type (e.g. index).
// Instead of asserting, the pass must fail gracefully.

// CHECK: unconverted operation found

func.func @no_crash_index_reduction(%A: index, %B: index) -> (index, index) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %0:2 = scf.parallel (%i0, %i1) = (%c1, %c3) to (%c2, %c6) step (%c1, %c3)
      init (%A, %B) -> (index, index) {
    scf.reduce(%i0, %i1 : index, index) {
    ^bb0(%lhs0: index, %rhs0: index):
      %r0 = arith.addi %lhs0, %rhs0 : index
      scf.reduce.return %r0 : index
    }, {
    ^bb0(%lhs1: index, %rhs1: index):
      %r1 = arith.muli %lhs1, %rhs1 : index
      scf.reduce.return %r1 : index
    }
  }
  return %0#0, %0#1 : index, index
}
