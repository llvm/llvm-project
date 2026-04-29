// RUN: mlir-opt %s --canonicalize -split-input-file | FileCheck %s --check-prefixes=CHECK,NOCSE
// RUN: mlir-opt %s --canonicalize='cse-between-iterations=true' -split-input-file | FileCheck %s --check-prefixes=CHECK,CSE
// Convergence / max-iterations interaction: only one pass-application iteration
// is allowed, so CSE unifies the duplicates but the follow-up fold cannot fire.
// RUN: mlir-opt %s --canonicalize='cse-between-iterations=true max-iterations=1' -split-input-file | FileCheck %s --check-prefixes=CHECK,ONESHOT

// Two structurally identical subexpressions cannot be folded away by
// canonicalization alone because they are distinct SSA values. Running CSE
// between iterations unifies them, which lets `arith.subi %a, %a -> 0` fire
// on the next iteration and the whole body collapses to a constant.

// CHECK-LABEL: @dup_subs
func.func @dup_subs(%x: i32, %y: i32) -> i32 {
  // NOCSE-COUNT-3: arith.subi
  // NOCSE-NOT:     arith.subi

  // CSE-NOT: arith.subi
  // CSE:     %[[C0:.*]] = arith.constant 0 : i32
  // CSE:     return %[[C0]]

  // Max-iterations=1: CSE fires once but the downstream subi(a, a) -> 0 fold
  // needs a second pattern-application iteration, which is disallowed.
  // ONESHOT-COUNT-2: arith.subi
  // ONESHOT-NOT:     arith.constant
  %a = arith.subi %x, %y : i32
  %b = arith.subi %x, %y : i32
  %c = arith.subi %a, %b : i32
  return %c : i32
}

// -----

// After CSE unifies the two redundant subi ops, the downstream `arith.subi
// %a, %a` folds to 0, which in turn makes the downstream `arith.addi 0, %y`
// fold to %y. This demonstrates that CSE-between-iterations enables a
// cascading simplification that canonicalization alone cannot achieve.

// CHECK-LABEL: @cascade
func.func @cascade(%x: i32, %y: i32) -> i32 {
  // NOCSE-COUNT-3: arith.subi
  // NOCSE:         arith.addi
  // NOCSE:         return

  // CSE-NOT: arith.subi
  // CSE-NOT: arith.addi
  // CSE:     return %arg1 : i32
  %a = arith.subi %x, %y : i32
  %b = arith.subi %x, %y : i32
  %c = arith.subi %a, %b : i32
  %d = arith.addi %c, %y : i32
  return %d : i32
}

// -----

// Nested regions must also be reached by CSE-between-iterations. The
// duplicate `arith.subi` ops inside the scf.for body are unified, unblocking
// the `arith.subi %a, %a -> 0` fold on the next iteration and then the
// `arith.addi 0, ...` fold that follows. The loop body still uses `%i` so
// the loop itself is not dead and survives canonicalization.

// CHECK-LABEL: @nested
func.func @nested(%lb: index, %ub: index, %step: index,
                  %x: i32, %y: i32, %init: i32) -> i32 {
  // NOCSE:         scf.for
  // NOCSE-COUNT-3: arith.subi

  // CSE:     scf.for
  // CSE-NOT: arith.subi
  // CSE:     scf.yield
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> i32 {
    %a = arith.subi %x, %y : i32
    %b = arith.subi %x, %y : i32
    %c = arith.subi %a, %b : i32
    %ic = arith.index_cast %i : index to i32
    %nxt = arith.addi %acc, %ic : i32
    %final = arith.addi %nxt, %c : i32
    scf.yield %final : i32
  }
  return %r : i32
}
