// IntegerRangeAnalysis convergence on scf.while with dynamic bounds.
//
// The carry range ratchets [0,0]->[0,1]->[0,2]->... per worklist visit;
// nested scf.if layers with arith chains (addi, muli) bounded by remui
// create enough worklist cascade to defeat the solver's back-to-back
// convergence shortcut. The per-state widening budget on
// IntegerValueRangeLattice forces the range to its max after a bounded
// number of strict refinements, so the analysis terminates instead of
// hanging for ~minutes (or 2^31 iterations).
//
// We assert:
//   - the analysis terminates and produces well-formed IR;
//   - the loop-carried iter arg of the outer scf.while widens to
//     [INT_MIN, INT_MAX] (the only sound result once the budget fires);
//   - transfer-function results inside the body stay tight (e.g.
//     `arith.remui ..., %c127` = [0, 126]), verifying the widening is
//     scoped to framework merge sites, not transfer-function joins.
//
// RUN: mlir-opt -int-range-optimizations %s | FileCheck %s

// CHECK-LABEL: func.func @grouped_gemm_while_hang
// CHECK-SAME: (%[[N:.*]]: i32, %{{.*}}: i1) -> i32
func.func @grouped_gemm_while_hang(%n: i32, %flag: i1) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c7 = arith.constant 7 : i32
  %c127 = arith.constant 127 : i32
  %init = arith.cmpi slt, %c0, %n : i32

  // CHECK: %[[OUTER:.*]]:2 = scf.while
  %res:2 = scf.while (%a0 = %c0, %cond = %init) : (i32, i1) -> (i32, i1) {
    scf.condition(%cond) %a0, %cond : i32, i1
  } do {
  ^bb0(%b0: i32, %bc: i1):
    %t0 = arith.addi %b0, %c1 : i32
    %ic = arith.cmpi slt, %t0, %n : i32

    // CHECK: scf.while
    %inner:2 = scf.while (%i0 = %t0, %iic = %ic) : (i32, i1) -> (i32, i1) {
      scf.condition(%iic) %i0, %iic : i32, i1
    } do {
    ^bb1(%j0: i32, %jc: i1):

      %L0 = scf.if %flag -> (i32) {
        %a0_0 = arith.addi %j0, %c1 : i32
        %a0_1 = arith.muli %a0_0, %c7 : i32
        %a0_r = arith.remui %a0_1, %c127 : i32
        scf.yield %a0_r : i32
      } else {
        %b0_0 = arith.addi %j0, %c3 : i32
        %b0_1 = arith.muli %b0_0, %c7 : i32
        %b0_r = arith.remui %b0_1, %c127 : i32
        scf.yield %b0_r : i32
      }

      %L1 = scf.if %flag -> (i32) {
        %a1_0 = arith.addi %L0, %c1 : i32
        %a1_1 = arith.muli %a1_0, %c7 : i32
        %a1_r = arith.remui %a1_1, %c127 : i32
        scf.yield %a1_r : i32
      } else {
        %b1_0 = arith.addi %L0, %c3 : i32
        %b1_1 = arith.muli %b1_0, %c7 : i32
        %b1_r = arith.remui %b1_1, %c127 : i32
        scf.yield %b1_r : i32
      }

      %nic = arith.cmpi slt, %L1, %n : i32
      // The yielded `arith.remui` result stays at [0, 126]: the widening
      // budget only fires on virtual `Lattice::join` at framework merge
      // sites, not on transfer-function joins for inferrable ops.
      // CHECK: test.reflect_bounds {smax = 126 : si32, smin = 0 : si32, umax = 126 : ui32, umin = 0 : ui32}
      %r_l1 = test.reflect_bounds %L1 : i32
      scf.yield %L1, %nic : i32, i1
    }

    %nc = arith.cmpi slt, %inner#0, %n : i32
    scf.yield %inner#0, %nc : i32, i1
  }
  // The outer loop-carried iter arg goes through region-successor merges
  // and is widened to maxRange after the budget is exhausted. The mere
  // presence of these bounds here is the convergence assertion: without
  // the patch the analysis would not terminate to print this attribute.
  // CHECK: %[[BOUNDED:.*]] = test.reflect_bounds
  // CHECK-SAME: {smax = 2147483647 : si32, smin = -2147483648 : si32, umax = 4294967295 : ui32, umin = 0 : ui32}
  // CHECK-SAME: %[[OUTER]]#0 : i32
  %r = test.reflect_bounds %res#0 : i32
  // CHECK: return %[[BOUNDED]] : i32
  return %r : i32
}
