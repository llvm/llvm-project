// IntegerRangeAnalysis non-convergence on scf.while with dynamic bounds.
//
// The carry range ratchets [0,0]->[0,1]->[0,2]->... without bound.
// Two nested scf.if layers with differing arith chains (addi, muli)
// bounded by remui create enough worklist cascade to prevent the
// solver's back-to-back convergence shortcut from firing.
//
// After the fix (visit cap in visitRegionSuccessors), the analysis
// converges in bounded time.
//
// RUN: mlir-opt -int-range-optimizations %s -o /dev/null

func.func @grouped_gemm_while_hang(%n: i32, %flag: i1) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c7 = arith.constant 7 : i32
  %c127 = arith.constant 127 : i32
  %init = arith.cmpi slt, %c0, %n : i32

  %res:2 = scf.while (%a0 = %c0, %cond = %init) : (i32, i1) -> (i32, i1) {
    scf.condition(%cond) %a0, %cond : i32, i1
  } do {
  ^bb0(%b0: i32, %bc: i1):
    %t0 = arith.addi %b0, %c1 : i32
    %ic = arith.cmpi slt, %t0, %n : i32

    %inner:2 = scf.while (%i0 = %t0, %iic = %ic) : (i32, i1) -> (i32, i1) {
      scf.condition(%iic) %i0, %iic : i32, i1
    } do {
    ^bb1(%j0: i32, %jc: i1):

      // Layer 0: branches must differ to prevent folding.
      // remui bounds ranges to [0,126], preventing overflow-cascade
      // convergence.  Both branches must have ops (not just passthrough)
      // to generate enough worklist items.
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

      // Layer 1: second nested scf.if feeds from layer 0.
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
      scf.yield %L1, %nic : i32, i1
    }

    %nc = arith.cmpi slt, %inner#0, %n : i32
    scf.yield %inner#0, %nc : i32, i1
  }
  return %res#0 : i32
}
