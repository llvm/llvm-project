// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=aggressive}))' \
// RUN:   | FileCheck %s

// Tests for SSIType preservation through SCF control-flow canonicalization
// patterns registered via RegionBranchOpInterface:
//
//   - MakeRegionBranchOpSuccessorInputsDead: must NOT fold a live SSI iter_arg
//     to its unique reachable dominating value across control-flow paths.
//   - RemoveDuplicateSuccessorInputUses: must NOT replace one SSI iter_arg with
//     another that has an identical operand signature.
//   - RemoveDeadRegionBranchOpSuccessorInputs: MAY remove a dead SSI iter_arg
//     (deadness is safe; only cross-path collapsing is forbidden).

// -----

// Verify that a live SSI iter_arg is NOT folded to its unique reachable value
// by MakeRegionBranchOpSuccessorInputsDead, even though every predecessor
// (loop entry and back-edge yield) forwards the same dominating value %x.

// CHECK-LABEL: func @scf_for_live_ssi_not_folded
// CHECK: %[[R:.*]] = scf.for {{.*}} iter_args(%[[ARG:.*]] = %{{.*}}) -> (!test.ssi_type)
// CHECK:   "test.use"(%[[ARG]])
// CHECK: "test.use"(%[[R]])
func.func @scf_for_live_ssi_not_folded(%x: !test.ssi_type,
                                        %lb: index, %ub: index, %step: index) {
  %r = scf.for %i = %lb to %ub step %step iter_args(%arg1 = %x)
      -> (!test.ssi_type) {
    "test.use"(%arg1) : (!test.ssi_type) -> ()
    scf.yield %x : !test.ssi_type
  }
  "test.use"(%r) : (!test.ssi_type) -> ()
  return
}

// -----

// Verify that two live SSI iter_args with an identical operand signature are
// NOT deduplicated by RemoveDuplicateSuccessorInputUses. Each represents a
// distinct SSI definition point even though all predecessors forward %x for
// both.

// CHECK-LABEL: func @scf_for_duplicate_ssi_args_not_deduped
// CHECK: scf.for {{.*}} iter_args(%[[A0:.*]] = %{{.*}}, %[[A1:.*]] = %{{.*}})
// CHECK:   "test.use"(%[[A0]])
// CHECK:   "test.use"(%[[A1]])
// CHECK: "test.use"(%{{.*}})
// CHECK: "test.use"(%{{.*}})
func.func @scf_for_duplicate_ssi_args_not_deduped(%x: !test.ssi_type,
                                                    %lb: index, %ub: index,
                                                    %step: index) {
  %r0, %r1 = scf.for %i = %lb to %ub step %step
      iter_args(%arg0 = %x, %arg1 = %x) -> (!test.ssi_type, !test.ssi_type) {
    "test.use"(%arg0) : (!test.ssi_type) -> ()
    "test.use"(%arg1) : (!test.ssi_type) -> ()
    scf.yield %x, %x : !test.ssi_type, !test.ssi_type
  }
  "test.use"(%r0) : (!test.ssi_type) -> ()
  "test.use"(%r1) : (!test.ssi_type) -> ()
  return
}

// -----

// Verify that a dead SSI iter_arg IS removed by
// RemoveDeadRegionBranchOpSuccessorInputs. Deadness is safe to fold; only
// collapsing a live arg across control-flow paths is forbidden.

// CHECK-LABEL: func @scf_for_dead_ssi_iter_arg_removed
// CHECK-NOT: iter_args
func.func @scf_for_dead_ssi_iter_arg_removed(%x: !test.ssi_type,
                                              %lb: index, %ub: index,
                                              %step: index) {
  scf.for %i = %lb to %ub step %step iter_args(%arg1 = %x)
      -> (!test.ssi_type) {
    scf.yield %x : !test.ssi_type
  }
  return
}

// -----

// Verify the same live-SSI preservation for scf.while: neither the
// before-region arg nor the after-region arg should be folded to %x by
// MakeRegionBranchOpSuccessorInputsDead.

// CHECK-LABEL: func @scf_while_live_ssi_not_folded
// CHECK: %[[R:.*]] = scf.while (%[[BA:.*]] = %{{.*}}) : (!test.ssi_type) -> !test.ssi_type
// CHECK:   "test.use"(%[[BA]])
// CHECK:   scf.condition(%{{.*}}) %[[BA]]
// CHECK: ^bb0(%[[AA:.*]]: !test.ssi_type):
// CHECK:   "test.use"(%[[AA]])
// CHECK: return %[[R]]
func.func @scf_while_live_ssi_not_folded(%x: !test.ssi_type,
                                          %cond: i1) -> !test.ssi_type {
  %res = scf.while (%arg0 = %x) : (!test.ssi_type) -> !test.ssi_type {
    "test.use"(%arg0) : (!test.ssi_type) -> ()
    scf.condition(%cond) %arg0 : !test.ssi_type
  } do {
  ^bb0(%arg1: !test.ssi_type):
    "test.use"(%arg1) : (!test.ssi_type) -> ()
    scf.yield %x : !test.ssi_type
  }
  return %res : !test.ssi_type
}
