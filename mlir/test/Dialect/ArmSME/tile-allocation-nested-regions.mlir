// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(test-arm-sme-tile-allocation))" -split-input-file -verify-diagnostics

func.func @nested_index_switch_requires_flat_cf(%cond: index) {
  // expected-error @+1 {{ArmSME tile allocation requires flattened control flow; run -convert-scf-to-cf before this pass}}
  scf.index_switch %cond -> vector<[8]x[8]xi16> default {
    %inner = arm_sme.get_tile : vector<[8]x[8]xi16>
    scf.yield %inner : vector<[8]x[8]xi16>
  }
  return
}

// -----

func.func @nested_scf_for_requires_flat_cf(%ub: index) {
  %init = arm_sme.get_tile : vector<[8]x[8]xi16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{ArmSME tile allocation requires flattened control flow; run -convert-scf-to-cf before this pass}}
  scf.for %i = %c0 to %ub step %c1 iter_args(%t = %init)
      -> (vector<[8]x[8]xi16>) {
    %inner = arm_sme.get_tile : vector<[8]x[8]xi16>
    scf.yield %inner : vector<[8]x[8]xi16>
  }
  return
}
