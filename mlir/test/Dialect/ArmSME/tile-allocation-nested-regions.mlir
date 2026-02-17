// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(test-arm-sme-tile-allocation))" -split-input-file -verify-diagnostics

module {
  func.func @arm_sme_tile_load_hor_f16(%arg0: memref<?x?xf16>) {
    %c0 = arith.constant 0 : index
    %0 = arm_sme.get_tile : vector<[8]x[8]xf16>
    %c8 = arith.constant 8 : index
    %vscale = vector.vscale
    %c8_vscale = arith.muli %c8, %vscale : index
    %cst = arith.constant dense<true> : vector<[8]xi1>
    %c0_0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    // expected-error @+1 {{ArmSME tile allocation requires flattened control flow; run -convert-scf-to-cf before this pass (e.g. via convert-arm-sme-to-llvm pipeline)}}
    %1 = scf.for %arg1 = %c0_0 to %c64 step %c8 iter_args(%arg2 = %0) -> (vector<[8]x[8]xf16>) {
      %2 = arith.addi %arg1, %c8 : index
      %3 = arm_sme.load_tile_slice %arg0[%2, %arg1], %cst, %arg2, %arg1 : memref<?x?xf16>, vector<[8]xi1>, vector<[8]x[8]xf16>
      scf.yield %3 : vector<[8]x[8]xf16>
    }
    return
  }
}

// -----

module {
  func.func @main() {
    %0 = index.constant 0
    %1 = arm_sme.get_tile : vector<[8]x[8]xi16>
    // expected-error @+1 {{ArmSME tile allocation requires flattened control flow; run -convert-scf-to-cf before this pass (e.g. via convert-arm-sme-to-llvm pipeline)}}
    %2 = scf.index_switch %0 -> vector<[8]x[8]xi16> default {
      %3 = arm_sme.get_tile : vector<[8]x[8]xi16>
      scf.yield %3 : vector<[8]x[8]xi16>
    }
    return
  }
}
