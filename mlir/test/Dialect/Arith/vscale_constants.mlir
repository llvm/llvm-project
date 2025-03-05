// RUN: mlir-opt %s | FileCheck %s

// Note: This test is checking value names (so deliberately is not using a regex match).

func.func @test_vscale_constant_names() {
  %vscale = vector.vscale
  %c8 = arith.constant 8 : index
  // CHECK: %c8_vscale = arith.muli
  %0 = arith.muli %vscale, %c8 : index
  %c10 = arith.constant 10 : index
  // CHECK: %c10_vscale = arith.muli
  %1 = arith.muli %c10, %vscale : index
  return
}
