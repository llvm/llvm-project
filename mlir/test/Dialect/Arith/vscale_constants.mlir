// RUN: mlir-opt %s | FileCheck %s

// Note: This test is checking value names (so deliberately is not using a regex match).

func.func @test_vscale_constant_names() {
  %0 = vector.vscale
  %1 = arith.constant 8 : index
  %2 = arith.constant 10 : index
  // CHECK: %c8_vscale = arith.muli %c8, %vscale : index
  %3 = arith.muli %1, %0 : index
  // CHECK: %c10_vscale = arith.muli %vscale, %c10 : index
  %4 = arith.muli %0, %2 : index
  return
}
