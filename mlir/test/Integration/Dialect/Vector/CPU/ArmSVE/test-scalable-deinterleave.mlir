// REQUIRES: arm-emulator

// DEFINE: %{entry_point} = entry
// DEFINE: %{compile} = mlir-opt %s -test-lower-to-llvm
// DEFINE: %{run} = %mcr_aarch64_cmd -march=aarch64 -mattr=+sve \
// DEFINE:  -e %{entry_point} -entry-point-result=void \
// DEFINE:  -shared-libs=%mlir_c_runner_utils,%mlir_arm_runner_utils

// RUN: %{compile} | %{run} | FileCheck %s

func.func @entry() {
  // Set the vector length to 256-bit (equivalent to vscale=2).
  // This allows the checks (below) to look at an entire vector.
  %c256 = arith.constant 256 : i32
  func.call @setArmVLBits(%c256) : (i32) -> ()
  func.call @test_deinterleave() : () -> ()
  return
}

func.func @test_deinterleave() {
  %step_vector = llvm.intr.experimental.stepvector : vector<[4]xi8>
  vector.print %step_vector : vector<[4]xi8>
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7 )
  %v1, %v2 = vector.deinterleave %step_vector : vector<[4]xi8> -> vector<[2]xi8>
  vector.print %v1 : vector<[2]xi8>
  vector.print %v2 : vector<[2]xi8>
  // CHECK: ( 0, 2, 4, 6 )
  // CHECK: ( 1, 3, 5, 7 )
  return
}

func.func private @setArmVLBits(%bits : i32)
