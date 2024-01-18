// DEFINE: %{entry_point} = main
// DEFINE: %{compile} = mlir-opt %s -convert-arm-sme-to-llvm \
// DEFINE:   -cse -canonicalize -test-lower-to-llvm
// DEFINE: %{run} = %mcr_aarch64_cmd \
// DEFINE:  -march=aarch64 -mattr=+sve,+sme \
// DEFINE:  -e %{entry_point} -entry-point-result=void \
// DEFINE:  -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%mlir_arm_runner_utils

// RUN: %{compile} | %{run} | FileCheck %s

/// Note: This is included in the SME tests rather than the SVE tests as it is
/// safe to assume the SME tests will be ran on an emulator, so will be able to
/// change the vector length.

func.func @checkVScale() {
  %vscale = vector.vscale
  vector.print str "vscale"
  vector.print %vscale : index
  return
}

func.func @main() {
  //      CHECK: vscale
  // CHECK-NEXT: 1
  %c128 = arith.constant 128 : i32
  func.call @setArmVLBits(%c128) : (i32) -> ()
  func.call @checkVScale() : () -> ()

  //      CHECK: vscale
  // CHECK-NEXT: 2
  %c256 = arith.constant 256 : i32
  func.call @setArmVLBits(%c256) : (i32) -> ()
  func.call @checkVScale() : () -> ()

  return
}

func.func private @setArmVLBits(%bits : i32)
