// DEFINE: %{entry_point} = main
// DEFINE: %{compile} = mlir-opt %s -test-lower-to-llvm
// DEFINE: %{run} = %mcr_aarch64_cmd -march=aarch64 -mattr=+sve \
// DEFINE:  -e %{entry_point} -entry-point-result=void \
// DEFINE:  -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%mlir_arm_runner_utils

// RUN: %{compile} | %{run} | FileCheck %s

func.func @checkVScale() {
  %vscale = vector.vscale
  vector.print str "vscale = "
  vector.print %vscale : index
  return
}

func.func @setAndCheckVL(%bits: i32) {
  func.call @setArmVLBits(%bits) : (i32) -> ()
  func.call @checkVScale() : () -> ()
  return
}

func.func @main() {
  //      CHECK: vscale = 1
  %c128 = arith.constant 128 : i32
  func.call @setAndCheckVL(%c128) : (i32) -> ()

  //      CHECK: vscale = 2
  %c256 = arith.constant 256 : i32
  func.call @setAndCheckVL(%c256) : (i32) -> ()

  //      CHECK: vscale = 4
  %c512 = arith.constant 512 : i32
  func.call @setAndCheckVL(%c512) : (i32) -> ()

  //      CHECK: vscale = 8
  %c1024 = arith.constant 1024 : i32
  func.call @setAndCheckVL(%c1024) : (i32) -> ()

  //      CHECK: vscale = 16
  %c2048 = arith.constant 2048 : i32
  func.call @setAndCheckVL(%c2048) : (i32) -> ()

  return
}

func.func private @setArmVLBits(%bits : i32)
