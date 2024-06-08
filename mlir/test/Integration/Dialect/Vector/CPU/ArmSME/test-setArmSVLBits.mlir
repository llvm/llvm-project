// REQUIRES: arm-emulator

// DEFINE: %{entry_point} = main
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE: --pass-pipeline="builtin.module(func.func(convert-arm-sme-to-llvm),test-lower-to-llvm)"
// DEFINE: %{run} = %mcr_aarch64_cmd \
// DEFINE:  -march=aarch64 -mattr=+sve,+sme \
// DEFINE:  -e %{entry_point} -entry-point-result=void \
// DEFINE:  -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%mlir_arm_runner_utils

// RUN: %{compile} | %{run} | FileCheck %s

func.func @checkSVL() {
  %svl_b = arm_sme.streaming_vl <byte>
  %svl_h = arm_sme.streaming_vl <half>
  %svl_w = arm_sme.streaming_vl <word>
  %svl_d = arm_sme.streaming_vl <double>
  vector.print str "SVL.b\n"
  vector.print %svl_b : index
  vector.print str "SVL.h\n"
  vector.print %svl_h : index
  vector.print str "SVL.w\n"
  vector.print %svl_w : index
  vector.print str "SVL.d\n"
  vector.print %svl_d : index
  return
}

func.func @setAndCheckSVL(%bits: i32) {
  func.call @setArmSVLBits(%bits) : (i32) -> ()
  func.call @checkSVL() : () -> ()
  return
}

func.func @main() {
  //      CHECK: SVL.b
  // CHECK-NEXT: 16
  //
  // CHECK-NEXT: SVL.h
  // CHECK-NEXT: 8
  //
  // CHECK-NEXT: SVL.w
  // CHECK-NEXT: 4
  //
  // CHECK-NEXT: SVL.d
  // CHECK-NEXT: 2
  %c128 = arith.constant 128 : i32
  func.call @setAndCheckSVL(%c128) : (i32) -> ()

  //      CHECK: SVL.b
  // CHECK-NEXT: 32
  //
  // CHECK-NEXT: SVL.h
  // CHECK-NEXT: 16
  //
  // CHECK-NEXT: SVL.w
  // CHECK-NEXT: 8
  //
  // CHECK-NEXT: SVL.d
  // CHECK-NEXT: 4
  %c256 = arith.constant 256 : i32
  func.call @setAndCheckSVL(%c256) : (i32) -> ()

  //      CHECK: SVL.b
  // CHECK-NEXT: 64
  //
  // CHECK-NEXT: SVL.h
  // CHECK-NEXT: 32
  //
  // CHECK-NEXT: SVL.w
  // CHECK-NEXT: 16
  //
  // CHECK-NEXT: SVL.d
  // CHECK-NEXT: 8
  %c512 = arith.constant 512 : i32
  func.call @setAndCheckSVL(%c512) : (i32) -> ()

  return
}

func.func private @setArmSVLBits(%bits : i32)
