// Check that the wide integer multiplication emulation produces the same result as wide
// multiplication. Emulate i16 ops with i8 ops.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

// RUN: mlir-opt %s --test-arith-emulate-wide-int="widest-int-supported=8" \
// RUN:             --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

// Ops in this function *only* will be emulated using i8 types.
func.func @emulate_muli(%lhs : i16, %rhs : i16) -> (i16) {
  %res = arith.muli %lhs, %rhs : i16
  return %res : i16
}

func.func @check_muli(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_muli(%lhs, %rhs) : (i16, i16) -> (i16)
  vector.print %res : i16
  return
}

func.func @entry() {
  %cst0 = arith.constant 0 : i16
  %cst1 = arith.constant 1 : i16
  %cst_n1 = arith.constant -1 : i16
  %cst_n3 = arith.constant -3 : i16

  %cst13 = arith.constant 13 : i16
  %cst37 = arith.constant 37 : i16
  %cst42 = arith.constant 42 : i16

  %cst256 = arith.constant 256 : i16
  %cst_i16_max = arith.constant 32767 : i16
  %cst_i16_min = arith.constant -32768 : i16

  // CHECK: 0
  func.call @check_muli(%cst0, %cst0) : (i16, i16) -> ()
  // CHECK-NEXT: 0
  func.call @check_muli(%cst0, %cst1) : (i16, i16) -> ()
  // CHECK-NEXT: 1
  func.call @check_muli(%cst1, %cst1) : (i16, i16) -> ()
  // CHECK-NEXT: -1
  func.call @check_muli(%cst1, %cst_n1) : (i16, i16) -> ()
  // CHECK-NEXT: 1
  func.call @check_muli(%cst_n1, %cst_n1) : (i16, i16) -> ()
  // CHECK-NEXT: -3
  func.call @check_muli(%cst1, %cst_n3) : (i16, i16) -> ()

  // CHECK-NEXT: 169
  func.call @check_muli(%cst13, %cst13) : (i16, i16) -> ()
  // CHECK-NEXT: 481
  func.call @check_muli(%cst13, %cst37) : (i16, i16) -> ()
  // CHECK-NEXT: 1554
  func.call @check_muli(%cst37, %cst42) : (i16, i16) -> ()

  // CHECK-NEXT: -256
  func.call @check_muli(%cst_n1, %cst256) : (i16, i16) -> ()
  // CHECK-NEXT: 3328
  func.call @check_muli(%cst256, %cst13) : (i16, i16) -> ()
  // CHECK-NEXT: 9472
  func.call @check_muli(%cst256, %cst37) : (i16, i16) -> ()
  // CHECK-NEXT: -768
  func.call @check_muli(%cst256, %cst_n3) : (i16, i16) -> ()

  // CHECK-NEXT: 32755
  func.call @check_muli(%cst13, %cst_i16_max) : (i16, i16) -> ()
  // CHECK-NEXT: -32768
  func.call @check_muli(%cst_i16_min, %cst37) : (i16, i16) -> ()

  // CHECK-NEXT: 1
  func.call @check_muli(%cst_i16_max, %cst_i16_max) : (i16, i16) -> ()
  // CHECK-NEXT: -32768
  func.call @check_muli(%cst_i16_min, %cst13) : (i16, i16) -> ()
  // CHECK-NEXT: 0
  func.call @check_muli(%cst_i16_min, %cst_i16_min) : (i16, i16) -> ()

  return
}
