// Check that the wide integer `arith.shrsi` emulation produces the same result as wide
// `arith.shrsi`. Emulate i16 ops with i8 ops.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

// RUN: mlir-opt %s --test-arith-emulate-wide-int="widest-int-supported=8" \
// RUN:             --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

// Ops in this function *only* will be emulated using i8 types.
func.func @emulate_shrsi(%lhs : i16, %rhs : i16) -> (i16) {
  %res = arith.shrsi %lhs, %rhs : i16
  return %res : i16
}

func.func @check_shrsi(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_shrsi(%lhs, %rhs) : (i16, i16) -> (i16)
  vector.print %res : i16
  return
}

func.func @entry() {
  %cst0 = arith.constant 0 : i16
  %cst1 = arith.constant 1 : i16
  %cst2 = arith.constant 2 : i16
  %cst7 = arith.constant 7 : i16
  %cst8 = arith.constant 8 : i16
  %cst9 = arith.constant 9 : i16
  %cst15 = arith.constant 15 : i16

  %cst_n1 = arith.constant -1 : i16

  %cst1337 = arith.constant 1337 : i16
  %cst_n1337 = arith.constant -1337 : i16

  %cst_i16_min = arith.constant -32768 : i16

  // CHECK:      -32768
  // CHECK-NEXT: -16384
  // CHECK-NEXT: -8192
  // CHECK-NEXT: -256
  // CHECK-NEXT: -128
  // CHECK-NEXT: -64
  // CHECK-NEXT: -1
  func.call @check_shrsi(%cst_i16_min, %cst0) : (i16, i16) -> ()
  func.call @check_shrsi(%cst_i16_min, %cst1) : (i16, i16) -> ()
  func.call @check_shrsi(%cst_i16_min, %cst2) : (i16, i16) -> ()
  func.call @check_shrsi(%cst_i16_min, %cst7) : (i16, i16) -> ()
  func.call @check_shrsi(%cst_i16_min, %cst8) : (i16, i16) -> ()
  func.call @check_shrsi(%cst_i16_min, %cst9) : (i16, i16) -> ()
  func.call @check_shrsi(%cst_i16_min, %cst15) : (i16, i16) -> ()

  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: -1
  // CHECK-NEXT: -1
  func.call @check_shrsi(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_shrsi(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_shrsi(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_shrsi(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_shrsi(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_shrsi(%cst_n1, %cst15) : (i16, i16) -> ()

  // CHECK-NEXT: 1337
  // CHECK-NEXT: 334
  // CHECK-NEXT: 10
  // CHECK-NEXT: 5
  // CHECK-NEXT: 2
  // CHECK-NEXT: 0
  func.call @check_shrsi(%cst1337, %cst0) : (i16, i16) -> ()
  func.call @check_shrsi(%cst1337, %cst2) : (i16, i16) -> ()
  func.call @check_shrsi(%cst1337, %cst7) : (i16, i16) -> ()
  func.call @check_shrsi(%cst1337, %cst8) : (i16, i16) -> ()
  func.call @check_shrsi(%cst1337, %cst9) : (i16, i16) -> ()
  func.call @check_shrsi(%cst1337, %cst15) : (i16, i16) -> ()

  // CHECK-NEXT: -1337
  // CHECK-NEXT: -335
  // CHECK-NEXT: -11
  // CHECK-NEXT: -6
  // CHECK-NEXT: -3
  // CHECK-NEXT: -1
  func.call @check_shrsi(%cst_n1337, %cst0) : (i16, i16) -> ()
  func.call @check_shrsi(%cst_n1337, %cst2) : (i16, i16) -> ()
  func.call @check_shrsi(%cst_n1337, %cst7) : (i16, i16) -> ()
  func.call @check_shrsi(%cst_n1337, %cst8) : (i16, i16) -> ()
  func.call @check_shrsi(%cst_n1337, %cst9) : (i16, i16) -> ()
  func.call @check_shrsi(%cst_n1337, %cst15) : (i16, i16) -> ()

  return
}
