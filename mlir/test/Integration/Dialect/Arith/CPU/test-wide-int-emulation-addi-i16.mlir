// Check that the wide integer addition emulation produces the same result as
// wide addition. Emulate i16 ops with i8 ops.

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
func.func @emulate_addi(%lhs : i16, %rhs : i16) -> (i16) {
  %res = arith.addi %lhs, %rhs : i16
  return %res : i16
}

func.func @check_addi(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_addi(%lhs, %rhs) : (i16, i16) -> (i16)
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
  func.call @check_addi(%cst0, %cst0) : (i16, i16) -> ()
  // CHECK-NEXT: 1
  func.call @check_addi(%cst0, %cst1) : (i16, i16) -> ()
  // CHECK-NEXT: 2
  func.call @check_addi(%cst1, %cst1) : (i16, i16) -> ()
  // CHECK-NEXT: 0
  func.call @check_addi(%cst1, %cst_n1) : (i16, i16) -> ()
  // CHECK-NEXT: -2
  func.call @check_addi(%cst_n1, %cst_n1) : (i16, i16) -> ()
  // CHECK-NEXT: -2
  func.call @check_addi(%cst1, %cst_n3) : (i16, i16) -> ()

  // CHECK-NEXT: 26
  func.call @check_addi(%cst13, %cst13) : (i16, i16) -> ()
  // CHECK-NEXT: 50
  func.call @check_addi(%cst13, %cst37) : (i16, i16) -> ()
  // CHECK-NEXT: 79
  func.call @check_addi(%cst37, %cst42) : (i16, i16) -> ()

  // CHECK-NEXT: 255
  func.call @check_addi(%cst_n1, %cst256) : (i16, i16) -> ()
  // CHECK-NEXT: 269
  func.call @check_addi(%cst256, %cst13) : (i16, i16) -> ()
  // CHECK-NEXT: 293
  func.call @check_addi(%cst256, %cst37) : (i16, i16) -> ()
  // CHECK-NEXT: 253
  func.call @check_addi(%cst256, %cst_n3) : (i16, i16) -> ()

  // CHECK-NEXT: -32756
  func.call @check_addi(%cst13, %cst_i16_max) : (i16, i16) -> ()
  // CHECK-NEXT: -32731
  func.call @check_addi(%cst_i16_min, %cst37) : (i16, i16) -> ()

  // CHECK-NEXT: -2
  func.call @check_addi(%cst_i16_max, %cst_i16_max) : (i16, i16) -> ()
  // CHECK-NEXT: -32755
  func.call @check_addi(%cst_i16_min, %cst13) : (i16, i16) -> ()
  // CHECK-NEXT: 0
  func.call @check_addi(%cst_i16_min, %cst_i16_min) : (i16, i16) -> ()

  return
}
