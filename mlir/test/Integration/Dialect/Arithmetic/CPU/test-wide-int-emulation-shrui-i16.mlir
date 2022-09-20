// Check that the wide integer `arith.shrui` emulation produces the same result as wide
// `arith.shrui`. Emulate i16 ops with i8 ops.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN:   FileCheck %s --match-full-lines

// RUN: mlir-opt %s --test-arith-emulate-wide-int="widest-int-supported=8" \
// RUN:             --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN:   FileCheck %s --match-full-lines

// Ops in this function *only* will be emulated using i8 types.
func.func @emulate_shrui(%lhs : i16, %rhs : i16) -> (i16) {
  %res = arith.shrui %lhs, %rhs : i16
  return %res : i16
}

func.func @check_shrui(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_shrui(%lhs, %rhs) : (i16, i16) -> (i16)
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

  %cst_1 = arith.constant -1 : i16
  %cst_3 = arith.constant -3 : i16

  %cst1337 = arith.constant 1337 : i16

  %cst_i16_min = arith.constant -32768 : i16

  // CHECK:      0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 32767
  // CHECK-NEXT: 1
  func.call @check_shrui(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_shrui(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_shrui(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_shrui(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_shrui(%cst_1, %cst1) : (i16, i16) -> ()
  func.call @check_shrui(%cst_1, %cst15) : (i16, i16) -> ()

  // CHECK-NEXT: 1337
  // CHECK-NEXT: 334
  // CHECK-NEXT: 10
  // CHECK-NEXT: 5
  // CHECK-NEXT: 2
  // CHECK-NEXT: 0
  func.call @check_shrui(%cst1337, %cst0) : (i16, i16) -> ()
  func.call @check_shrui(%cst1337, %cst2) : (i16, i16) -> ()
  func.call @check_shrui(%cst1337, %cst7) : (i16, i16) -> ()
  func.call @check_shrui(%cst1337, %cst8) : (i16, i16) -> ()
  func.call @check_shrui(%cst1337, %cst9) : (i16, i16) -> ()
  func.call @check_shrui(%cst1337, %cst15) : (i16, i16) -> ()

  // CHECK-NEXT: 16384
  // CHECK-NEXT: 8192
  // CHECK-NEXT: 256
  // CHECK-NEXT: 128
  // CHECK-NEXT: 64
  // CHECK-NEXT: 1
  func.call @check_shrui(%cst_i16_min, %cst1) : (i16, i16) -> ()
  func.call @check_shrui(%cst_i16_min, %cst2) : (i16, i16) -> ()
  func.call @check_shrui(%cst_i16_min, %cst7) : (i16, i16) -> ()
  func.call @check_shrui(%cst_i16_min, %cst8) : (i16, i16) -> ()
  func.call @check_shrui(%cst_i16_min, %cst9) : (i16, i16) -> ()
  func.call @check_shrui(%cst_i16_min, %cst15) : (i16, i16) -> ()

  return
}
