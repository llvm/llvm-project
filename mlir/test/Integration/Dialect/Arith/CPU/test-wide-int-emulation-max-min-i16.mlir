// Check that the wide integer `arith.max*i`/`min*i` emulation produces the
// same result as wide ops. Emulate i16 ops with i8 ops.
// Ops in functions prefixed with `emulate` will be emulated using i8 types.

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

func.func @emulate_maxui(%lhs : i16, %rhs : i16) -> (i16) {
  %res = arith.maxui %lhs, %rhs : i16
  return %res : i16
}

func.func @check_maxui(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_maxui(%lhs, %rhs) : (i16, i16) -> (i16)
  vector.print %res : i16
  return
}

func.func @emulate_maxsi(%lhs : i16, %rhs : i16) -> (i16) {
  %res = arith.maxsi %lhs, %rhs : i16
  return %res : i16
}

func.func @check_maxsi(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_maxsi(%lhs, %rhs) : (i16, i16) -> (i16)
  vector.print %res : i16
  return
}

func.func @emulate_minui(%lhs : i16, %rhs : i16) -> (i16) {
  %res = arith.minui %lhs, %rhs : i16
  return %res : i16
}

func.func @check_minui(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_minui(%lhs, %rhs) : (i16, i16) -> (i16)
  vector.print %res : i16
  return
}

func.func @emulate_minsi(%lhs : i16, %rhs : i16) -> (i16) {
  %res = arith.minsi %lhs, %rhs : i16
  return %res : i16
}

func.func @check_minsi(%lhs : i16, %rhs : i16) -> () {
  %res = func.call @emulate_minsi(%lhs, %rhs) : (i16, i16) -> (i16)
  vector.print %res : i16
  return
}


func.func @entry() {
  %cst0 = arith.constant 0 : i16
  %cst1 = arith.constant 1 : i16
  %cst7 = arith.constant 7 : i16
  %cst_n1 = arith.constant -1 : i16
  %cst1337 = arith.constant 1337 : i16
  %cst4096 = arith.constant 4096 : i16
  %cst_i16_min = arith.constant -32768 : i16

  // CHECK:      0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: -1
  // CHECK-NEXT: -1
  // CHECK-NEXT: -1
  // CHECK-NEXT: 1337
  // CHECK-NEXT: 4096
  // CHECK-NEXT: -32768
  func.call @check_maxui(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_maxui(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_maxui(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_maxui(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_maxui(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_maxui(%cst1, %cst_n1) : (i16, i16) -> ()
  func.call @check_maxui(%cst_n1, %cst1337) : (i16, i16) -> ()
  func.call @check_maxui(%cst1337, %cst1337) : (i16, i16) -> ()
  func.call @check_maxui(%cst4096, %cst4096) : (i16, i16) -> ()
  func.call @check_maxui(%cst1337, %cst_i16_min) : (i16, i16) -> ()

  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1337
  // CHECK-NEXT: 1337
  // CHECK-NEXT: 4096
  // CHECK-NEXT: 1337
  func.call @check_maxsi(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_maxsi(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_maxsi(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_maxsi(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_maxsi(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_maxsi(%cst1, %cst_n1) : (i16, i16) -> ()
  func.call @check_maxsi(%cst_n1, %cst1337) : (i16, i16) -> ()
  func.call @check_maxsi(%cst1337, %cst1337) : (i16, i16) -> ()
  func.call @check_maxsi(%cst4096, %cst4096) : (i16, i16) -> ()
  func.call @check_maxsi(%cst1337, %cst_i16_min) : (i16, i16) -> ()

  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1337
  // CHECK-NEXT: 1337
  // CHECK-NEXT: 4096
  // CHECK-NEXT: 1337
  func.call @check_minui(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_minui(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_minui(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_minui(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_minui(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_minui(%cst1, %cst_n1) : (i16, i16) -> ()
  func.call @check_minui(%cst_n1, %cst1337) : (i16, i16) -> ()
  func.call @check_minui(%cst1337, %cst1337) : (i16, i16) -> ()
  func.call @check_minui(%cst4096, %cst4096) : (i16, i16) -> ()
  func.call @check_minui(%cst1337, %cst_i16_min) : (i16, i16) -> ()

  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  // CHECK-NEXT: -1
  // CHECK-NEXT: -1
  // CHECK-NEXT: -1
  // CHECK-NEXT: 1337
  // CHECK-NEXT: 4096
  // CHECK-NEXT: -32768
  func.call @check_minsi(%cst0, %cst0) : (i16, i16) -> ()
  func.call @check_minsi(%cst0, %cst1) : (i16, i16) -> ()
  func.call @check_minsi(%cst1, %cst0) : (i16, i16) -> ()
  func.call @check_minsi(%cst1, %cst1) : (i16, i16) -> ()
  func.call @check_minsi(%cst_n1, %cst1) : (i16, i16) -> ()
  func.call @check_minsi(%cst1, %cst_n1) : (i16, i16) -> ()
  func.call @check_minsi(%cst_n1, %cst1337) : (i16, i16) -> ()
  func.call @check_minsi(%cst1337, %cst1337) : (i16, i16) -> ()
  func.call @check_minsi(%cst4096, %cst4096) : (i16, i16) -> ()
  func.call @check_minsi(%cst1337, %cst_i16_min) : (i16, i16) -> ()

  return
}
