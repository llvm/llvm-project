// Check that the ceildivsi lowering is correct.
// We do not check any poison or UB values, as it is not possible to catch them.

// RUN: mlir-opt %s --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-runner -e entry --entry-point-result=void \
// RUN:               --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @check_ceildivsi(%lhs : i32, %rhs : i32) -> () {
  %res = arith.ceildivsi %lhs, %rhs : i32
  vector.print %res : i32
  return
}

func.func @entry() {
  %int_min = arith.constant -2147483648 : i32
  %int_max = arith.constant 2147483647 : i32
  %minus_three = arith.constant -3 : i32
  %minus_two = arith.constant -2 : i32
  %minus_one = arith.constant -1 : i32
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  %three = arith.constant 3 : i32 

  // INT_MAX divided by values.
  // CHECK: 1
  func.call @check_ceildivsi(%int_max, %int_max) : (i32, i32) -> ()
  // CHECK-NEXT: 0
  func.call @check_ceildivsi(%int_max, %int_min) : (i32, i32) -> ()   
  // CHECK-NEXT: -2147483647
  func.call @check_ceildivsi(%int_max, %minus_one) : (i32, i32) -> ()
  // CHECK-NEXT: -1073741823
  func.call @check_ceildivsi(%int_max, %minus_two) : (i32, i32) -> ()
  // CHECK-NEXT: 2147483647
  func.call @check_ceildivsi(%int_max, %one) : (i32, i32) -> ()
  // CHECK-NEXT: 1073741824
  func.call @check_ceildivsi(%int_max, %two) : (i32, i32) -> ()

  // INT_MIN divided by values.
  // We do not check the result of INT_MIN divided by -1, as it is UB.
  // CHECK-NEXT: 1
  func.call @check_ceildivsi(%int_min, %int_min) : (i32, i32) -> ()
  // CHECK-NEXT: -1
  func.call @check_ceildivsi(%int_min, %int_max) : (i32, i32) -> ()
  // CHECK-NEXT: 1073741824
  func.call @check_ceildivsi(%int_min, %minus_two) : (i32, i32) -> ()
  // CHECK-NEXT: -2147483648
  func.call @check_ceildivsi(%int_min, %one) : (i32, i32) -> ()
  // CHECK-NEXT: -1073741824
  func.call @check_ceildivsi(%int_min, %two) : (i32, i32) -> ()

  // Divide values by INT_MIN.
  // CHECK-NEXT: 0
  func.call @check_ceildivsi(%one, %int_min) : (i32, i32) -> ()
  // CHECK-NEXT: 0
  func.call @check_ceildivsi(%two, %int_min) : (i32, i32) -> ()
  // CHECK-NEXT: 1
  func.call @check_ceildivsi(%minus_one, %int_min) : (i32, i32) -> ()
  // CHECK-NEXT: 1
  func.call @check_ceildivsi(%minus_two, %int_min) : (i32, i32) -> ()

  // Divide values by INT_MAX.
  // CHECK-NEXT: 1
  func.call @check_ceildivsi(%one, %int_max) : (i32, i32) -> ()
  // CHECK-NEXT: 1
  func.call @check_ceildivsi(%two, %int_max) : (i32, i32) -> ()
  // CHECK-NEXT: 0
  func.call @check_ceildivsi(%minus_one, %int_max) : (i32, i32) -> ()
  // CHECK-NEXT: 0
  func.call @check_ceildivsi(%minus_two, %int_max) : (i32, i32) -> ()

  // Check divisions by 2.
  // CHECK-NEXT: -1
  func.call @check_ceildivsi(%minus_three, %two) : (i32, i32) -> ()
  // CHECK-NEXT: -1
  func.call @check_ceildivsi(%minus_two, %two) : (i32, i32) -> ()
  // CHECK-NEXT: 0
  func.call @check_ceildivsi(%minus_one, %two) : (i32, i32) -> ()
  // CHECK-NEXT: 0
  func.call @check_ceildivsi(%zero, %two) : (i32, i32) -> ()
  // CHECK-NEXT: 1
  func.call @check_ceildivsi(%one, %two) : (i32, i32) -> ()
  // CHECK-NEXT: 1
  func.call @check_ceildivsi(%two, %two) : (i32, i32) -> ()
  // CHECK-NEXT: 2
  func.call @check_ceildivsi(%three, %two) : (i32, i32) -> ()

  // Check divisions by -2.
  // CHECK-NEXT: 2
  func.call @check_ceildivsi(%minus_three, %minus_two) : (i32, i32) -> ()
  // CHECK-NEXT: 1
  func.call @check_ceildivsi(%minus_two, %minus_two) : (i32, i32) -> ()
  // CHECK-NEXT: 1
  func.call @check_ceildivsi(%minus_one, %minus_two) : (i32, i32) -> ()
  // CHECK-NEXT: 0
  func.call @check_ceildivsi(%zero, %minus_two) : (i32, i32) -> ()
  // CHECK-NEXT: 0
  func.call @check_ceildivsi(%one, %minus_two) : (i32, i32) -> ()
  // CHECK-NEXT: -1
  func.call @check_ceildivsi(%two, %minus_two) : (i32, i32) -> ()
  // CHECK-NEXT: -1
  func.call @check_ceildivsi(%three, %minus_two) : (i32, i32) -> ()
  return
}
