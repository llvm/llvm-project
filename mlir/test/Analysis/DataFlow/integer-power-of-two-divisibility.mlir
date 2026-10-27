// RUN: mlir-opt %s -test-int-divisibility-analysis="power-of-two" | FileCheck %s

// CHECK-LABEL: @arithmetic
func.func @arithmetic(%a: index, %b: index, %p: i1) {
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %a8 = arith.muli %a, %c8 : index
  %b12 = arith.muli %b, %c12 : index
  %sum = arith.addi %a8, %b12 : index
  // CHECK: divisibility = "udiv = 4, sdiv = 4"
  "test.int_divisibility"(%sum) : (index) -> ()
  %difference = arith.subi %a8, %b12 : index
  // CHECK: divisibility = "udiv = 4, sdiv = 4"
  "test.int_divisibility"(%difference) : (index) -> ()
  %selected = arith.select %p, %a8, %b12 : index
  // CHECK: divisibility = "udiv = 4, sdiv = 4"
  "test.int_divisibility"(%selected) : (index) -> ()
  return
}

// CHECK-LABEL: @wide_constants_and_products
func.func @wide_constants_and_products(%a: i128) {
  %wide = arith.constant 170141183460469231731687303715884105728 : i128
  // CHECK: divisibility = "udiv = 9223372036854775808, sdiv = 9223372036854775808"
  "test.int_divisibility"(%wide) : (i128) -> ()
  %a_wide = arith.muli %a, %wide : i128
  %two = arith.constant 2 : i128
  %product = arith.muli %a_wide, %two : i128
  // Saturated divisor products must not assert that the value is exactly zero.
  // CHECK: divisibility = "udiv = 9223372036854775808, sdiv = 9223372036854775808"
  "test.int_divisibility"(%product) : (i128) -> ()
  %zero = arith.constant 0 : i128
  %zero_product = arith.muli %a, %zero : i128
  // CHECK: divisibility = "udiv = 0, sdiv = 0"
  "test.int_divisibility"(%zero_product) : (i128) -> ()
  return
}

// CHECK-LABEL: @division_is_pessimistic
func.func @division_is_pessimistic(%a: index) {
  %c24 = arith.constant 24 : index
  %c3 = arith.constant 3 : index
  %product = arith.muli %a, %c24 : index
  %quotient = arith.divui %product, %c3 : index
  // CHECK: divisibility = "udiv = 1, sdiv = 1"
  "test.int_divisibility"(%quotient) : (index) -> ()
  %zero = arith.constant 0 : index
  %undefined = arith.divui %a, %zero : index
  // CHECK: divisibility = "udiv = 1, sdiv = 1"
  "test.int_divisibility"(%undefined) : (index) -> ()
  return
}

// CHECK-LABEL: @induction_variables
func.func @induction_variables() {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c24 = arith.constant 24 : index
  scf.for %iv = %c0 to %c24 step %c6 {
    // Retain the power-of-two part of gcd(0, 6), not the odd factor three.
    // CHECK: divisibility = "udiv = 2, sdiv = 2"
    "test.int_divisibility"(%iv) : (index) -> ()
  }
  %c3 = arith.constant 3 : index
  scf.for %iv = %c3 to %c24 step %c6 {
    // CHECK: divisibility = "udiv = 1, sdiv = 1"
    "test.int_divisibility"(%iv) : (index) -> ()
  }
  return
}

// CHECK-LABEL: @wide_induction_variable
func.func @wide_induction_variable() {
  %c0 = arith.constant 0 : i128
  %c8 = arith.constant 8 : i128
  %c32 = arith.constant 32 : i128
  scf.for %iv = %c0 to %c32 step %c8 : i128 {
    // Loop-bound inference representing constants in int64_t is not invoked.
    // CHECK: divisibility = "udiv = 1, sdiv = 1"
    "test.int_divisibility"(%iv) : (i128) -> ()
  }
  return
}
