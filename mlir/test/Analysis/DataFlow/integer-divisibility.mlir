// RUN: mlir-opt --split-input-file --test-int-divisibility-analysis --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @constant
func.func @constant() -> index {
  %0 = arith.constant 8 : index
  // CHECK: divisibility = "udiv = 8, sdiv = 8"
  %1 = "test.int_divisibility"(%0) : (index) -> index
  return %1 : index
}

// -----

// CHECK-LABEL: @muli_constant
func.func @muli_constant(%arg0 : index) -> index {
  %c4 = arith.constant 4 : index
  %0 = arith.muli %arg0, %c4 : index
  // CHECK: divisibility = "udiv = 4, sdiv = 4"
  %1 = "test.int_divisibility"(%0) : (index) -> index
  return %1 : index
}

// -----

// CHECK-LABEL: @addi_gcd_of_muli_operands
func.func @addi_gcd_of_muli_operands(%arg0 : index, %arg1 : index) -> index {
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %a = arith.muli %arg0, %c8 : index
  %b = arith.muli %arg1, %c12 : index
  %0 = arith.addi %a, %b : index
  // gcd(8, 12) = 4.
  // CHECK: divisibility = "udiv = 4, sdiv = 4"
  %1 = "test.int_divisibility"(%0) : (index) -> index
  return %1 : index
}

// -----

// CHECK-LABEL: @addi_same_divisibility
func.func @addi_same_divisibility(%arg0 : index, %arg1 : index) -> index {
  %c16 = arith.constant 16 : index
  %a = arith.muli %arg0, %c16 : index
  %b = arith.muli %arg1, %c16 : index
  %0 = arith.addi %a, %b : index
  // CHECK: divisibility = "udiv = 16, sdiv = 16"
  %1 = "test.int_divisibility"(%0) : (index) -> index
  return %1 : index
}

// -----

// CHECK-LABEL: @affine_apply_mul
func.func @affine_apply_mul(%arg0 : index) -> index {
  %c2 = arith.constant 2 : index
  %seed = arith.muli %arg0, %c2 : index
  %0 = affine.apply affine_map<(d0) -> (d0 * 16)>(%seed)
  // 2 * 16 = 32.
  // CHECK: divisibility = "udiv = 32, sdiv = 32"
  %1 = "test.int_divisibility"(%0) : (index) -> index
  return %1 : index
}

// -----

// CHECK-LABEL: @affine_apply_mul_then_floordiv
func.func @affine_apply_mul_then_floordiv(%arg0 : index) -> index {
  %0 = affine.apply affine_map<(d0) -> (d0 * 16)>(%arg0)
  %1 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%0)
  // 16 floordiv 4 = 4.
  // CHECK: divisibility = "udiv = 4, sdiv = 4"
  %2 = "test.int_divisibility"(%1) : (index) -> index
  return %2 : index
}

// -----

// CHECK-LABEL: @affine_apply_mod_zero
func.func @affine_apply_mod_zero(%arg0 : index) -> index {
  %0 = affine.apply affine_map<(d0) -> (d0 * 16)>(%arg0)
  %1 = affine.apply affine_map<(d0) -> (d0 mod 16)>(%0)
  // 16 % 16 == 0, so x mod 16 is always 0 -> divisibility 0 (lattice top).
  // CHECK: divisibility = "udiv = 0, sdiv = 0"
  %2 = "test.int_divisibility"(%1) : (index) -> index
  return %2 : index
}

// -----

// CHECK-LABEL: @affine_apply_constant
func.func @affine_apply_constant() -> index {
  %0 = affine.apply affine_map<() -> (64)>()
  // CHECK: divisibility = "udiv = 64, sdiv = 64"
  %1 = "test.int_divisibility"(%0) : (index) -> index
  return %1 : index
}

// -----

// CHECK-LABEL: @scf_for_constant_step
func.func @scf_for_constant_step() {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c8 = arith.constant 8 : index
  scf.for %iv = %c0 to %c64 step %c8 {
    // CHECK: divisibility = "udiv = 8, sdiv = 8"
    %0 = "test.int_divisibility"(%iv) : (index) -> index
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_nontrivial_gcd
func.func @scf_for_nontrivial_gcd() {
  %c12 = arith.constant 12 : index
  %c100 = arith.constant 100 : index
  %c18 = arith.constant 18 : index
  scf.for %iv = %c12 to %c100 step %c18 {
    // gcd(12, 18) = 6.
    // CHECK: divisibility = "udiv = 6, sdiv = 6"
    %0 = "test.int_divisibility"(%iv) : (index) -> index
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_coprime
func.func @scf_for_coprime() {
  %c15 = arith.constant 15 : index
  %c100 = arith.constant 100 : index
  %c8 = arith.constant 8 : index
  scf.for %iv = %c15 to %c100 step %c8 {
    // gcd(15, 8) = 1.
    // CHECK: divisibility = "udiv = 1, sdiv = 1"
    %0 = "test.int_divisibility"(%iv) : (index) -> index
  }
  return
}

// -----

// CHECK-LABEL: @affine_apply_mul_plus_const
func.func @affine_apply_mul_plus_const(%arg0 : index) -> index {
  %c4 = arith.constant 4 : index
  %seed = arith.muli %arg0, %c4 : index
  %0 = affine.apply affine_map<(d0) -> (d0 * 8 + 16)>(%seed)
  // seed has udiv = 4, multiplied by 8 -> 32, then +16. gcd(32,16) = 16.
  // CHECK: divisibility = "udiv = 16, sdiv = 16"
  %1 = "test.int_divisibility"(%0) : (index) -> index
  return %1 : index
}
