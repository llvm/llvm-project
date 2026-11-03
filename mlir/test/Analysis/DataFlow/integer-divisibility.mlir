// RUN: mlir-opt --split-input-file --test-int-divisibility-analysis %s | FileCheck %s

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

// CHECK-LABEL: @scf_forall_constant_step
func.func @scf_forall_constant_step() {
  scf.forall (%iv) = (0) to (12) step (3) {
    // CHECK: divisibility = "udiv = 3, sdiv = 3"
    %0 = "test.int_divisibility"(%iv) : (index) -> index
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_nested_tiled
func.func @scf_for_nested_tiled(%n : index, %m : index) {
  %c0 = arith.constant 0 : index
  %c48 = arith.constant 48 : index
  %c12 = arith.constant 12 : index
  scf.for %i = %c0 to %n step %c48 {
    // CHECK: divisibility = "udiv = 48, sdiv = 48"
    %0 = "test.int_divisibility"(%i) : (index) -> index
    scf.for %j = %i to %m step %c12 {
      // CHECK: divisibility = "udiv = 4, sdiv = 12"
      %1 = "test.int_divisibility"(%j) : (index) -> index
    }
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

// CHECK-LABEL: @scf_for_negative_lower_bound
func.func @scf_for_negative_lower_bound() {
  %lb = arith.constant -1 : index
  %ub = arith.constant 6 : index
  %step = arith.constant 3 : index
  scf.for %iv = %lb to %ub step %step {
    // CHECK: divisibility = "udiv = 1, sdiv = 1"
    %0 = "test.int_divisibility"(%iv) : (index) -> index
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_negative_lower_bound_i64
func.func @scf_for_negative_lower_bound_i64() {
  %lb = arith.constant -1 : i64
  %ub = arith.constant 6 : i64
  %step = arith.constant 3 : i64
  scf.for %iv = %lb to %ub step %step : i64 {
    // CHECK: divisibility = "udiv = 1, sdiv = 1"
    %0 = "test.int_divisibility"(%iv) : (i64) -> i64
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_negative_odd_divisor
func.func @scf_for_negative_odd_divisor() {
  %lb = arith.constant -6 : i32
  %ub = arith.constant 8 : i32
  %step = arith.constant 9 : i32
  scf.for %iv = %lb to %ub step %step : i32 {
    // The IVs are -6 and 3: 3 divides both signed values, but not both
    // unsigned representations.
    // CHECK: divisibility = "udiv = 1, sdiv = 3"
    %0 = "test.int_divisibility"(%iv) : (i32) -> i32
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_negative_power_of_two
func.func @scf_for_negative_power_of_two() {
  %lb = arith.constant -8 : i32
  %ub = arith.constant 9 : i32
  %step = arith.constant 16 : i32
  scf.for %iv = %lb to %ub step %step : i32 {
    // CHECK: divisibility = "udiv = 8, sdiv = 8"
    %0 = "test.int_divisibility"(%iv) : (i32) -> i32
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_min_signed_lower_bound
func.func @scf_for_min_signed_lower_bound() {
  %lb = arith.constant -9223372036854775808 : i64
  %ub = arith.constant 0 : i64
  %step = arith.constant 4611686018427387904 : i64
  scf.for %iv = %lb to %ub step %step : i64 {
    // CHECK: divisibility = "udiv = 4611686018427387904, sdiv = 4611686018427387904"
    %0 = "test.int_divisibility"(%iv) : (i64) -> i64
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_index_constant_truncation
func.func @scf_for_index_constant_truncation() {
  %lb = arith.constant 4294967299 : index
  %ub = arith.constant 4294967313 : index
  %step = arith.constant 7 : index
  scf.for %iv = %lb to %ub step %step {
    // On a 32-bit target, the lower bound becomes 3, which is not divisible
    // by 7 even though the 64-bit lower bound is.
    // CHECK: divisibility = "udiv = 1, sdiv = 1"
    %0 = "test.int_divisibility"(%iv) : (index) -> index
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_index_power_of_two_truncation
func.func @scf_for_index_power_of_two_truncation() {
  %lb = arith.constant 4294967296 : index
  %ub = arith.constant 4294967312 : index
  %step = arith.constant 8 : index
  scf.for %iv = %lb to %ub step %step {
    // The lower bound truncates to zero on 32-bit targets; both widths
    // retain the factor of eight.
    // CHECK: divisibility = "udiv = 8, sdiv = 8"
    %0 = "test.int_divisibility"(%iv) : (index) -> index
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_dynamic_index_wrap
func.func @scf_for_dynamic_index_wrap(%arg : index) {
  %seven = arith.constant 7 : index
  %ub = arith.constant 100 : index
  %lb = arith.muli %arg, %seven : index
  scf.for %iv = %lb to %ub step %seven {
    // %arg = 613566757 gives a 32-bit lower bound of 3.
    // CHECK: divisibility = "udiv = 1, sdiv = 1"
    %0 = "test.int_divisibility"(%iv) : (index) -> index
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_unsigned_lower_bound
func.func @scf_for_unsigned_lower_bound() {
  %lb = arith.constant -4 : i32
  %ub = arith.constant -1 : i32
  %step = arith.constant 3 : i32
  scf.for unsigned %iv = %lb to %ub step %step : i32 {
    // CHECK: divisibility = "udiv = 3, sdiv = 1"
    %0 = "test.int_divisibility"(%iv) : (i32) -> i32
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_unsigned_i32_width
func.func @scf_for_unsigned_i32_width() {
  %lb = arith.constant -11 : i32
  %ub = arith.constant -4 : i32
  %step = arith.constant 7 : i32
  scf.for unsigned %iv = %lb to %ub step %step : i32 {
    // 2^32 - 11 is divisible by 7; 2^64 - 11 is not.
    // CHECK: divisibility = "udiv = 7, sdiv = 1"
    %0 = "test.int_divisibility"(%iv) : (i32) -> i32
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_negative_without_zero_crossing
func.func @scf_for_negative_without_zero_crossing() {
  %lb = arith.constant -4 : i32
  %ub = arith.constant 0 : i32
  %step = arith.constant 3 : i32
  scf.for %iv = %lb to %ub step %step : i32 {
    // The IVs are -4 and -1, both divisible by 3 as unsigned i32 values.
    // CHECK: divisibility = "udiv = 3, sdiv = 1"
    %0 = "test.int_divisibility"(%iv) : (i32) -> i32
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_unsigned_positive_lower_bound
func.func @scf_for_unsigned_positive_lower_bound() {
  %lb = arith.constant 12 : i32
  %ub = arith.constant 100 : i32
  %step = arith.constant 18 : i32
  scf.for unsigned %iv = %lb to %ub step %step : i32 {
    // CHECK: divisibility = "udiv = 6, sdiv = 6"
    %0 = "test.int_divisibility"(%iv) : (i32) -> i32
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_dynamic_lower_bound
func.func @scf_for_dynamic_lower_bound(%cond : i1) {
  %negative = arith.constant -6 : i32
  %positive = arith.constant 3 : i32
  %ub = arith.constant 8 : i32
  %step = arith.constant 3 : i32
  %lb = arith.select %cond, %negative, %positive : i32
  scf.for %iv = %lb to %ub step %step : i32 {
    // Both lower bounds are divisible by 3 as signed values.
    // CHECK: divisibility = "udiv = 1, sdiv = 3"
    %0 = "test.int_divisibility"(%iv) : (i32) -> i32
  }
  return
}

// -----

// CHECK-LABEL: @scf_for_unsigned_dynamic_lower_bound
func.func @scf_for_unsigned_dynamic_lower_bound(%cond : i1) {
  %zero = arith.constant 0 : i32
  %negative = arith.constant -4 : i32
  %ub = arith.constant -1 : i32
  %step = arith.constant 3 : i32
  %lb = arith.select %cond, %negative, %zero : i32
  scf.for unsigned %iv = %lb to %ub step %step : i32 {
    // The unsigned lower bound is a multiple of 3 in either case.
    // CHECK: divisibility = "udiv = 3, sdiv = 1"
    %0 = "test.int_divisibility"(%iv) : (i32) -> i32
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
