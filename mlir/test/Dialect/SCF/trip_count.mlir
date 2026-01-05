// RUN: mlir-opt %s  -test-scf-for-utils --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @trip_count_index_zero_to_zero(
func.func @trip_count_index_zero_to_zero(%a : i32, %b : i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: "test.trip-count" = 0
  %r = scf.for %i = %c0 to %c0 step %c1 iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_index_zero_to_zero_step_dyn(
func.func @trip_count_index_zero_to_zero_step_dyn(%a : i32, %b : i32, %step : index) -> i32 {
  %c0 = arith.constant 0 : index

  // CHECK: "test.trip-count" = 0
  %r = scf.for %i = %c0 to %c0 step %step iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_zero_to_zero(
func.func @trip_count_i32_zero_to_zero(%a : i32, %b : i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32

  // CHECK: "test.trip-count" = 0
  %r = scf.for %i = %c0 to %c0 step %c1 iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r : i32
}

// -----


// CHECK-LABEL: func.func @trip_count_i32_zero_to_zero_step_dyn(
func.func @trip_count_i32_zero_to_zero_step_dyn(%a : i32, %b : i32, %step : i32) -> i32 {
  %c0 = arith.constant 0 : i32

  // CHECK: "test.trip-count" = 0
  %r = scf.for %i = %c0 to %c0 step %step iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_index_one_to_zero(
func.func @trip_count_index_one_to_zero(%a : i32, %b : i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Index type has a unknown bitwidth, we can't compute a loop tripcount
  // in theory because of overflow concerns.
  // CHECK: "test.trip-count" = 0
  %r2 = scf.for %i = %c1 to %c0 step %c1 iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r2 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_one_to_zero(
func.func @trip_count_i32_one_to_zero(%a : i32, %b : i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32

  // CHECK: "test.trip-count" = 0
  %r2 = scf.for %i = %c1 to %c0 step %c1 iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r2 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_one_to_zero_dyn_step(
func.func @trip_count_i32_one_to_zero_dyn_step(%a : i32, %b : i32, %step : i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32

  // CHECK: "test.trip-count" = 0
  %r2 = scf.for %i = %c1 to %c0 step %step iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r2 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_index_negative_step(
func.func @trip_count_index_negative_step(%a : i32, %b : i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c-1 = arith.constant -1 : index

  // Negative step is invalid, loop won't execute.
  // CHECK: "test.trip-count" = 0
  %r3 = scf.for %i = %c1 to %c0 step %c-1 iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r3 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_negative_step(
func.func @trip_count_i32_negative_step(%a : i32, %b : i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c-1 = arith.constant -1 : i32

  // Negative step is invalid, loop won't execute.
  // CHECK: "test.trip-count" = 0
  %r3 = scf.for %i = %c1 to %c0 step %c-1 iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r3 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_index_negative_step_unsigned_loop(
func.func @trip_count_index_negative_step_unsigned_loop(%a : i32, %b : i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c-1 = arith.constant -1 : index

  // Negative step is invalid, loop won't execute.
  // CHECK: "test.trip-count" = 0
  %r3 = scf.for unsigned %i = %c1 to %c0 step %c-1 iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r3 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_negative_step_unsigned_loop(
func.func @trip_count_i32_negative_step_unsigned_loop(%a : i32, %b : i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c-1 = arith.constant -1 : i32

  // Negative step is invalid, loop won't execute.
  // CHECK: "test.trip-count" = 0
  %r3 = scf.for unsigned %i = %c1 to %c0 step %c-1 iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r3 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_index_normal_loop(
func.func @trip_count_index_normal_loop(%a : i32, %b : i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index

  // Index type has a unknown bitwidth, we can't compute a loop tripcount
  // in theory because of overflow concerns.
  // CHECK: "test.trip-count" = 5
  %r4 = scf.for %i = %c0 to %c10 step %c2 iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r4 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_normal_loop(
func.func @trip_count_i32_normal_loop(%a : i32, %b : i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %c10 = arith.constant 10 : i32

  // Normal loop
  // CHECK: "test.trip-count" = 5
  %r4 = scf.for %i = %c0 to %c10 step %c2 iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r4 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_index_signed_crossing_zero(
func.func @trip_count_index_signed_crossing_zero(%a : i32, %b : i32) -> i32 {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index

  // Index type has a unknown bitwidth, we can't compute a loop tripcount
  // in theory because of overflow concerns.
  // CHECK: "test.trip-count" = 2
  %r5 = scf.for %i = %c-1 to %c1 step %c1 iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r5 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_signed_crossing_zero(
func.func @trip_count_i32_signed_crossing_zero(%a : i32, %b : i32) -> i32 {
  %c-1 = arith.constant -1 : i32
  %c1 = arith.constant 1 : i32

  // This loop execute with signed comparison, but not unsigned, because it is crossing 0.
  // CHECK: "test.trip-count" = 2
  %r5 = scf.for %i = %c-1 to %c1 step %c1 iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r5 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_index_unsigned_crossing_zero(
func.func @trip_count_index_unsigned_crossing_zero(%a : i32, %b : i32) -> i32 {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index

  // Index type has a unknown bitwidth, we can't compute a loop tripcount
  // in theory because of overflow concerns.
  // CHECK: "test.trip-count" = 0
  %r6 = scf.for unsigned %i = %c-1 to %c1 step %c1 iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r6 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_unsigned_crossing_zero(
func.func @trip_count_i32_unsigned_crossing_zero(%a : i32, %b : i32) -> i32 {
  %c-1 = arith.constant -1 : i32
  %c1 = arith.constant 1 : i32

  // This loop execute with signed comparison, but not unsigned, because it is crossing 0.
  // CHECK: "test.trip-count" = 0
  %r6 = scf.for unsigned %i = %c-1 to %c1 step %c1 iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r6 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_unsigned_crossing_zero_dyn_step(
func.func @trip_count_i32_unsigned_crossing_zero_dyn_step(%a : i32, %b : i32, %step : i32) -> i32 {
  %c-1 = arith.constant -1 : i32
  %c1 = arith.constant 1 : i32

  // This loop execute with signed comparison, but not unsigned, because it is crossing 0.
  // CHECK: "test.trip-count" = 0
  %r6 = scf.for unsigned %i = %c-1 to %c1 step %step iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r6 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_index_negative_bounds_signed(
func.func @trip_count_index_negative_bounds_signed(%a : i32, %b : i32) -> i32 {
  %c-10 = arith.constant -10 : index
  %c-1 = arith.constant -1 : index
  %c2 = arith.constant 2 : index

  // Index type has a unknown bitwidth, we can't compute a loop tripcount
  // in theory because of overflow concerns.
  // CHECK: "test.trip-count" = 5
  %r7 = scf.for %i = %c-10 to %c-1 step %c2 iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r7 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_negative_bounds_signed(
func.func @trip_count_i32_negative_bounds_signed(%a : i32, %b : i32) -> i32 {
  %c-10 = arith.constant -10 : i32
  %c-1 = arith.constant -1 : i32
  %c2 = arith.constant 2 : i32

  // This loop execute with signed comparison, because both bounds are
  // negative and there is no crossing of 0 here.
  // CHECK: "test.trip-count" = 5
  %r7 = scf.for %i = %c-10 to %c-1 step %c2 iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r7 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_index_negative_bounds_unsigned(
func.func @trip_count_index_negative_bounds_unsigned(%a : i32, %b : i32) -> i32 {
  %c-10 = arith.constant -10 : index
  %c-1 = arith.constant -1 : index
  %c2 = arith.constant 2 : index

  // Index type has a unknown bitwidth, we can't compute a loop tripcount
  // in theory because of overflow concerns.
  // CHECK: "test.trip-count" = 5
  %r8 = scf.for %i = %c-10 to %c-1 step %c2 iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r8 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_negative_bounds_unsigned(
func.func @trip_count_i32_negative_bounds_unsigned(%a : i32, %b : i32) -> i32 {
  %c-10 = arith.constant -10 : i32
  %c-1 = arith.constant -1 : i32
  %c2 = arith.constant 2 : i32

  // CHECK: "test.trip-count" = 5
  %r8 = scf.for %i = %c-10 to %c-1 step %c2 iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r8 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_index_overflow_signed(
func.func @trip_count_index_overflow_signed(%a : i32, %b : i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c_max = arith.constant 2147483647 : index   // 2^31 - 1
  %c_min = arith.constant 2147483648 : index  // -2^31

  // Index type has a unknown bitwidth, we can't compute a loop tripcount
  // in theory because of overflow concerns.
  // CHECK: "test.trip-count" = 1
  %r9 = scf.for %i = %c_max to %c_min step %c1 iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r9 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_overflow_signed(
func.func @trip_count_i32_overflow_signed(%a : i32, %b : i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c_max = arith.constant 2147483647 : i32   // 2^31 - 1
  %c_min = arith.constant 2147483648 : i32  // -2^31

  // This loop crosses the 2^31 threshold, which would overflow a signed 32-bit integer.
  // CHECK: "test.trip-count" = 0
  %r9 = scf.for %i = %c_max to %c_min step %c1 iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r9 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_overflow_signed_dyn_step(
func.func @trip_count_i32_overflow_signed_dyn_step(%a : i32, %b : i32, %step : i32) -> i32 {
  %c_max = arith.constant 2147483647 : i32   // 2^31 - 1
  %c_min = arith.constant 2147483648 : i32  // -2^31

  // This loop crosses the 2^31 threshold, which would overflow a signed 32-bit integer.
  // CHECK: "test.trip-count" = 0
  %r9 = scf.for %i = %c_max to %c_min step %step iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r9 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_index_overflow_unsigned(
func.func @trip_count_index_overflow_unsigned(%a : i32, %b : i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c_max = arith.constant 2147483647 : index   // 2^31 - 1
  %c_min = arith.constant 2147483648 : index  // -2^31

  // Index type has a unknown bitwidth, we can't compute a loop tripcount
  // in theory because of overflow concerns.
  // CHECK: "test.trip-count" = 1
  %r10 = scf.for unsigned %i = %c_max to %c_min step %c1 iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r10 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_overflow_unsigned(
func.func @trip_count_i32_overflow_unsigned(%a : i32, %b : i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c_max = arith.constant 2147483647 : i32   // 2^31 - 1
  %c_min = arith.constant 2147483648 : i32  // -2^31

  // The same loop with unsigned comparison executes normally
  // CHECK: "test.trip-count" = 1
  %r10 = scf.for unsigned %i = %c_max to %c_min step %c1 iter_args(%0 = %a) -> i32 : i32 {
    scf.yield %b : i32
  }
  return %r10 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_index_overflow_64bit_signed(
func.func @trip_count_index_overflow_64bit_signed(%a : i32, %b : i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c_max = arith.constant 9223372036854775807 : index   // 2^63 - 1
  %c_min = arith.constant -9223372036854775808 : index  // -2^63

  // This loop crosses the 2^63 threshold, which would overflow a signed 64-bit integer.
  // Index type has a unknown bitwidth, we can't compute a loop tripcount.
  // CHECK: "test.trip-count" = 0
  %r11 = scf.for %i = %c_max to %c_min step %c1 iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r11 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i64_overflow_64bit_signed(
func.func @trip_count_i64_overflow_64bit_signed(%a : i32, %b : i32) -> i32 {
  %c1 = arith.constant 1 : i64
  %c_max = arith.constant 9223372036854775807 : i64   // 2^63 - 1
  %c_min = arith.constant -9223372036854775808 : i64  // -2^63

  // This loop crosses the 2^63 threshold, which would overflow a signed 64-bit integer.
  // CHECK: "test.trip-count" = 0
  %r11 = scf.for %i = %c_max to %c_min step %c1 iter_args(%0 = %a) -> i32 : i64 {
    scf.yield %b : i32
  }
  return %r11 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_index_overflow_64bit_unsigned(
func.func @trip_count_index_overflow_64bit_unsigned(%a : i32, %b : i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c_max = arith.constant 9223372036854775807 : index   // 2^63 - 1
  %c_min = arith.constant -9223372036854775808 : index  // -2^63

  // Index type has a unknown bitwidth, we can't compute a loop tripcount
  // in theory because of overflow concerns.
  // CHECK: "test.trip-count" = 1
  %r12 = scf.for unsigned %i = %c_max to %c_min step %c1 iter_args(%0 = %a) -> i32 {
    scf.yield %b : i32
  }
  return %r12 : i32
}

// -----

// CHECK-LABEL: func.func @trip_count_i32_overflow_64bit_unsigned(
func.func @trip_count_i32_overflow_64bit_unsigned(%a : i32, %b : i32) -> i32 {
  %c1 = arith.constant 1 : i64
  %c_max = arith.constant 9223372036854775807 : i64   // 2^63 - 1
  %c_min = arith.constant -9223372036854775808 : i64  // -2^63

  // The same loop with unsigned comparison executes normally
  // CHECK: "test.trip-count" = 1
  %r12 = scf.for unsigned %i = %c_max to %c_min step %c1 iter_args(%0 = %a) -> i32 : i64 {
    scf.yield %b : i32
  }
  return %r12 : i32
}

// -----

// CHECK-LABEL:func.func @trip_count_step_greater_than_iteration(
func.func @trip_count_step_greater_than_iteration() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c17_i32 = arith.constant 17 : i32
  %c16_i32 = arith.constant 16 : i32
  // CHECK: "test.trip-count" = 1
  %1 = scf.for %arg0 = %c16_i32 to %c17_i32 step %c4_i32 iter_args(%arg1 = %c0_i32) -> (i32)  : i32 {
    scf.yield %arg0 : i32
  }
  return %1 : i32
}


// -----

// CHECK-LABEL:func.func @trip_count_arith_add(
func.func @trip_count_arith_add(%lb : i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c17_i32 = arith.constant 17 : i32
  %c16_i32 = arith.constant 16 : i32
  // Can't compute a trip-count in the absence of overflow flag.
  // CHECK: "test.trip-count" = "none"
  %ub = arith.addi %lb, %c16_i32 : i32
  %1 = scf.for %arg0 = %lb to %ub step %c4_i32 iter_args(%arg1 = %c0_i32) -> (i32)  : i32 {
    scf.yield %arg0 : i32
  }
  return %1 : i32
}

// -----

// CHECK-LABEL:func.func @trip_count_arith_add_negative(
func.func @trip_count_arith_add_negative(%lb : i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c-16_i32 = arith.constant -16 : i32
  // Can't compute a trip-count in the absence of overflow flag.
  // CHECK: "test.trip-count" = "none"
  %ub = arith.addi %lb, %c-16_i32 : i32
  %1 = scf.for %arg0 = %lb to %ub step %c4_i32 iter_args(%arg1 = %c0_i32) -> (i32)  : i32 {
    scf.yield %arg0 : i32
  }
  return %1 : i32
}

// -----

// CHECK-LABEL:func.func @trip_count_arith_add_nsw_loop_signed(
func.func @trip_count_arith_add_nsw_loop_signed(%lb : i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  %ub = arith.addi %lb, %c16_i32 overflow<nsw> : i32
  // CHECK: "test.trip-count" = 4
  %1 = scf.for %arg0 = %lb to %ub step %c4_i32 iter_args(%arg1 = %c0_i32) -> (i32)  : i32 {
    scf.yield %arg0 : i32
  }
  return %1 : i32
}

// -----

// CHECK-LABEL:func.func @trip_count_arith_add_negative_nsw_loop_signed(
func.func @trip_count_arith_add_negative_nsw_loop_signed(%lb : i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c-16_i32 = arith.constant -16 : i32
  %ub = arith.addi %lb, %c-16_i32 overflow<nsw> : i32
  // CHECK: "test.trip-count" = 0
  %1 = scf.for %arg0 = %lb to %ub step %c4_i32 iter_args(%arg1 = %c0_i32) -> (i32)  : i32 {
    scf.yield %arg0 : i32
  }
  return %1 : i32
}

// -----

// CHECK-LABEL:func.func @trip_count_arith_add_negative_nsw_loop_signed_step_dyn(
func.func @trip_count_arith_add_negative_nsw_loop_signed_step_dyn(%lb : i32, %step : i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c-16_i32 = arith.constant -16 : i32
  %ub = arith.addi %lb, %c-16_i32 overflow<nsw> : i32
  // CHECK: "test.trip-count" = 0
  %1 = scf.for %arg0 = %lb to %ub step %step iter_args(%arg1 = %c0_i32) -> (i32)  : i32 {
    scf.yield %arg0 : i32
  }
  return %1 : i32
}

// -----

// CHECK-LABEL:func.func @trip_count_arith_add_nsw_loop_unsigned(
func.func @trip_count_arith_add_nsw_loop_unsigned(%lb : i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  // Can't compute a trip-count when the overflow flag mismatches the loop comparison signess
  // CHECK: "test.trip-count" = "none"
  %ub = arith.addi %lb, %c16_i32 overflow<nsw> : i32
  %1 = scf.for unsigned %arg0 = %lb to %ub step %c4_i32 iter_args(%arg1 = %c0_i32) -> (i32)  : i32 {
    scf.yield %arg0 : i32
  }
  return %1 : i32
}

// -----

// CHECK-LABEL:func.func @trip_count_arith_add_negative_nsw_loop_unsigned(
func.func @trip_count_arith_add_negative_nsw_loop_unsigned(%lb : i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c-16_i32 = arith.constant -16 : i32
  // Can't compute a trip-count when the overflow flag mismatches the loop comparison signess
  // CHECK: "test.trip-count" = "none"
  %ub = arith.addi %lb, %c-16_i32 overflow<nsw> : i32
  %1 = scf.for unsigned %arg0 = %lb to %ub step %c4_i32 iter_args(%arg1 = %c0_i32) -> (i32)  : i32 {
    scf.yield %arg0 : i32
  }
  return %1 : i32
}

// -----

// CHECK-LABEL:func.func @trip_count_arith_add_nuw_loop_signed(
func.func @trip_count_arith_add_nuw_loop_signed(%lb : i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  // Can't compute a trip-count when the overflow flag mismatches the loop comparison signess
  // CHECK: "test.trip-count" = "none"
  %ub = arith.addi %lb, %c16_i32 overflow<nuw> : i32
  %1 = scf.for %arg0 = %lb to %ub step %c4_i32 iter_args(%arg1 = %c0_i32) -> (i32)  : i32 {
    scf.yield %arg0 : i32
  }
  return %1 : i32
}

// -----

// CHECK-LABEL:func.func @trip_count_arith_add_negative_nuw_loop_signed(
func.func @trip_count_arith_add_negative_nuw_loop_signed(%lb : i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c-16_i32 = arith.constant -16 : i32
  // Can't compute a trip-count when the overflow flag mismatches the loop comparison signess
  // CHECK: "test.trip-count" = "none"
  %ub = arith.addi %lb, %c-16_i32 overflow<nuw> : i32
  %1 = scf.for %arg0 = %lb to %ub step %c4_i32 iter_args(%arg1 = %c0_i32) -> (i32)  : i32 {
    scf.yield %arg0 : i32
  }
  return %1 : i32
}

// -----

// CHECK-LABEL:func.func @trip_count_arith_add_nuw_loop_unsigned(
func.func @trip_count_arith_add_nuw_loop_unsigned(%lb : i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  // CHECK: "test.trip-count" = 4
  %ub = arith.addi %lb, %c16_i32 overflow<nuw> : i32
  %1 = scf.for unsigned %arg0 = %lb to %ub step %c4_i32 iter_args(%arg1 = %c0_i32) -> (i32)  : i32 {
    scf.yield %arg0 : i32
  }
  return %1 : i32
}

// -----

// CHECK-LABEL:func.func @trip_count_arith_add_negative_nuw_loop_unsigned(
func.func @trip_count_arith_add_negative_nuw_loop_unsigned(%lb : i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c-16_i32 = arith.constant -16 : i32
  // CHECK: "test.trip-count" = 0
  %ub = arith.addi %lb, %c-16_i32 overflow<nuw> : i32
  %1 = scf.for unsigned %arg0 = %lb to %ub step %c4_i32 iter_args(%arg1 = %c0_i32) -> (i32)  : i32 {
    scf.yield %arg0 : i32
  }
  return %1 : i32
}

// -----

// CHECK-LABEL:func.func @trip_count_arith_add_negative_nuw_loop_unsigned_step_dyn(
func.func @trip_count_arith_add_negative_nuw_loop_unsigned_step_dyn(%lb : i32, %step : i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c-16_i32 = arith.constant -16 : i32
  // CHECK: "test.trip-count" = 0
  %ub = arith.addi %lb, %c-16_i32 overflow<nuw> : i32
  %1 = scf.for unsigned %arg0 = %lb to %ub step %step iter_args(%arg1 = %c0_i32) -> (i32)  : i32 {
    scf.yield %arg0 : i32
  }
  return %1 : i32
}

// -----

// CHECK-LABEL:func.func @trip_count_arith_add_nuw_loop_unsigned_invalid(
func.func @trip_count_arith_add_nuw_loop_unsigned_invalid(%lb : i32, %other : i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c16_i32 = arith.constant 16 : i32
  // The addition here is not adding from %lb
  // CHECK: "test.trip-count" = "none"
  %ub = arith.addi %other, %c16_i32 overflow<nuw> : i32
  %1 = scf.for unsigned %arg0 = %lb to %ub step %c4_i32 iter_args(%arg1 = %c0_i32) -> (i32)  : i32 {
    scf.yield %arg0 : i32
  }
  return %1 : i32
}