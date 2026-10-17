// RUN: mlir-opt -int-range-optimizations %s | FileCheck %s

// CHECK-LABEL: func.func @repro_bitwidth_mismatch
func.func @repro_bitwidth_mismatch() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  // CHECK: test.region_types_compat
  %0 = "test.region_types_compat"(%c0_i32) ({
  ^bb0(%arg0: i64):
    %c1_i64 = arith.constant 1 : i64
    test.types_compat_yield %c1_i64 : i64
  }) : (i32) -> i32
  return %0 : i32
}

// -----

// CHECK-LABEL: func.func @repro_crash() -> !test.i32 {
func.func @repro_crash() -> !test.i32 {
  %cst = arith.constant 1 : i32
  // CHECK: %[[RES:.*]] = test.region_type_changer
  %0 = "test.region_type_changer"(%cst) ({
  ^bb0(%arg0: i32):
    "test.types_compat_yield"(%arg0) : (i32) -> ()
  }) : (i32) -> !test.i32
  // CHECK: return %[[RES]] : !test.i32
  return %0 : !test.i32
}
