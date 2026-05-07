// RUN: mlir-opt -int-range-optimizations %s | FileCheck %s

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
