// RUN: mlir-opt %s -test-wrap-in-zero-trip-check -split-input-file  | FileCheck %s

func.func @no_wrap_scf_while_in_zero_trip_check(%bound : i32) -> i32 {
  %cst0 = arith.constant 0 : i32
  %cst5 = arith.constant 5 : i32
  %res:2 = scf.while (%iter = %cst0) : (i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %iter, %bound : i32
    %inv = arith.addi %bound, %cst5 : i32
    scf.condition(%cond) %iter, %inv : i32, i32
  } do {
  ^bb0(%arg1: i32, %arg2: i32):
    %next = arith.addi %arg1, %arg2 : i32
    scf.yield %next : i32
  }
  return %res#0 : i32
}

// TODO(pzread): Update the test once the wrapInZeroTripCheck is implemented.
// CHECK-LABEL: func.func @no_wrap_scf_while_in_zero_trip_check
// CHECK-NOT:     scf.if
// CHECK:         scf.while
