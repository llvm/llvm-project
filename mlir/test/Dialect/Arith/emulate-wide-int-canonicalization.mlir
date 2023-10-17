// RUN: mlir-opt --arith-emulate-wide-int="widest-int-supported=32" --canonicalize %s | FileCheck %s

// Check that we can fold away the 'hi' part calculation when it is know to be zero.
//
// CHECK-LABEL: func @uitofp_i16_ext_f64
// CHECK-SAME:    ([[ARG:%.+]]: i16) -> f64
// CHECK-NEXT:    [[EXT:%.+]] = arith.extui [[ARG]] : i16 to i32
// CHECK-NEXT:    [[FP:%.+]]  = arith.uitofp [[EXT]] : i32 to f64
// CHECK-NEXT:    return [[FP]] : f64
func.func @uitofp_i16_ext_f64(%a : i16) -> f64 {
  %ext = arith.extui %a : i16 to i64
  %r = arith.uitofp %ext : i64 to f64
  return %r : f64
}
