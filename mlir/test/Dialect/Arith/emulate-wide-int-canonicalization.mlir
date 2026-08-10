// RUN: mlir-opt --arith-emulate-wide-int="widest-int-supported=32" --canonicalize %s | FileCheck %s

// Check that we can fold away the 'hi' part calculation when it is known to be zero.
//
// CHECK-LABEL: func @uitofp_i16_ext_f64
// CHECK-SAME:    ([[ARG:%.+]]: i16) -> f64
// CHECK:         [[EXT:%.+]] = arith.extui [[ARG]] : i16 to i32
// CHECK:         [[ORI:%.+]] = arith.ori [[EXT]], {{.*}} : i32
// CHECK:         [[FP:%.+]]  = arith.uitofp [[ORI]] : i32 to f64
// CHECK-NEXT:    return [[FP]] : f64
func.func @uitofp_i16_ext_f64(%a : i16) -> f64 {
  %ext = arith.extui %a : i16 to i64
  %c = arith.constant 1 : i64
  %or = arith.ori %ext, %c : i64
  %r = arith.uitofp %or : i64 to f64
  return %r : f64
}
