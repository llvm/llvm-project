// RUN: mlir-opt %s --convert-arith-to-apfloat -split-input-file | FileCheck %s

// CHECK-LABEL:   func.func private @_mlir_apfloat_add(i32, i64, i64) -> i64

// CHECK-LABEL:   func.func @foo() -> f8E4M3FN {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 2.250000e+00 : f8E4M3FN
// CHECK:           return %[[CONSTANT_0]] : f8E4M3FN
// CHECK:         }

// CHECK-LABEL:   func.func @full_example() {
// CHECK:           %[[cst:.*]] = arith.constant 1.375000e+00 : f8E4M3FN
// CHECK:           %[[rhs:.*]] = call @foo() : () -> f8E4M3FN
// CHECK:           %[[lhs_casted:.*]] = arith.bitcast %[[cst]] : f8E4M3FN to i8
// CHECK:           %[[lhs_ext:.*]] = arith.extui %[[lhs_casted]] : i8 to i64
// CHECK:           %[[rhs_casted:.*]] = arith.bitcast %[[rhs]] : f8E4M3FN to i8
// CHECK:           %[[rhs_ext:.*]] = arith.extui %[[rhs_casted]] : i8 to i64
// CHECK:           %[[c10_i32:.*]] = arith.constant 10 : i32
// CHECK:           %[[res:.*]] = call @_mlir_apfloat_add(%[[c10_i32]], %[[lhs_ext]], %[[rhs_ext]]) : (i32, i64, i64) -> i64
// CHECK:           %[[res_trunc:.*]] = arith.trunci %[[res]] : i64 to i8
// CHECK:           %[[res_casted:.*]] = arith.bitcast %[[res_trunc]] : i8 to f8E4M3FN
// CHECK:           vector.print %[[res_casted]] : f8E4M3FN
// CHECK:           return
// CHECK:         }

// Put rhs into separate function so that it won't be constant-folded.
func.func @foo() -> f8E4M3FN {
  %cst = arith.constant 2.2 : f8E4M3FN
  return %cst : f8E4M3FN
}

func.func @full_example() {
  %a = arith.constant 1.4 : f8E4M3FN
  %b = func.call @foo() : () -> (f8E4M3FN)
  %c = arith.addf %a, %b : f8E4M3FN

  vector.print %c : f8E4M3FN
  return
}

// -----

// CHECK: func.func private @_mlir_apfloat_add(i32, i64, i64) -> i64
// CHECK: %[[sem:.*]] = arith.constant 18 : i32
// CHECK: call @_mlir_apfloat_add(%[[sem]], %{{.*}}, %{{.*}}) : (i32, i64, i64) -> i64
func.func @addf(%arg0: f4E2M1FN, %arg1: f4E2M1FN) {
  %0 = arith.addf %arg0, %arg1 : f4E2M1FN
  return
}

// -----

// CHECK: func.func private @_mlir_apfloat_subtract(i32, i64, i64) -> i64
// CHECK: %[[sem:.*]] = arith.constant 18 : i32
// CHECK: call @_mlir_apfloat_subtract(%[[sem]], %{{.*}}, %{{.*}}) : (i32, i64, i64) -> i64
func.func @subf(%arg0: f4E2M1FN, %arg1: f4E2M1FN) {
  %0 = arith.subf %arg0, %arg1 : f4E2M1FN
  return
}

// -----

// CHECK: func.func private @_mlir_apfloat_multiply(i32, i64, i64) -> i64
// CHECK: %[[sem:.*]] = arith.constant 18 : i32
// CHECK: call @_mlir_apfloat_multiply(%[[sem]], %{{.*}}, %{{.*}}) : (i32, i64, i64) -> i64
func.func @subf(%arg0: f4E2M1FN, %arg1: f4E2M1FN) {
  %0 = arith.mulf %arg0, %arg1 : f4E2M1FN
  return
}

// -----

// CHECK: func.func private @_mlir_apfloat_divide(i32, i64, i64) -> i64
// CHECK: %[[sem:.*]] = arith.constant 18 : i32
// CHECK: call @_mlir_apfloat_divide(%[[sem]], %{{.*}}, %{{.*}}) : (i32, i64, i64) -> i64
func.func @subf(%arg0: f4E2M1FN, %arg1: f4E2M1FN) {
  %0 = arith.divf %arg0, %arg1 : f4E2M1FN
  return
}

// -----

// CHECK: func.func private @_mlir_apfloat_remainder(i32, i64, i64) -> i64
// CHECK: %[[sem:.*]] = arith.constant 18 : i32
// CHECK: call @_mlir_apfloat_remainder(%[[sem]], %{{.*}}, %{{.*}}) : (i32, i64, i64) -> i64
func.func @remf(%arg0: f4E2M1FN, %arg1: f4E2M1FN) {
  %0 = arith.remf %arg0, %arg1 : f4E2M1FN
  return
}
