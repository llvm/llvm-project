// RUN: mlir-opt %s --convert-arith-to-apfloat -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:   func.func private @_mlir_apfloat_add(i32, i64, i64) -> i64

// CHECK-LABEL:   func.func @foo() -> f8E4M3FN {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 2.250000e+00 : f8E4M3FN
// CHECK:           return %[[CONSTANT_0]] : f8E4M3FN
// CHECK:         }

// CHECK-LABEL:   func.func @bar() -> f6E3M2FN {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 3.000000e+00 : f6E3M2FN
// CHECK:           return %[[CONSTANT_0]] : f6E3M2FN
// CHECK:         }

// Illustrate that both f8E4M3FN and f6E3M2FN calling the same _mlir_apfloat_add is fine
// because each gets its own semantics enum and gets bitcast/extui/trunci to its own width.
// CHECK-LABEL:   func.func @full_example() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.375000e+00 : f8E4M3FN
// CHECK:           %[[VAL_0:.*]] = call @foo() : () -> f8E4M3FN
// CHECK:           %[[BITCAST_0:.*]] = arith.bitcast %[[CONSTANT_0]] : f8E4M3FN to i8
// CHECK:           %[[EXTUI_0:.*]] = arith.extui %[[BITCAST_0]] : i8 to i64
// CHECK:           %[[BITCAST_1:.*]] = arith.bitcast %[[VAL_0]] : f8E4M3FN to i8
// CHECK:           %[[EXTUI_1:.*]] = arith.extui %[[BITCAST_1]] : i8 to i64
//                  // fltSemantics semantics for f8E4M3FN
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_1:.*]] = call @_mlir_apfloat_add(%[[CONSTANT_1]], %[[EXTUI_0]], %[[EXTUI_1]]) : (i32, i64, i64) -> i64
// CHECK:           %[[TRUNCI_0:.*]] = arith.trunci %[[VAL_1]] : i64 to i8
// CHECK:           %[[BITCAST_2:.*]] = arith.bitcast %[[TRUNCI_0]] : i8 to f8E4M3FN
// CHECK:           vector.print %[[BITCAST_2]] : f8E4M3FN

// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2.500000e+00 : f6E3M2FN
// CHECK:           %[[VAL_2:.*]] = call @bar() : () -> f6E3M2FN
// CHECK:           %[[BITCAST_3:.*]] = arith.bitcast %[[CONSTANT_2]] : f6E3M2FN to i6
// CHECK:           %[[EXTUI_2:.*]] = arith.extui %[[BITCAST_3]] : i6 to i64
// CHECK:           %[[BITCAST_4:.*]] = arith.bitcast %[[VAL_2]] : f6E3M2FN to i6
// CHECK:           %[[EXTUI_3:.*]] = arith.extui %[[BITCAST_4]] : i6 to i64
//                  // fltSemantics semantics for f6E3M2FN
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 16 : i32
// CHECK:           %[[VAL_3:.*]] = call @_mlir_apfloat_add(%[[CONSTANT_3]], %[[EXTUI_2]], %[[EXTUI_3]]) : (i32, i64, i64) -> i64
// CHECK:           %[[TRUNCI_1:.*]] = arith.trunci %[[VAL_3]] : i64 to i6
// CHECK:           %[[BITCAST_5:.*]] = arith.bitcast %[[TRUNCI_1]] : i6 to f6E3M2FN
// CHECK:           vector.print %[[BITCAST_5]] : f6E3M2FN
// CHECK:           return
// CHECK:         }

// Put rhs into separate function so that it won't be constant-folded.
func.func @foo() -> f8E4M3FN {
  %cst = arith.constant 2.2 : f8E4M3FN
  return %cst : f8E4M3FN
}

func.func @bar() -> f6E3M2FN {
  %cst = arith.constant 3.2 : f6E3M2FN
  return %cst : f6E3M2FN
}

func.func @full_example() {
  %a = arith.constant 1.4 : f8E4M3FN
  %b = func.call @foo() : () -> (f8E4M3FN)
  %c = arith.addf %a, %b : f8E4M3FN
  vector.print %c : f8E4M3FN

  %d = arith.constant 2.4 : f6E3M2FN
  %e = func.call @bar() : () -> (f6E3M2FN)
  %f = arith.addf %d, %e : f6E3M2FN
  vector.print %f : f6E3M2FN
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

// Test decl collision (different type)
// expected-error@+1{{matched function '_mlir_apfloat_add' but with different type: '(i32, i32, f32) -> index' (expected '(i32, i64, i64) -> i64')}}
func.func private @_mlir_apfloat_add(i32, i32, f32) -> index
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
