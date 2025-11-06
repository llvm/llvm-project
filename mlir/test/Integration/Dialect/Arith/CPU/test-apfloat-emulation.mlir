// Check that the ceildivsi lowering is correct.
// We do not check any poison or UB values, as it is not possible to catch them.

// RUN: mlir-opt %s --convert-arith-to-apfloat

// Put rhs into separate function so that it won't be constant-folded.
func.func @foo() -> f4E2M1FN {
  %cst = arith.constant 5.0 : f4E2M1FN
  return %cst : f4E2M1FN
}

func.func @entry() {
  %a = arith.constant 5.0 : f4E2M1FN
  %b = func.call @foo() : () -> (f4E2M1FN)
  %c = arith.addf %a, %b : f4E2M1FN
  vector.print %c : f4E2M1FN
  return
}

// CHECK-LABEL:   func.func private @_mlir_apfloat_add(i32, i64, i64) -> i64

// CHECK-LABEL:   func.func @foo() -> f4E2M1FN {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 4.000000e+00 : f4E2M1FN
// CHECK:           return %[[CONSTANT_0]] : f4E2M1FN
// CHECK:         }

// CHECK-LABEL:   func.func @entry() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 18 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 6 : i64
// CHECK:           %[[VAL_0:.*]] = call @foo() : () -> f4E2M1FN
// CHECK:           %[[BITCAST_0:.*]] = arith.bitcast %[[VAL_0]] : f4E2M1FN to i4
// CHECK:           %[[EXTUI_0:.*]] = arith.extui %[[BITCAST_0]] : i4 to i64
// CHECK:           %[[VAL_1:.*]] = call @_mlir_apfloat_add(%[[CONSTANT_0]], %[[EXTUI_0]], %[[CONSTANT_1]]) : (i32, i64, i64) -> i64
// CHECK:           %[[TRUNCI_0:.*]] = arith.trunci %[[VAL_1]] : i64 to i4
// CHECK:           %[[BITCAST_1:.*]] = arith.bitcast %[[TRUNCI_0]] : i4 to f4E2M1FN
// CHECK:           vector.print %[[BITCAST_1]] : f4E2M1FN
// CHECK:           return
// CHECK:         }