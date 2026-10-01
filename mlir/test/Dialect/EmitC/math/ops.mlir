// RUN: mlir-opt %s -math-expand-ops=ops=rsqrt -convert-math-to-emitc \
// RUN: -convert-arith-to-emitc | FileCheck %s

/// This file checks the conversion of math ops whose EmitC lowering requires
/// expansion across multiple dialects, s.a. arith.
/// The FileCheck coverage is intentionally minimal, since the full MathToEmitC
/// lowering is already covered in
/// `test/Conversion/MathToEmitC/math-to-emitc.mlir`, their expansion in
/// `test/Dialect/Math/expand-math.mlir`, but not the combination of the two.

/// Vector cases excluded: `math.rsqrt` expands through `arith.constant` to
/// materialize the numerator `1.0`, and ArithToEmitC does not convert
/// `VectorType` to an EmitC type.

/// Tensor cases excluded: `math.rsqrt` expands through `arith.divf`, and the
/// resulting `emitc.div` does not accept tensor operands.

// CHECK-LABEL:   func.func @rsqrt32
// CHECK-SAME:      %[[SRC:.*]]: f32) -> f32
func.func @rsqrt32(%float: f32) -> (f32)  {
// CHECK-NOT:       math.sqrt
// CHECK:           %[[CONST:.*]] = "emitc.constant"() <{value = 1.000000e+00 : f32}>
// CHECK:           %[[SQRT:.*]] = emitc.call_opaque "sqrtf"(%[[SRC]])
// CHECK:           %[[DIV:.*]] = emitc.div %[[CONST]], %[[SQRT]]
  %float_result = math.rsqrt %float : f32
// CHECK:           return %[[DIV]] : f32
  return %float_result : f32
}

// CHECK-LABEL:   func.func @rsqrt64
// CHECK-SAME:      %[[SRC:.*]]: f64) -> f64
func.func @rsqrt64(%float: f64) -> (f64)  {
// CHECK-NOT:       math.sqrt
// CHECK:           %[[CONST:.*]] = "emitc.constant"() <{value = 1.000000e+00 : f64}>
// CHECK:           %[[SQRT:.*]] = emitc.call_opaque "sqrt"(%[[SRC]])
// CHECK:           %[[DIV:.*]] = emitc.div %[[CONST]], %[[SQRT]]
  %float_result = math.rsqrt %float : f64
// CHECK:           return %[[DIV]] : f64
  return %float_result : f64
}

/// `math.sqrt` is only lowered for f32/f64.
// CHECK-LABEL:   func.func @negative_rsqrt16
func.func @negative_rsqrt16(%float: f16) -> (f16)  {
// CHECK:           math.sqrt
  %float_result = math.rsqrt %float : f16
  return %float_result : f16
}
