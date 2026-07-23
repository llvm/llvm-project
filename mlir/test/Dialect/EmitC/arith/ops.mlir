// RUN: mlir-opt -arith-expand -convert-arith-to-emitc %s | FileCheck %s

/// This file checks the combined `-arith-expand | -convert-arith-to-emitc`
/// pipeline with intentionally minimal FileCheck coverage.
/// The full expansion is covered in `test/Dialect/Arith/expand-ops.mlir`, and
/// the full ArithToEmitC lowering is covered in
/// `test/Conversion/ArithToEmitC/arith-to-emitc.mlir`; this test only covers
/// that the combination of the two produces the expected EmitC ops.

/// `arith.maximumf`/`arith.minimumf` expand through `arith.cmpf`.
/// Vector cases excluded because ArithToEmitC cannot convert `VectorType`.
/// Tensor cases excluded because ArithToEmitC only lowers scalar `cmpf`.

// CHECK-LABEL: func @maximumf
// CHECK-SAME:  %[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> f32
func.func @maximumf(%a: f32, %b: f32) -> f32 {
// CHECK-NOT:   arith
// CHECK:       %[[CMP:.*]] = emitc.cmp gt, %[[ARG0]], %[[ARG1]] : (f32, f32) -> i1
// CHECK:       %[[OR:.*]] = emitc.logical_or %{{.*}}, %[[CMP]] : i1, i1
// CHECK:       %[[SEL0:.*]] = emitc.conditional %[[OR]], %[[ARG0]], %[[ARG1]] : f32
// CHECK:       %[[SEL1:.*]] = emitc.conditional %{{.*}}, %[[ARG1]], %[[SEL0]] : f32
  %result = arith.maximumf %a, %b : f32
// CHECK:       return %[[SEL1]] : f32
  return %result : f32
}

// CHECK-LABEL: func @minimumf
// CHECK-SAME:  %[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> f32
func.func @minimumf(%a: f32, %b: f32) -> f32 {
// CHECK-NOT:   arith
// CHECK:       %[[CMP:.*]] = emitc.cmp lt, %[[ARG0]], %[[ARG1]] : (f32, f32) -> i1
// CHECK:       %[[OR:.*]] = emitc.logical_or %{{.*}}, %[[CMP]] : i1, i1
// CHECK:       %[[SEL0:.*]] = emitc.conditional %[[OR]], %[[ARG0]], %[[ARG1]] : f32
// CHECK:       %[[SEL1:.*]] = emitc.conditional %{{.*}}, %[[ARG1]], %[[SEL0]] : f32
  %result = arith.minimumf %a, %b : f32
// CHECK:       return %[[SEL1]] : f32
  return %result : f32
}

// CHECK-LABEL: func @minsi
// CHECK-SAME:  %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
func.func @minsi(%a: i32, %b: i32) -> i32 {
// CHECK-NOT:   arith
// CHECK:       %[[CMP:.*]] = emitc.cmp lt, %[[ARG0]], %[[ARG1]] : (i32, i32) -> i1
// CHECK:       %[[SEL:.*]] = emitc.conditional %[[CMP]], %[[ARG0]], %[[ARG1]] : i32
  %result = arith.minsi %a, %b : i32
// CHECK:       return %[[SEL]] : i32
  return %result : i32
}

// CHECK-LABEL: func @maxsi
// CHECK-SAME:  %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
func.func @maxsi(%a: i32, %b: i32) -> i32 {
// CHECK-NOT:   arith
// CHECK:       %[[CMP:.*]] = emitc.cmp gt, %[[ARG0]], %[[ARG1]] : (i32, i32) -> i1
// CHECK:       %[[SEL:.*]] = emitc.conditional %[[CMP]], %[[ARG0]], %[[ARG1]] : i32
  %result = arith.maxsi %a, %b : i32
// CHECK:       return %[[SEL]] : i32
  return %result : i32
}
