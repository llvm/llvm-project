// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-arith-to-llvm))" \
// RUN:   -mlir-print-op-generic %s | FileCheck %s

// Check that inherent attributes are converted to target properties while
// discardable attributes remain in the attribute dictionary.

func.func @attribute_storage(%f0: f32, %f1: f32, %i0: i32, %i1: i32) {
  // CHECK: "llvm.fadd"
  // CHECK-SAME: <{fastmathFlags = #llvm.fastmath<fast>}>
  // CHECK-SAME: {test.discardable = 0 : i64}
  %0 = arith.addf %f0, %f1 fastmath<fast>
      {"test.discardable" = 0 : i64} : f32

  // CHECK: "llvm.add"
  // CHECK-SAME: <{overflowFlags = 1 : i32}>
  // CHECK-SAME: {test.discardable = 1 : i64}
  %1 = arith.addi %i0, %i1 overflow<nsw>
      {"test.discardable" = 1 : i64} : i32

  // CHECK: "llvm.zext"
  // CHECK-SAME: <{nonNeg}>
  // CHECK-SAME: {test.discardable = 2 : i64}
  %2 = arith.extui %i0 nneg {"test.discardable" = 2 : i64} : i32 to i64

  // CHECK: "llvm.intr.experimental.constrained.fadd"
  // CHECK-SAME: <{fastmathFlags = #llvm.fastmath<none>,
  // CHECK-SAME: fpExceptionBehavior = 0 : i64, roundingmode = 1 : i64}>
  // CHECK-SAME: {test.discardable = 3 : i64}
  %3 = arith.addf %f0, %f1 to_nearest_even fastmath<fast>
      {"test.discardable" = 3 : i64} : f32

  // CHECK: "llvm.fpext"
  // CHECK-SAME: <{fastmathFlags = #llvm.fastmath<nnan>}>
  %4 = arith.extf %f0 fastmath<nnan> : f32 to f64

  // CHECK: "llvm.fptrunc"
  // CHECK-SAME: <{fastmathFlags = #llvm.fastmath<nnan>}>
  %5 = arith.truncf %f0 fastmath<nnan> : f32 to f16
  return
}
