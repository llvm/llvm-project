// REQUIRES: system-linux || system-darwin
// TODO: Run only on Linux until we figure out how to build
// mlir_apfloat_wrappers in a platform-independent way.

// Case 1: All floating-point arithmetics is lowered through APFloat.
// RUN: mlir-opt %s --convert-math-to-apfloat --convert-to-llvm | \
// RUN: mlir-runner -e entryfp8 --entry-point-result=void \
// RUN:             --shared-libs=%mlir_c_runner_utils \
// RUN:             --shared-libs=%mlir_apfloat_wrappers | FileCheck %s --check-prefix=CHECK-FP8

// Case 2: Only unsupported arithmetics is lowered through APFloat.
//         Arithmetics on f32 is lowered directly to LLVM.
// RUN: mlir-opt %s --convert-to-llvm --convert-math-to-apfloat \
// RUN:          --convert-to-llvm --reconcile-unrealized-casts | \
// RUN: mlir-runner -e entryfp32 --entry-point-result=void \
// RUN:             --shared-libs=%mlir_c_runner_utils \
// RUN:             --shared-libs=%mlir_apfloat_wrappers | FileCheck %s --check-prefix=CHECK-FP32

func.func @entryfp8() {
  %neg14fp8 = arith.constant -1.4 : f8E4M3FN
  %abs = math.absf %neg14fp8 : f8E4M3FN
  // CHECK-FP8: 1.375
  vector.print %abs : f8E4M3FN

  // see llvm/unittests/ADT/APFloatTest::TEST(APFloatTest, Float8E8M0FNUFMA)
  %twof8E8M0FNU = arith.constant 2.0 : f8E8M0FNU
  %fourf8E8M0FNU = arith.constant 4.0 : f8E8M0FNU
  %eightf8E8M0FNU = arith.constant 8.0 : f8E8M0FNU
  %fma = math.fma %fourf8E8M0FNU, %twof8E8M0FNU, %eightf8E8M0FNU : f8E8M0FNU
  // CHECK-FP8: 16
  vector.print %fma : f8E8M0FNU

  // CHECK-FP8: 0
  %isinf = math.isinf %neg14fp8 : f8E4M3FN
  vector.print %isinf : i1
  // CHECK-FP8: 0
  %isnan = math.isnan %neg14fp8 : f8E4M3FN
  vector.print %isnan : i1
  %isnormal = math.isnormal %neg14fp8 : f8E4M3FN
  // CHECK-FP8: 1
  vector.print %isnormal : i1
  %isfinite = math.isfinite %neg14fp8 : f8E4M3FN
  // CHECK-FP8: 1
  vector.print %isfinite : i1

  return
}

func.func @entryfp32() {
  %neg14 = arith.constant -1.4 : f32
  %abs = math.absf %neg14 : f32
  // CHECK-FP32: 1.4
  vector.print %abs : f32

  %two = arith.constant 2.0 : f32
  %four = arith.constant 4.0 : f32
  %eight = arith.constant 8.0 : f32
  %fma = math.fma %four, %two, %eight : f32
  // CHECK-FP32: 16
  vector.print %fma : f32

  // CHECK-FP32: 0
  %isinf = math.isinf %neg14 : f32
  vector.print %isinf : i1
  // CHECK-FP32: 0
  %isnan = math.isnan %neg14 : f32
  vector.print %isnan : i1
  %isnormal = math.isnormal %neg14 : f32
  // CHECK-FP32: 1
  vector.print %isnormal : i1
  %isfinite = math.isfinite %neg14 : f32
  // CHECK-FP32: 1
  vector.print %isfinite : i1

  return
}
