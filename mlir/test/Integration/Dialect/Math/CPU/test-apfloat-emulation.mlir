// REQUIRES: system-linux || system-darwin
// TODO: Run only on Linux and MacOS until we figure out how to build
// mlir_apfloat_wrappers in a platform-independent way.

// RUN: mlir-opt %s --convert-math-to-apfloat --convert-to-llvm  | \
// RUN: mlir-runner -e entry --entry-point-result=void \
// RUN:             --shared-libs=%mlir_c_runner_utils \
// RUN:             --shared-libs=%mlir_apfloat_wrappers | FileCheck %s

func.func @entry() {

  // FP8

  %neg14fp8 = arith.constant -1.4 : f8E4M3FN
  %absfp8 = math.absf %neg14fp8 : f8E4M3FN
  // CHECK: 1.375
  vector.print %absfp8 : f8E4M3FN

  // see llvm/unittests/ADT/APFloatTest::TEST(APFloatTest, Float8E8M0FNUFMA)
  %twof8E8M0FNU = arith.constant 2.0 : f8E8M0FNU
  %fourf8E8M0FNU = arith.constant 4.0 : f8E8M0FNU
  %eightf8E8M0FNU = arith.constant 8.0 : f8E8M0FNU
  %fmafp8 = math.fma %fourf8E8M0FNU, %twof8E8M0FNU, %eightf8E8M0FNU : f8E8M0FNU
  // CHECK: 16
  vector.print %fmafp8 : f8E8M0FNU

  %isinffp8 = math.isinf %neg14fp8 : f8E4M3FN
  // CHECK: 0
  vector.print %isinffp8 : i1

  %isnanfp8 = math.isnan %neg14fp8 : f8E4M3FN
  // CHECK: 0
  vector.print %isnanfp8 : i1

  %isnormalfp8 = math.isnormal %neg14fp8 : f8E4M3FN
  // CHECK: 1
  vector.print %isnormalfp8 : i1

  %isfinitefp8 = math.isfinite %neg14fp8 : f8E4M3FN
  // CHECK: 1
  vector.print %isfinitefp8 : i1
  
  // FP32
  
  %neg14fp32 = arith.constant -1.4 : f32
  %absfp32 = math.absf %neg14fp32 : f32
  // CHECK: 1.4
  vector.print %absfp32 : f32

  %twofp32 = arith.constant 2.0 : f32
  %fourfp32 = arith.constant 4.0 : f32
  %eightfp32 = arith.constant 8.0 : f32
  %fmafp32 = math.fma %fourfp32, %twofp32, %eightfp32 : f32
  // CHECK: 16
  vector.print %fmafp32 : f32

  %isinffp32 = math.isinf %neg14fp32 : f32
  // CHECK: 0
  vector.print %isinffp32 : i1

  %isnanfp32 = math.isnan %neg14fp32 : f32
  // CHECK: 0
  vector.print %isnanfp32 : i1

  %isnormalfp32 = math.isnormal %neg14fp32 : f32
  // CHECK: 1
  vector.print %isnormalfp32 : i1

  %isfinitefp32 = math.isfinite %neg14fp32 : f32
  // CHECK: 1
  vector.print %isfinitefp32 : i1

  return
}
