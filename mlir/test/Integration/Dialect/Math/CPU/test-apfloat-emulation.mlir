// REQUIRES: system-linux || system-darwin
// TODO: Run only on Linux until we figure out how to build
// mlir_apfloat_wrappers in a platform-independent way.

// Case 1: All floating-point arithmetics is lowered through APFloat.
// RUN: mlir-opt %s --convert-math-to-apfloat --convert-to-llvm | \
// RUN: mlir-runner -e entry --entry-point-result=void \
// RUN:             --shared-libs=%mlir_c_runner_utils \
// RUN:             --shared-libs=%mlir_apfloat_wrappers | FileCheck %s

// Case 2: Only unsupported arithmetics (f8E4M3FN) is lowered through APFloat.
//         Arithmetics on f32 is lowered directly to LLVM.
// RUN: mlir-opt %s --convert-to-llvm --convert-math-to-apfloat \
// RUN:          --convert-to-llvm --reconcile-unrealized-casts | \
// RUN: mlir-runner -e entry --entry-point-result=void \
// RUN:             --shared-libs=%mlir_c_runner_utils \
// RUN:             --shared-libs=%mlir_apfloat_wrappers | FileCheck %s

func.func @entry() {
  %neg14fp8 = arith.constant -1.4 : f8E4M3FN
  %neg14fp32 = arith.constant 1.4 : f32

  // CHECK: 1.375
  %c2 = math.absf %neg14fp8 : f8E4M3FN
  vector.print %c2 : f8E4M3FN

  // CHECK: 1.4
  %c3 = math.absf %neg14fp32 : f32
  vector.print %c3 : f32

  // see llvm/unittests/ADT/APFloatTest::TEST(APFloatTest, Float8E8M0FNUFMA)
  %twof8E8M0FNU = arith.constant 2.0 : f8E8M0FNU
  %fourf8E8M0FNU = arith.constant 4.0 : f8E8M0FNU
  %eightf8E8M0FNU = arith.constant 8.0 : f8E8M0FNU

  // CHECK: 16
  %c4 = math.fma %fourf8E8M0FNU, %twof8E8M0FNU, %eightf8E8M0FNU : f8E8M0FNU
  // vector.print %c4 : f8E8M0FNU

  return
}
