// REQUIRES: system-linux
// TODO: Run only on Linux until we figure out how to build
// mlir_apfloat_wrappers in a platform-independent way.

// Case 1: All floating-point arithmetics is lowered through APFloat.
// RUN: mlir-opt %s --convert-arith-to-apfloat --convert-to-llvm | \
// RUN: mlir-runner -e entry --entry-point-result=void \
// RUN:             --shared-libs=%mlir_c_runner_utils \
// RUN:             --shared-libs=%mlir_apfloat_wrappers | FileCheck %s

// Case 2: Only unsupported arithmetics (f8E4M3FN) is lowered through APFloat.
//         Arithmetics on f32 is lowered directly to LLVM.
// RUN: mlir-opt %s --convert-to-llvm --convert-arith-to-apfloat \
// RUN:          --convert-to-llvm --reconcile-unrealized-casts | \
// RUN: mlir-runner -e entry --entry-point-result=void \
// RUN:             --shared-libs=%mlir_c_runner_utils \
// RUN:             --shared-libs=%mlir_apfloat_wrappers | FileCheck %s

// Put rhs into separate function so that it won't be constant-folded.
func.func @foo() -> (f8E4M3FN, f32) {
  %cst1 = arith.constant 2.2 : f8E4M3FN
  %cst2 = arith.constant 2.2 : f32
  return %cst1, %cst2 : f8E4M3FN, f32
}

func.func @entry() {
  %a1 = arith.constant 1.4 : f8E4M3FN
  %a2 = arith.constant 1.4 : f32
  %b1, %b2 = func.call @foo() : () -> (f8E4M3FN, f32)

  // CHECK: 2.2
  vector.print %b2 : f32

  // CHECK-NEXT: 3.5
  %c1 = arith.addf %a1, %b1 : f8E4M3FN  // not supported by LLVM
  vector.print %c1 : f8E4M3FN

  // CHECK-NEXT: 3.6
  %c2 = arith.addf %a2, %b2 : f32       // supported by LLVM
  vector.print %c2 : f32

  // CHECK-NEXT: 2.25
  %cvt = arith.truncf %b2 : f32 to f8E4M3FN
  vector.print %cvt : f8E4M3FN

  return
}
