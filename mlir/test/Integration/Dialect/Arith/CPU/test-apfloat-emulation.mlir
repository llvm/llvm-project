// REQUIRES: system-linux || system-darwin
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

  // CHECK-NEXT: -2.25
  %negated = arith.negf %cvt : f8E4M3FN
  vector.print %negated : f8E4M3FN

  // CHECK-NEXT: -2.25
  %min = arith.minimumf %cvt, %negated : f8E4M3FN
  vector.print %min : f8E4M3FN

  // CHECK-NEXT: 1
  %cmp1 = arith.cmpf "olt", %cvt, %c1 : f8E4M3FN
  vector.print %cmp1 : i1

  // CHECK-NEXT: 1
  // Bit pattern: 01, interpreted as signed integer: 1
  %cvt_int_signed = arith.fptosi %cvt : f8E4M3FN to i2
  vector.print %cvt_int_signed : i2

  // CHECK-NEXT: -2
  // Bit pattern: 10, interpreted as signed integer: -2
  %cvt_int_unsigned = arith.fptoui %cvt : f8E4M3FN to i2
  vector.print %cvt_int_unsigned : i2

  // CHECK-NEXT: -6
  // Bit pattern: 1...11110111, interpreted as signed: -9
  // Closest f4E2M1FN value: -6.0
  %c9 = arith.constant -9 : i16
  %cvt_from_signed_int = arith.sitofp %c9 : i16 to f4E2M1FN
  vector.print %cvt_from_signed_int : f4E2M1FN

  // CHECK-NEXT: 6
  // Bit pattern: 1...11110111, interpreted as unsigned: 65527
  // Closest f4E2M1FN value: 6.0
  %cvt_from_unsigned_int = arith.uitofp %c9 : i16 to f4E2M1FN
  vector.print %cvt_from_unsigned_int : f4E2M1FN

  return
}
