// Verify that the LUT-based f8E4M3FN → f32 lowering produces the correct f32
// value for a selection of well-known bit patterns.

// RUN: mlir-opt %s \
// RUN:     --convert-arith-extf-to-lut \
// RUN:     --finalize-memref-to-llvm \
// RUN:     --convert-arith-to-llvm \
// RUN:     --convert-vector-to-llvm \
// RUN:     --convert-func-to-llvm \
// RUN:     --reconcile-unrealized-casts \
// RUN: | mlir-runner -e entry --entry-point-result=void \
// RUN:               --shared-libs=%mlir_c_runner_utils \
// RUN: | FileCheck %s --match-full-lines

func.func @check(%bits: i8) {
  %f8  = arith.bitcast %bits : i8 to f8E4M3FN
  %f32 = arith.extf %f8 : f8E4M3FN to f32
  vector.print %f32 : f32
  return
}

func.func @entry() {
  // +0.0  (bit pattern 0x00)
  %b0 = arith.constant 0 : i8
  // CHECK: 0
  func.call @check(%b0) : (i8) -> ()

  // -0.0  (bit pattern 0x80)
  %b1 = arith.constant -128 : i8
  // CHECK: -0
  func.call @check(%b1) : (i8) -> ()

  // 1.0  (bit pattern 0x38: exp=7, mant=0)
  %b2 = arith.constant 56 : i8
  // CHECK: 1
  func.call @check(%b2) : (i8) -> ()

  // -1.0  (bit pattern 0xB8)
  %b3 = arith.constant -72 : i8
  // CHECK: -1
  func.call @check(%b3) : (i8) -> ()

  // 2.0  (bit pattern 0x40: exp=8, mant=0)
  %b4 = arith.constant 64 : i8
  // CHECK: 2
  func.call @check(%b4) : (i8) -> ()

  // 0.5  (bit pattern 0x30: exp=6, mant=0)
  %b5 = arith.constant 48 : i8
  // CHECK: 0.5
  func.call @check(%b5) : (i8) -> ()

  // max finite: 448.0  (bit pattern 0x7E: exp=15, mant=0b110)
  %b6 = arith.constant 126 : i8
  // CHECK: 448
  func.call @check(%b6) : (i8) -> ()

  return
}
