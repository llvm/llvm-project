// RUN: mlir-opt %s -allocate-arm-sme-tiles -convert-arm-sme-to-llvm -canonicalize | FileCheck %s

// This test verifies the tile mask operand of the zero intrinsic zeroes
// the correct tiles. Both integer and floating-point datatypes are checked.

// -----

// CHECK-LABEL: zero_za_b
func.func @zero_za_b() {
  // CHECK: "arm_sme.intr.zero"() <{tile_mask = 255 : i32}> : () -> ()
  %zero_za0b = arm_sme.zero : vector<[16]x[16]xi8>
  return
}

// -----

// CHECK-LABEL: zero_za_h
func.func @zero_za_h() {
  // CHECK:      "arm_sme.intr.zero"() <{tile_mask = 85 : i32}> : () -> ()
  %zero_za0h = arm_sme.zero : vector<[8]x[8]xi16>
  // CHECK-NEXT: "arm_sme.intr.zero"() <{tile_mask = 170 : i32}> : () -> ()
  %zero_za1h = arm_sme.zero : vector<[8]x[8]xf16>
  return
}

// -----

// CHECK-LABEL: zero_za_s
func.func @zero_za_s() {
  // CHECK:      arm_sme.intr.zero"() <{tile_mask = 17 : i32}> : () -> ()
  %zero_za0s = arm_sme.zero : vector<[4]x[4]xi32>
  // CHECK-NEXT: arm_sme.intr.zero"() <{tile_mask = 34 : i32}> : () -> ()
  %zero_za1s = arm_sme.zero : vector<[4]x[4]xi32>
  // CHECK-NEXT: arm_sme.intr.zero"() <{tile_mask = 68 : i32}> : () -> ()
  %zero_za2s = arm_sme.zero : vector<[4]x[4]xi32>
  // CHECK-NEXT: arm_sme.intr.zero"() <{tile_mask = 136 : i32}> : () -> ()
  %zero_za3s = arm_sme.zero : vector<[4]x[4]xf32>
  return
}

// -----

// CHECK-LABEL: zero_za_d
func.func @zero_za_d() {
  // CHECK:      "arm_sme.intr.zero"() <{tile_mask = 1 : i32}> : () -> ()
  %zero_za0d = arm_sme.zero : vector<[2]x[2]xi64>
  // CHECK-NEXT: "arm_sme.intr.zero"() <{tile_mask = 2 : i32}> : () -> ()
  %zero_za1d = arm_sme.zero : vector<[2]x[2]xi64>
  // CHECK-NEXT: "arm_sme.intr.zero"() <{tile_mask = 4 : i32}> : () -> ()
  %zero_za2d = arm_sme.zero : vector<[2]x[2]xi64>
  // CHECK-NEXT: "arm_sme.intr.zero"() <{tile_mask = 8 : i32}> : () -> ()
  %zero_za3d = arm_sme.zero : vector<[2]x[2]xi64>
  // CHECK-NEXT: "arm_sme.intr.zero"() <{tile_mask = 16 : i32}> : () -> ()
  %zero_za4d = arm_sme.zero : vector<[2]x[2]xi64>
  // CHECK-NEXT: "arm_sme.intr.zero"() <{tile_mask = 32 : i32}> : () -> ()
  %zero_za5d = arm_sme.zero : vector<[2]x[2]xi64>
  // CHECK-NEXT: "arm_sme.intr.zero"() <{tile_mask = 64 : i32}> : () -> ()
  %zero_za6d = arm_sme.zero : vector<[2]x[2]xi64>
  // CHECK-NEXT: "arm_sme.intr.zero"() <{tile_mask = 128 : i32}> : () -> ()
  %zero_za7d = arm_sme.zero : vector<[2]x[2]xf64>
  return
}
