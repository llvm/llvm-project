// RUN: mlir-opt %s -convert-vector-to-llvm="enable-arm-sme" \
// RUN:           -allocate-arm-sme-tiles -canonicalize      \
// RUN:           -allow-unregistered-dialect                \
// RUN: | FileCheck %s

// This test verifies the tile mask operand of the zero intrinsic zeroes
// the correct tiles. Both integer and floating-point datatypes are checked.

// -----

// CHECK-LABEL: zero_za_b
func.func @zero_za_b() {
  // CHECK-DAG: %[[TILE_ID:.*]] = arith.constant 0 : i8
  // CHECK-DAG: %[[ZERO_MASK:.*]] = arith.constant 255 : i32

  // CHECK:      "arm_sme.intr.zero"(%[[ZERO_MASK]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA0B:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID]] : i8 to vector<[16]x[16]xi8>
  %zero_za0b = arm_sme.zero : vector<[16]x[16]xi8>
  "prevent.dce"(%zero_za0b) : (vector<[16]x[16]xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: zero_za_h
func.func @zero_za_h() {
  // CHECK-DAG: %[[TILE_ID_ZA0H:.*]] = arith.constant 0 : i16
  // CHECK-DAG: %[[TILE_ID_ZA1H:.*]] = arith.constant 1 : i16

  // CHECK-DAG: %[[ZERO_MASK_ZA0H:.*]] = arith.constant 85 : i32
  // CHECK-DAG: %[[ZERO_MASK_ZA1H:.*]] = arith.constant 170 : i32

  // CHECK:      "arm_sme.intr.zero"(%[[ZERO_MASK_ZA0H]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA0H:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID_ZA0H]] : i16 to vector<[8]x[8]xi16>
  %zero_za0h = arm_sme.zero : vector<[8]x[8]xi16>
  "prevent.dce"(%zero_za0h) : (vector<[8]x[8]xi16>) -> ()
  // CHECK:      "arm_sme.intr.zero"(%[[ZERO_MASK_ZA1H]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA1H:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID_ZA1H]] : i16 to vector<[8]x[8]xf16>
  %zero_za1h = arm_sme.zero : vector<[8]x[8]xf16>
  "prevent.dce"(%zero_za1h) : (vector<[8]x[8]xf16>) -> ()
  return
}

// -----

// CHECK-LABEL: zero_za_s
func.func @zero_za_s() {
  // CHECK-DAG: %[[TILE_ID_ZA0S:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[TILE_ID_ZA1S:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[TILE_ID_ZA2S:.*]] = arith.constant 2 : i32
  // CHECK-DAG: %[[TILE_ID_ZA3S:.*]] = arith.constant 3 : i32

  // CHECK-DAG: %[[ZERO_MASK_ZA0S:.*]] = arith.constant 17 : i32
  // CHECK-DAG: %[[ZERO_MASK_ZA1S:.*]] = arith.constant 34 : i32
  // CHECK-DAG: %[[ZERO_MASK_ZA2S:.*]] = arith.constant 68 : i32
  // CHECK-DAG: %[[ZERO_MASK_ZA3S:.*]] = arith.constant 136 : i32

  // CHECK:      "arm_sme.intr.zero"(%[[ZERO_MASK_ZA0S]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA0S:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID_ZA0S]] : i32 to vector<[4]x[4]xi32>
  %zero_za0s = arm_sme.zero : vector<[4]x[4]xi32>
  "prevent.dce"(%zero_za0s) : (vector<[4]x[4]xi32>) -> ()
  // CHECK:      "arm_sme.intr.zero"(%[[ZERO_MASK_ZA1S]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA1S:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID_ZA1S]] : i32 to vector<[4]x[4]xi32>
  %zero_za1s = arm_sme.zero : vector<[4]x[4]xi32>
  "prevent.dce"(%zero_za1s) : (vector<[4]x[4]xi32>) -> ()
  // CHECK:      "arm_sme.intr.zero"(%[[ZERO_MASK_ZA2S]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA2S:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID_ZA2S]] : i32 to vector<[4]x[4]xi32>
  %zero_za2s = arm_sme.zero : vector<[4]x[4]xi32>
  "prevent.dce"(%zero_za2s) : (vector<[4]x[4]xi32>) -> ()
  // CHECK:     "arm_sme.intr.zero"(%[[ZERO_MASK_ZA3S]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA3S:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID_ZA3S]] : i32 to vector<[4]x[4]xf32>
  %zero_za3s = arm_sme.zero : vector<[4]x[4]xf32>
  "prevent.dce"(%zero_za3s) : (vector<[4]x[4]xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: zero_za_d
func.func @zero_za_d() {
  // CHECK-DAG: %[[TILE_ID_ZA0D:.*]] = arith.constant 0 : i64
  // CHECK-DAG: %[[TILE_ID_ZA1D:.*]] = arith.constant 1 : i64
  // CHECK-DAG: %[[TILE_ID_ZA2D:.*]] = arith.constant 2 : i64
  // CHECK-DAG: %[[TILE_ID_ZA3D:.*]] = arith.constant 3 : i64
  // CHECK-DAG: %[[TILE_ID_ZA4D:.*]] = arith.constant 4 : i64
  // CHECK-DAG: %[[TILE_ID_ZA5D:.*]] = arith.constant 5 : i64
  // CHECK-DAG: %[[TILE_ID_ZA6D:.*]] = arith.constant 6 : i64
  // CHECK-DAG: %[[TILE_ID_ZA7D:.*]] = arith.constant 7 : i64

  // CHECK-DAG: %[[ZERO_MASK_ZA0D:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[ZERO_MASK_ZA1D:.*]] = arith.constant 2 : i32
  // CHECK-DAG: %[[ZERO_MASK_ZA2D:.*]] = arith.constant 4 : i32
  // CHECK-DAG: %[[ZERO_MASK_ZA3D:.*]] = arith.constant 8 : i32
  // CHECK-DAG: %[[ZERO_MASK_ZA4D:.*]] = arith.constant 16 : i32
  // CHECK-DAG: %[[ZERO_MASK_ZA5D:.*]] = arith.constant 32 : i32
  // CHECK-DAG: %[[ZERO_MASK_ZA6D:.*]] = arith.constant 64 : i32
  // CHECK-DAG: %[[ZERO_MASK_ZA7D:.*]] = arith.constant 128 : i32

  // CHECK:      "arm_sme.intr.zero"(%[[ZERO_MASK_ZA0D]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA0D:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID_ZA0D]] : i64 to vector<[2]x[2]xi64>
  %zero_za0d = arm_sme.zero : vector<[2]x[2]xi64>
  "prevent.dce"(%zero_za0d) : (vector<[2]x[2]xi64>) -> ()
  // CHECK:     "arm_sme.intr.zero"(%[[ZERO_MASK_ZA1D]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA1D:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID_ZA1D]] : i64 to vector<[2]x[2]xi64>
  %zero_za1d = arm_sme.zero : vector<[2]x[2]xi64>
  "prevent.dce"(%zero_za1d) : (vector<[2]x[2]xi64>) -> ()
  // CHECK:      "arm_sme.intr.zero"(%[[ZERO_MASK_ZA2D]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA2D:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID_ZA2D]] : i64 to vector<[2]x[2]xi64>
  %zero_za2d = arm_sme.zero : vector<[2]x[2]xi64>
  "prevent.dce"(%zero_za2d) : (vector<[2]x[2]xi64>) -> ()
  // CHECK:     "arm_sme.intr.zero"(%[[ZERO_MASK_ZA3D]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA3D:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID_ZA3D]] : i64 to vector<[2]x[2]xi64>
  %zero_za3d = arm_sme.zero : vector<[2]x[2]xi64>
  "prevent.dce"(%zero_za3d) : (vector<[2]x[2]xi64>) -> ()
  // CHECK:     "arm_sme.intr.zero"(%[[ZERO_MASK_ZA4D]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA4D:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID_ZA4D]] : i64 to vector<[2]x[2]xi64>
  %zero_za4d = arm_sme.zero : vector<[2]x[2]xi64>
  "prevent.dce"(%zero_za4d) : (vector<[2]x[2]xi64>) -> ()
  // CHECK:     "arm_sme.intr.zero"(%[[ZERO_MASK_ZA5D]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA5D:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID_ZA5D]] : i64 to vector<[2]x[2]xi64>
  %zero_za5d = arm_sme.zero : vector<[2]x[2]xi64>
  "prevent.dce"(%zero_za5d) : (vector<[2]x[2]xi64>) -> ()
  // CHECK:     "arm_sme.intr.zero"(%[[ZERO_MASK_ZA6D]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA6D:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID_ZA6D]] : i64 to vector<[2]x[2]xi64>
  %zero_za6d = arm_sme.zero : vector<[2]x[2]xi64>
  "prevent.dce"(%zero_za6d) : (vector<[2]x[2]xi64>) -> ()
  // CHECK:     "arm_sme.intr.zero"(%[[ZERO_MASK_ZA7D]]) : (i32) -> ()
  // CHECK-NEXT: %[[ZERO_ZA7D:.*]] = arm_sme.cast_tile_to_vector %[[TILE_ID_ZA7D]] : i64 to vector<[2]x[2]xf64>
  %zero_za7d = arm_sme.zero : vector<[2]x[2]xf64>
  "prevent.dce"(%zero_za7d) : (vector<[2]x[2]xf64>) -> ()
  return
}
