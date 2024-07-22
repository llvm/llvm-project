// RUN: mlir-opt %s -test-arm-sme-tile-allocation -split-input-file | FileCheck %s

// -----

// Note: Tile IDs >= 16 are in-memory tile IDs (i.e. spills).

// CHECK-LABEL: mixed_tiles
func.func @mixed_tiles() {
  // ZA0.Q, ZA2.Q, ZA4.Q, ZA6.Q, ZA8.Q, ZA10.Q, ZA12.Q, ZA14.Q
  // CHECK-NEXT: tile_id = 0
  %za0_h = arm_sme.get_tile : vector<[8]x[8]xi16>
  // ZA1.Q, ZA5.Q, ZA9.Q, ZA13.Q
  // CHECK-NEXT: tile_id = 1
  %za1_s = arm_sme.get_tile : vector<[4]x[4]xi32>
  // ZA3.D ZA3.Q, ZA11.Q
  // CHECK-NEXT: tile_id = 3
  %za3_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  // ZA7.Q
  // CHECK-NEXT: tile_id = 7
  %za7_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // ZA15.Q is still free.
  "test.some_use"(%za0_h) : (vector<[8]x[8]xi16>) -> ()
  "test.some_use"(%za1_s) : (vector<[4]x[4]xi32>) -> ()
  "test.some_use"(%za3_d) : (vector<[2]x[2]xi64>) -> ()
  "test.some_use"(%za7_q) : (vector<[1]x[1]xi128>) -> ()
  return
}

// -----

// CHECK-LABEL: za_b
func.func @za_b() {
  // CHECK-NEXT: tile_id = 0
  %za0_b = arm_sme.get_tile : vector<[16]x[16]xi8>
  // Next tile is in-memory:
  // CHECK-NEXT: tile_id = 16
  %next_tile = arm_sme.get_tile : vector<[16]x[16]xi8>
  "test.some_use"(%za0_b) : (vector<[16]x[16]xi8>) -> ()
  "test.some_use"(%next_tile) : (vector<[16]x[16]xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: za_b_overlapping_za_q
func.func @za_b_overlapping_za_q() {
  // CHECK-NEXT: tile_id = 0
  %za0_b = arm_sme.get_tile : vector<[16]x[16]xi8>
  // Next tile is in-memory:
  // CHECK-NEXT: tile_id = 16
  %next_tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  "test.some_use"(%za0_b) : (vector<[16]x[16]xi8>) -> ()
  "test.some_use"(%next_tile) : (vector<[1]x[1]xi128>) -> ()
  return
}

// -----

// CHECK-LABEL: za_h
func.func @za_h() {
  // CHECK-NEXT: tile_id = 0
  %za0_h = arm_sme.get_tile : vector<[8]x[8]xi16>
  // CHECK-NEXT: tile_id = 1
  %za1_h = arm_sme.get_tile : vector<[8]x[8]xi16>
  // Next tile is in-memory:
  // CHECK-NEXT: tile_id = 16
  %next_tile = arm_sme.get_tile : vector<[8]x[8]xi16>
  "test.some_use"(%za0_h) : (vector<[8]x[8]xi16>) -> ()
  "test.some_use"(%za1_h) : (vector<[8]x[8]xi16>) -> ()
  "test.some_use"(%next_tile) : (vector<[8]x[8]xi16>) -> ()
  return
}

// -----

// CHECK-LABEL: za_h_overlapping_za_s
func.func @za_h_overlapping_za_s() {
  // ZA0.Q, ZA2.Q, ZA4.Q, ZA6.Q, ZA8.Q, ZA10.Q, ZA12.Q, ZA14.Q
  // CHECK-NEXT: tile_id = 0
  %za0_h = arm_sme.get_tile : vector<[8]x[8]xi16>
  // ZA1.Q, ZA5.Q, ZA9.Q, ZA13.Q
  // CHECK-NEXT: tile_id = 1
  %za1_s = arm_sme.get_tile : vector<[4]x[4]xi32>
  // ZA3.Q, ZA7.Q, ZA11.Q, ZA15.Q
  // CHECK-NEXT: tile_id = 3
  %za3_s = arm_sme.get_tile : vector<[4]x[4]xi32>
  "test.some_use"(%za0_h) : (vector<[8]x[8]xi16>) -> ()
  "test.some_use"(%za1_s) : (vector<[4]x[4]xi32>) -> ()
  "test.some_use"(%za3_s) : (vector<[4]x[4]xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: za_h_overlapping_za_d
func.func @za_h_overlapping_za_d() {
  // ZA0.Q, ZA2.Q, ZA4.Q, ZA6.Q, ZA8.Q, ZA10.Q, ZA12.Q, ZA14.Q
  // CHECK-NEXT: tile_id = 0
  %za0_h = arm_sme.get_tile : vector<[8]x[8]xi16>
  // ZA1.Q, ZA9.Q
  // CHECK-NEXT: tile_id = 1
  %za1_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  // ZA3.Q, ZA11.Q
  // CHECK-NEXT: tile_id = 3
  %za3_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  // ZA5.Q, ZA13.Q
  // CHECK-NEXT: tile_id = 5
  %za5_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  // ZA7.Q, ZA15.Q
  // CHECK-NEXT: tile_id = 7
  %za7_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  "test.some_use"(%za0_h) : (vector<[8]x[8]xi16>) -> ()
  "test.some_use"(%za1_d) : (vector<[2]x[2]xi64>) -> ()
  "test.some_use"(%za3_d) : (vector<[2]x[2]xi64>) -> ()
  "test.some_use"(%za5_d) : (vector<[2]x[2]xi64>) -> ()
  "test.some_use"(%za7_d) : (vector<[2]x[2]xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: za_h_overlapping_za_q
func.func @za_h_overlapping_za_q() {
  // CHECK-NEXT: tile_id = 0
  %za0_h = arm_sme.get_tile : vector<[8]x[8]xi16>
  // CHECK-NEXT: tile_id = 1
  %za1_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 3
  %za3_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 5
  %za5_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 7
  %za7_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 9
  %za9_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 11
  %za11_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 13
  %za13_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 15
  %za15_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // Next tile is in-memory:
  // CHECK-NEXT: tile_id = 16
  %next_tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  "test.some_use"(%za0_h) : (vector<[8]x[8]xi16>) -> ()
  "test.some_use"(%za1_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za3_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za5_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za7_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za9_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za11_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za13_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za15_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%next_tile) : (vector<[1]x[1]xi128>) -> ()
  return
}

// -----

// CHECK-LABEL: za_s
func.func @za_s() {
  // CHECK-NEXT: tile_id = 0
  %za0_s = arm_sme.get_tile : vector<[4]x[4]xi32>
  // CHECK-NEXT: tile_id = 1
  %za1_s = arm_sme.get_tile : vector<[4]x[4]xi32>
  // CHECK-NEXT: tile_id = 2
  %za2_s = arm_sme.get_tile : vector<[4]x[4]xi32>
  // CHECK-NEXT: tile_id = 3
  %za3_s = arm_sme.get_tile : vector<[4]x[4]xi32>
  // Next tile is in-memory:
  // CHECK-NEXT: tile_id = 16
  %next_tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  "test.some_use"(%za0_s) : (vector<[4]x[4]xi32>) -> ()
  "test.some_use"(%za1_s) : (vector<[4]x[4]xi32>) -> ()
  "test.some_use"(%za2_s) : (vector<[4]x[4]xi32>) -> ()
  "test.some_use"(%za3_s) : (vector<[4]x[4]xi32>) -> ()
  "test.some_use"(%next_tile) : (vector<[4]x[4]xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: za_s_overlapping_za_d
func.func @za_s_overlapping_za_d() {
  // ZA0.Q, ZA4.Q, ZA8.Q, ZA12.Q
  // CHECK-NEXT: tile_id = 0
  %za0_s = arm_sme.get_tile : vector<[4]x[4]xi32>
  // ZA1.Q, ZA5.Q, ZA9.Q, ZA13.Q
  // CHECK-NEXT: tile_id = 1
  %za1_s = arm_sme.get_tile : vector<[4]x[4]xi32>
  // ZA2.Q, ZA6.Q, ZA10.Q, ZA14.Q
  // CHECK-NEXT: tile_id = 2
  %za2_s = arm_sme.get_tile : vector<[4]x[4]xi32>
  // ZA3.Q, ZA11.Q
  // CHECK-NEXT: tile_id = 3
  %za3_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  // ZA7.Q, ZA15.Q
  // CHECK-NEXT: tile_id = 7
  %za7_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  "test.some_use"(%za0_s) : (vector<[4]x[4]xi32>) -> ()
  "test.some_use"(%za1_s) : (vector<[4]x[4]xi32>) -> ()
  "test.some_use"(%za2_s) : (vector<[4]x[4]xi32>) -> ()
  "test.some_use"(%za3_d) : (vector<[2]x[2]xi64>) -> ()
  "test.some_use"(%za7_d) : (vector<[2]x[2]xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: za_s_overlapping_za_q
func.func @za_s_overlapping_za_q() {
  // CHECK-NEXT: tile_id = 0
  %za0_s = arm_sme.get_tile : vector<[4]x[4]xi32>
  // CHECK-NEXT: tile_id = 1
  %za1_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 2
  %za2_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 3
  %za3_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 5
  %za5_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 6
  %za6_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 7
  %za7_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 9
  %za9_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 10
  %za10_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 11
  %za11_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 13
  %za13_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 14
  %za14_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 15
  %za15_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // Next tile is in-memory:
  // CHECK-NEXT: tile_id = 16
  %next_tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  "test.some_use"(%za0_s) : (vector<[4]x[4]xi32>) -> ()
  "test.some_use"(%za1_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za2_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za3_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za5_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za6_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za7_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za9_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za10_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za11_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za13_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za14_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za15_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%next_tile) : (vector<[1]x[1]xi128>) -> ()
  return
}

// -----

// CHECK-LABEL: za_d
func.func @za_d() {
  // CHECK-NEXT: tile_id = 0
  %za0_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  // CHECK-NEXT: tile_id = 1
  %za1_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  // CHECK-NEXT: tile_id = 2
  %za2_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  // CHECK-NEXT: tile_id = 3
  %za3_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  // CHECK-NEXT: tile_id = 4
  %za4_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  // CHECK-NEXT: tile_id = 5
  %za5_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  // CHECK-NEXT: tile_id = 6
  %za6_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  // CHECK-NEXT: tile_id = 7
  %za7_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  // Next tile is in-memory:
  // CHECK-NEXT: tile_id = 16
  %next_tile = arm_sme.get_tile : vector<[2]x[2]xi64>
  "test.some_use"(%za0_d) : (vector<[2]x[2]xi64>) -> ()
  "test.some_use"(%za1_d) : (vector<[2]x[2]xi64>) -> ()
  "test.some_use"(%za2_d) : (vector<[2]x[2]xi64>) -> ()
  "test.some_use"(%za3_d) : (vector<[2]x[2]xi64>) -> ()
  "test.some_use"(%za4_d) : (vector<[2]x[2]xi64>) -> ()
  "test.some_use"(%za5_d) : (vector<[2]x[2]xi64>) -> ()
  "test.some_use"(%za6_d) : (vector<[2]x[2]xi64>) -> ()
  "test.some_use"(%za7_d) : (vector<[2]x[2]xi64>) -> ()
  "test.some_use"(%next_tile) : (vector<[2]x[2]xi64>) -> ()
  return
}

// -----

// CHECK-LABEL: za_d_overlapping_za_q
func.func @za_d_overlapping_za_q() {
  // CHECK-NEXT: tile_id = 0
  %za0_d = arm_sme.get_tile : vector<[2]x[2]xi64>
  // CHECK-NEXT: tile_id = 1
  %za1_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 2
  %za2_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 3
  %za3_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 4
  %za4_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 5
  %za5_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 6
  %za6_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 7
  %za7_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 9
  %za9_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 10
  %za10_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 11
  %za11_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 12
  %za12_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 13
  %za13_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 14
  %za14_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 15
  %za15_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // Next tile is in-memory:
  // CHECK-NEXT: tile_id = 16
  %next_tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  "test.some_use"(%za0_d) : (vector<[2]x[2]xi64>) -> ()
  "test.some_use"(%za1_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za2_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za3_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za4_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za5_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za6_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za7_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za9_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za10_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za11_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za12_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za13_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za14_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za15_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%next_tile) : (vector<[1]x[1]xi128>) -> ()
  return
}

// -----

// CHECK-LABEL: za_q
func.func @za_q() {
  // CHECK-NEXT: tile_id = 0
  %za0_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 1
  %za1_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 2
  %za2_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 3
  %za3_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 4
  %za4_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 5
  %za5_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 6
  %za6_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 7
  %za7_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 8
  %za8_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 9
  %za9_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 10
  %za10_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 11
  %za11_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 12
  %za12_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 13
  %za13_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 14
  %za14_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // CHECK-NEXT: tile_id = 15
  %za15_q = arm_sme.get_tile : vector<[1]x[1]xi128>
  // Next tile is in-memory:
  // CHECK-NEXT: tile_id = 16
  %next_tile = arm_sme.get_tile : vector<[1]x[1]xi128>
  "test.some_use"(%za0_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za1_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za2_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za3_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za4_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za5_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za6_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za7_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za8_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za9_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za10_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za11_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za12_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za13_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za14_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%za15_q) : (vector<[1]x[1]xi128>) -> ()
  "test.some_use"(%next_tile) : (vector<[1]x[1]xi128>) -> ()
  return
}
