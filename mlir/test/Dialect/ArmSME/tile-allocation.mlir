// RUN: mlir-opt %s -allocate-arm-sme-tiles -split-input-file -verify-diagnostics | FileCheck %s

// -----

// CHECK-LABEL: mixed_tiles
// CHECK-SAME: attributes {arm_sme.tiles_in_use = 65534 : i32}
func.func @mixed_tiles() {
  // ZA0.Q, ZA2.Q, ZA4.Q, ZA6.Q, ZA8.Q, ZA10.Q, ZA12.Q, ZA14.Q
  // CHECK-NEXT: arith.constant 0
  %za0_h = arm_sme.get_tile_id : i16
  // ZA1.Q, ZA5.Q, ZA9.Q, ZA13.Q
  // CHECK-NEXT: arith.constant 1
  %za1_s = arm_sme.get_tile_id : i32
  // ZA3.D ZA3.Q, ZA11.Q
  // CHECK-NEXT: arith.constant 3
  %za3_d = arm_sme.get_tile_id : i64
  // ZA7.Q
  // CHECK-NEXT: arith.constant 7
  %za7_q = arm_sme.get_tile_id : i128
  // ZA15.Q is still free.
  return
}

// -----

// CHECK-LABEL: za_b
// CHECK-SAME: attributes {arm_sme.tiles_in_use = 65535 : i32}
func.func @za_b() {
  // CHECK-NEXT: arith.constant 0
  %za0_b = arm_sme.get_tile_id : i8
  return
}

// -----

func.func @za_b__out_of_tiles() {
  %za0_b = arm_sme.get_tile_id : i8
  // expected-error@+2 {{failed to legalize operation 'arm_sme.get_tile_id' that was explicitly marked illegal}}
  // expected-error@+1 {{ran out of SME virtual tiles!}}
  %next_tile = arm_sme.get_tile_id : i8
  return
}

// -----

func.func @za_b_overlapping_za_q() {
  %za0_b = arm_sme.get_tile_id : i8
  // expected-error@+2 {{failed to legalize operation 'arm_sme.get_tile_id' that was explicitly marked illegal}}
  // expected-error@+1 {{ran out of SME virtual tiles!}}
  %next_tile = arm_sme.get_tile_id : i128
  return
}

// -----

// CHECK-LABEL: za0_h
// CHECK-SAME: attributes {arm_sme.tiles_in_use = 43690 : i32}
func.func @za0_h() {
  // CHECK-NEXT: arith.constant 0
  %za0_h = arm_sme.get_tile_id : i16
  return
}

// -----

// CHECK-LABEL: za_h
// CHECK-SAME: attributes {arm_sme.tiles_in_use = 65535 : i32}
func.func @za_h() {
  // CHECK-NEXT: arith.constant 0
  %za0_h = arm_sme.get_tile_id : i16
  // CHECK-NEXT: arith.constant 1
  %za1_h = arm_sme.get_tile_id : i16
  return
}

// -----

func.func @za_h__out_of_tiles() {
  %za0_h = arm_sme.get_tile_id : i16
  %za1_h = arm_sme.get_tile_id : i16
  // expected-error@+2 {{failed to legalize operation 'arm_sme.get_tile_id' that was explicitly marked illegal}}
  // expected-error@+1 {{ran out of SME virtual tiles!}}
  %next_tile = arm_sme.get_tile_id : i16
  return
}

// -----

// CHECK-LABEL: za_h_overlapping_za_s
// CHECK-SAME: attributes {arm_sme.tiles_in_use = 65535 : i32}
func.func @za_h_overlapping_za_s() {
  // ZA0.Q, ZA2.Q, ZA4.Q, ZA6.Q, ZA8.Q, ZA10.Q, ZA12.Q, ZA14.Q
  // CHECK-NEXT: arith.constant 0
  %za0_h = arm_sme.get_tile_id : i16
  // ZA1.Q, ZA5.Q, ZA9.Q, ZA13.Q
  // CHECK-NEXT: arith.constant 1
  %za1_s = arm_sme.get_tile_id : i32
  // ZA3.Q, ZA7.Q, ZA11.Q, ZA15.Q
  // CHECK-NEXT: arith.constant 3
  %za3_s = arm_sme.get_tile_id : i32
  return
}

// -----

// CHECK-LABEL: za_h_overlapping_za_d
// CHECK-SAME: attributes {arm_sme.tiles_in_use = 65535 : i32}
func.func @za_h_overlapping_za_d() {
  // ZA0.Q, ZA2.Q, ZA4.Q, ZA6.Q, ZA8.Q, ZA10.Q, ZA12.Q, ZA14.Q
  // CHECK-NEXT: arith.constant 0
  %za0_h = arm_sme.get_tile_id : i16
  // ZA1.Q, ZA9.Q
  // CHECK-NEXT: arith.constant 1
  %za1_d = arm_sme.get_tile_id : i64
  // ZA3.Q, ZA11.Q
  // CHECK-NEXT: arith.constant 3
  %za3_d = arm_sme.get_tile_id : i64
  // ZA5.Q, ZA13.Q
  // CHECK-NEXT: arith.constant 5
  %za5_d = arm_sme.get_tile_id : i64
  // ZA7.Q, ZA15.Q
  // CHECK-NEXT: arith.constant 7
  %za7_d = arm_sme.get_tile_id : i64
  return
}

// -----

func.func @za_h_overlapping_za_q() {
  %za0_h = arm_sme.get_tile_id : i16
  %za0_q = arm_sme.get_tile_id : i128
  %za2_q = arm_sme.get_tile_id : i128
  %za4_q = arm_sme.get_tile_id : i128
  %za6_q = arm_sme.get_tile_id : i128
  %za8_q = arm_sme.get_tile_id : i128
  %za10_q = arm_sme.get_tile_id : i128
  %za12_q = arm_sme.get_tile_id : i128
  %za14_q = arm_sme.get_tile_id : i128
  // expected-error@+2 {{failed to legalize operation 'arm_sme.get_tile_id' that was explicitly marked illegal}}
  // expected-error@+1 {{ran out of SME virtual tiles!}}
  %next_tile = arm_sme.get_tile_id : i128
  return
}

// -----

// CHECK-LABEL: za0_s
// CHECK-SAME: attributes {arm_sme.tiles_in_use = 34952 : i32}
func.func @za0_s() {
  // CHECK-NEXT: arith.constant 0
  %za0_s = arm_sme.get_tile_id : i32
  return
}

// -----

// CHECK-LABEL: za_s
// CHECK-SAME: attributes {arm_sme.tiles_in_use = 65535 : i32}
func.func @za_s() {
  // CHECK-NEXT: arith.constant 0
  %za0_s = arm_sme.get_tile_id : i32
  // CHECK-NEXT: arith.constant 1
  %za1_s = arm_sme.get_tile_id : i32
  // CHECK-NEXT: arith.constant 2
  %za2_s = arm_sme.get_tile_id : i32
  // CHECK-NEXT: arith.constant 3
  %za3_s = arm_sme.get_tile_id : i32
  return
}

// -----

func.func @za_s__out_of_tiles() {
  %za0_s = arm_sme.get_tile_id : i32
  %za1_s = arm_sme.get_tile_id : i32
  %za2_s = arm_sme.get_tile_id : i32
  %za3_s = arm_sme.get_tile_id : i32
  // expected-error@+2 {{failed to legalize operation 'arm_sme.get_tile_id' that was explicitly marked illegal}}
  // expected-error@+1 {{ran out of SME virtual tiles!}}
  %next_tile = arm_sme.get_tile_id : i32
  return
}

// -----

// CHECK-LABEL: za_s_overlapping_za_d
// CHECK-SAME: attributes {arm_sme.tiles_in_use = 65535 : i32}
func.func @za_s_overlapping_za_d() {
  // ZA0.Q, ZA4.Q, ZA8.Q, ZA12.Q
  // CHECK-NEXT: arith.constant 0
  %za0_s = arm_sme.get_tile_id : i32
  // ZA1.Q, ZA5.Q, ZA9.Q, ZA13.Q
  // CHECK-NEXT: arith.constant 1
  %za1_s = arm_sme.get_tile_id : i32
  // ZA2.Q, ZA6.Q, ZA10.Q, ZA14.Q
  // CHECK-NEXT: arith.constant 2
  %za2_s = arm_sme.get_tile_id : i32
  // ZA3.Q, ZA11.Q
  // CHECK-NEXT: arith.constant 3
  %za3_d = arm_sme.get_tile_id : i64
  // ZA7.Q, ZA15.Q
  // CHECK-NEXT: arith.constant 7
  %za7_d = arm_sme.get_tile_id : i64
  return
}

// -----

func.func @za_s_overlapping_za_q() {
  %za0_s = arm_sme.get_tile_id : i32
  %za1_q = arm_sme.get_tile_id : i128
  %za2_q = arm_sme.get_tile_id : i128
  %za3_q = arm_sme.get_tile_id : i128
  %za5_q = arm_sme.get_tile_id : i128
  %za6_q = arm_sme.get_tile_id : i128
  %za7_q = arm_sme.get_tile_id : i128
  %za9_q = arm_sme.get_tile_id : i128
  %za10_q = arm_sme.get_tile_id : i128
  %za11_q = arm_sme.get_tile_id : i128
  %za13_q = arm_sme.get_tile_id : i128
  %za14_q = arm_sme.get_tile_id : i128
  %za15_q = arm_sme.get_tile_id : i128
  // expected-error@+2 {{failed to legalize operation 'arm_sme.get_tile_id' that was explicitly marked illegal}}
  // expected-error@+1 {{ran out of SME virtual tiles!}}
  %next_tile = arm_sme.get_tile_id : i128
  return
}

// -----

// CHECK-LABEL: za0_d
// CHECK-SAME: attributes {arm_sme.tiles_in_use = 32896 : i32}
func.func @za0_d() {
  // CHECK-NEXT: arith.constant 0
  %za0_d = arm_sme.get_tile_id : i64
  return
}

// -----

// CHECK-LABEL: za_d
// CHECK-SAME: attributes {arm_sme.tiles_in_use = 65535 : i32}
func.func @za_d() {
  // CHECK-NEXT: arith.constant 0
  %za0_d = arm_sme.get_tile_id : i64
  // CHECK-NEXT: arith.constant 1
  %za1_d = arm_sme.get_tile_id : i64
  // CHECK-NEXT: arith.constant 2
  %za2_d = arm_sme.get_tile_id : i64
  // CHECK-NEXT: arith.constant 3
  %za3_d = arm_sme.get_tile_id : i64
  // CHECK-NEXT: arith.constant 4
  %za4_d = arm_sme.get_tile_id : i64
  // CHECK-NEXT: arith.constant 5
  %za5_d = arm_sme.get_tile_id : i64
  // CHECK-NEXT: arith.constant 6
  %za6_d = arm_sme.get_tile_id : i64
  // CHECK-NEXT: arith.constant 7
  %za7_d = arm_sme.get_tile_id : i64
  return
}

// -----

func.func @za_d__out_of_tiles() {
  %za0_d = arm_sme.get_tile_id : i64
  %za1_d = arm_sme.get_tile_id : i64
  %za2_d = arm_sme.get_tile_id : i64
  %za3_d = arm_sme.get_tile_id : i64
  %za4_d = arm_sme.get_tile_id : i64
  %za5_d = arm_sme.get_tile_id : i64
  %za6_d = arm_sme.get_tile_id : i64
  %za7_d = arm_sme.get_tile_id : i64
  // expected-error@+2 {{failed to legalize operation 'arm_sme.get_tile_id' that was explicitly marked illegal}}
  // expected-error@+1 {{ran out of SME virtual tiles!}}
  %next_tile = arm_sme.get_tile_id : i64
  return
}

// -----

func.func @za_d_overlapping_za_q() {
  %za0_d = arm_sme.get_tile_id : i64
  %za1_q = arm_sme.get_tile_id : i128
  %za2_q = arm_sme.get_tile_id : i128
  %za3_q = arm_sme.get_tile_id : i128
  %za4_q = arm_sme.get_tile_id : i128
  %za5_q = arm_sme.get_tile_id : i128
  %za6_q = arm_sme.get_tile_id : i128
  %za7_q = arm_sme.get_tile_id : i128
  %za9_q = arm_sme.get_tile_id : i128
  %za10_q = arm_sme.get_tile_id : i128
  %za11_q = arm_sme.get_tile_id : i128
  %za12_q = arm_sme.get_tile_id : i128
  %za13_q = arm_sme.get_tile_id : i128
  %za14_q = arm_sme.get_tile_id : i128
  %za15_q = arm_sme.get_tile_id : i128
  // expected-error@+2 {{failed to legalize operation 'arm_sme.get_tile_id' that was explicitly marked illegal}}
  // expected-error@+1 {{ran out of SME virtual tiles!}}
  %next_tile = arm_sme.get_tile_id : i128
  return
}

// -----

// CHECK-LABEL: za0_q
// CHECK-SAME: attributes {arm_sme.tiles_in_use = 32768 : i32}
func.func @za0_q() {
  // CHECK-NEXT: arith.constant 0
  %za0_q = arm_sme.get_tile_id : i128
  return
}

// -----

// CHECK-LABEL: za_q
// CHECK-SAME: attributes {arm_sme.tiles_in_use = 65535 : i32}
func.func @za_q() {
  // CHECK-NEXT: arith.constant 0
  %za0_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 1
  %za1_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 2
  %za2_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 3
  %za3_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 4
  %za4_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 5
  %za5_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 6
  %za6_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 7
  %za7_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 8
  %za8_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 9
  %za9_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 10
  %za10_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 11
  %za11_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 12
  %za12_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 13
  %za13_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 14
  %za14_q = arm_sme.get_tile_id : i128
  // CHECK-NEXT: arith.constant 15
  %za15_q = arm_sme.get_tile_id : i128
  return
}

// -----

func.func @za_q__out_of_tiles() {
  %za0_q = arm_sme.get_tile_id : i128
  %za1_q = arm_sme.get_tile_id : i128
  %za2_q = arm_sme.get_tile_id : i128
  %za3_q = arm_sme.get_tile_id : i128
  %za4_q = arm_sme.get_tile_id : i128
  %za5_q = arm_sme.get_tile_id : i128
  %za6_q = arm_sme.get_tile_id : i128
  %za7_q = arm_sme.get_tile_id : i128
  %za8_q = arm_sme.get_tile_id : i128
  %za9_q = arm_sme.get_tile_id : i128
  %za10_q = arm_sme.get_tile_id : i128
  %za11_q = arm_sme.get_tile_id : i128
  %za12_q = arm_sme.get_tile_id : i128
  %za13_q = arm_sme.get_tile_id : i128
  %za14_q = arm_sme.get_tile_id : i128
  %za15_q = arm_sme.get_tile_id : i128
  // expected-error@+2 {{failed to legalize operation 'arm_sme.get_tile_id' that was explicitly marked illegal}}
  // expected-error@+1 {{ran out of SME virtual tiles!}}
  %next_tile = arm_sme.get_tile_id : i128
  return
}
