// RUN: mlir-opt %s --convert-arith-extf-to-lut | FileCheck %s

// CHECK-DAG: memref.global "private" constant @__extf_lut_f4E2M1FN : memref<16xf32>
// CHECK-DAG: memref.global "private" constant @__extf_lut_f8E4M3FN : memref<256xf32>
// CHECK-DAG: memref.global "private" constant @__extf_lut_f8E5M2 : memref<256xf32>

// f4E2M1FN — 16-entry LUT.
func.func @extf_f4E2M1FN(%a: f4E2M1FN) -> f32 {
  // CHECK-LABEL: @extf_f4E2M1FN
  // CHECK:       memref.get_global @__extf_lut_f4E2M1FN : memref<16xf32>
  // CHECK:       arith.bitcast {{.*}} : f4E2M1FN to i4
  // CHECK:       arith.index_castui {{.*}} : i4 to index
  // CHECK:       memref.load {{.*}}[{{.*}}] : memref<16xf32>
  // CHECK-NOT:   arith.extf
  %r = arith.extf %a : f4E2M1FN to f32
  return %r : f32
}

// Single f8E4M3FN input — table should appear once, op replaced by LUT sequence.
func.func @extf_f8E4M3FN(%a: f8E4M3FN) -> f32 {
  // CHECK-LABEL: @extf_f8E4M3FN
  // CHECK:       memref.get_global @__extf_lut_f8E4M3FN : memref<256xf32>
  // CHECK:       arith.bitcast {{.*}} : f8E4M3FN to i8
  // CHECK:       arith.index_castui {{.*}} : i8 to index
  // CHECK:       memref.load {{.*}}[{{.*}}] : memref<256xf32>
  // CHECK-NOT:   arith.extf
  %r = arith.extf %a : f8E4M3FN to f32
  return %r : f32
}

// Two f8E4M3FN inputs — same table reused, not duplicated.
func.func @extf_f8E4M3FN_twice(%a: f8E4M3FN, %b: f8E4M3FN) -> f32 {
  // CHECK-LABEL: @extf_f8E4M3FN_twice
  // CHECK-COUNT-2: memref.get_global @__extf_lut_f8E4M3FN
  %ra = arith.extf %a : f8E4M3FN to f32
  %rb = arith.extf %b : f8E4M3FN to f32
  %sum = arith.addf %ra, %rb : f32
  return %sum : f32
}

// Second f8 format — separate table emitted.
func.func @extf_f8E5M2(%a: f8E5M2) -> f32 {
  // CHECK-LABEL: @extf_f8E5M2
  // CHECK:       memref.get_global @__extf_lut_f8E5M2 : memref<256xf32>
  // CHECK-NOT:   arith.extf
  %r = arith.extf %a : f8E5M2 to f32
  return %r : f32
}

// Vector f8E4M3FN → f32: LUT looked up via vector.gather.
func.func @extf_vec_f8E4M3FN(%a: vector<4xf8E4M3FN>) -> vector<4xf32> {
  // CHECK-LABEL: @extf_vec_f8E4M3FN
  // CHECK:       memref.get_global @__extf_lut_f8E4M3FN : memref<256xf32>
  // CHECK:       arith.bitcast {{.*}} : vector<4xf8E4M3FN> to vector<4xi8>
  // CHECK:       arith.extui {{.*}} : vector<4xi8> to vector<4xi32>
  // CHECK:       vector.constant_mask [4] : vector<4xi1>
  // CHECK:       vector.gather {{.*}} : memref<256xf32>
  // CHECK-NOT:   arith.extf
  %r = arith.extf %a : vector<4xf8E4M3FN> to vector<4xf32>
  return %r : vector<4xf32>
}
