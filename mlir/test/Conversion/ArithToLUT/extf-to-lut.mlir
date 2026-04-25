// RUN: mlir-opt %s --convert-arith-fp8-extf-to-lut | FileCheck %s

// CHECK-DAG: memref.global "private" constant @__extf_lut_f8E4M3FN : memref<256xf32>
// CHECK-DAG: memref.global "private" constant @__extf_lut_f8E5M2 : memref<256xf32>

// Single f8E4M3FN input — table should appear once, op replaced by LUT sequence.
func.func @extf_f8E4M3FN(%a: f8E4M3FN) -> f32 {
  // CHECK-LABEL: @extf_f8E4M3FN
  // CHECK:       memref.get_global @__extf_lut_f8E4M3FN : memref<256xf32>
  // CHECK:       arith.bitcast {{.*}} : f8E4M3FN to i8
  // CHECK:       arith.extui {{.*}} : i8 to i32
  // CHECK:       arith.index_cast {{.*}} : i32 to index
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
