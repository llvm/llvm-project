// RUN: mlir-opt %s -test-arm-sme-tile-allocation -split-input-file | FileCheck %s

// CHECK-LABEL: @always_spill_larger_or_equal_tile_type
// CHECK: arm_sme.zero {tile_id = 0 : i32} : vector<[4]x[4]xf32>
// CHECK: arm_sme.zero {tile_id = 1 : i32} : vector<[4]x[4]xf32>
// CHECK: arm_sme.zero {tile_id = 2 : i32} : vector<[4]x[4]xf32>
// CHECK: arm_sme.zero {tile_id = 3 : i32} : vector<[4]x[4]xf32>
// CHECK: arm_sme.tile_load {{.*}} {tile_id = 16 : i32} : memref<?x?xf16>, vector<[8]x[8]xf16>
func.func @always_spill_larger_or_equal_tile_type(%memref: memref<?x?xf16>) -> (vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[8]x[8]xf16>) {
  %c0 = arith.constant 0 : index
  %0 = arm_sme.zero : vector<[4]x[4]xf32>
  %1 = arm_sme.zero : vector<[4]x[4]xf32>
  %2 = arm_sme.zero : vector<[4]x[4]xf32>
  %3 = arm_sme.zero : vector<[4]x[4]xf32>
  // The load will be spilled (even though the zero's are 'trivial' spills) as a single `f32` tile would not fit the load.
  %load = arm_sme.tile_load %memref[%c0, %c0] : memref<?x?xf16>, vector<[8]x[8]xf16>
  return %0, %1, %2, %3, %load : vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[8]x[8]xf16>
}

// -----

// CHECK-LABEL: @spill_larger_tile_type
// CHECK: arm_sme.zero {tile_id = 16 : i32} : vector<[16]x[16]xi8>
// CHECK: arm_sme.tile_load {{.*}} {tile_id = 0 : i32} : memref<?x?xf32>, vector<[4]x[4]xf32>
// CHECK: arm_sme.tile_load {{.*}} {tile_id = 1 : i32} : memref<?x?xf32>, vector<[4]x[4]xf32>
// CHECK: arm_sme.tile_load {{.*}} {tile_id = 2 : i32} : memref<?x?xf32>, vector<[4]x[4]xf32>
// CHECK: arm_sme.tile_load {{.*}} {tile_id = 3 : i32} : memref<?x?xf32>, vector<[4]x[4]xf32>
func.func @spill_larger_tile_type(%memref: memref<?x?xf32>) -> (vector<[16]x[16]xi8>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>) {
  %c0 = arith.constant 0 : index
  // Spilling the `arm_sme.zero` should free up space for all four f32 tiles.
  %0 = arm_sme.zero : vector<[16]x[16]xi8>
  %1 = arm_sme.tile_load %memref[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  %2 = arm_sme.tile_load %memref[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  %3 = arm_sme.tile_load %memref[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  %4 = arm_sme.tile_load %memref[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  return %0, %1, %2, %3, %4 : vector<[16]x[16]xi8>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>
}
