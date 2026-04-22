// RUN: mlir-opt %s  -split-input-file -affine-loop-tile="tile-size=32" -verify-diagnostics | FileCheck %s
// RUN: mlir-opt %s -split-input-file -affine-loop-tile="tile-sizes=16,16" | FileCheck %s --check-prefix=TILE16

// -----

// There is no dependence violated in this case. No error should be raised.

// CHECK-DAG: [[$LB:#map[0-9]*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[$UB:#map[0-9]*]] = affine_map<(d0) -> (d0 + 32)>

// CHECK-LABEL: func @valid_to_tile()
func.func @valid_to_tile() {
  %0 = memref.alloc() : memref<64xf32>

  affine.for %i = 0 to 64 {
    %1 = affine.load %0[%i] : memref<64xf32>
    %2 = arith.addf %1, %1 : f32
    affine.store %2, %0[%i] : memref<64xf32>
  }

  return
}

// CHECK:       affine.for %{{.*}} = 0 to 64 step 32 {
// CHECK-NEXT:    affine.for %{{.*}} = [[$LB]](%{{.*}}) to [[$UB]](%{{.*}}) {

// -----

// There are dependences along the diagonal of the 2d iteration space,
// specifically, they are of direction (+, -).
// The default tiling method (hyper-rect) will violate tiling legality.
// We expect a remark that points that issue out to be emitted.

func.func @illegal_loop_with_diag_dependence() {
  %A = memref.alloc() : memref<64x64xf32>

  // No tiling here.
  // CHECK:       affine.for %{{.*}} = 0 to 64 {
  // CHECK-NEXT:    affine.for %{{.*}} = 0 to 64 {
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      %0 = affine.load %A[%j, %i] : memref<64x64xf32>
      %1 = affine.load %A[%i, %j - 1] : memref<64x64xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %A[%i, %j] : memref<64x64xf32>
    }
  }

  return
}

// -----

// Verify that tiling is not applied when anti-dependences would be violated.

// TILE16-LABEL: func @anti_dep
// TILE16:       affine.for %{{.*}} = 0 to 1023 {
// TILE16-NEXT:    affine.for %{{.*}} = 1 to 1024 {
// TILE16-NOT:     affine.for %{{.*}} = 0 to 1023 step 16
func.func @anti_dep(%arr: memref<1024x1024xi32>) {
  affine.for %i = 0 to 1023 {
    affine.for %j = 1 to 1024 {
      %val = affine.load %arr[%i + 1, %j - 1] : memref<1024x1024xi32>
      affine.store %val, %arr[%i, %j] : memref<1024x1024xi32>
    }
  }
  return
}
