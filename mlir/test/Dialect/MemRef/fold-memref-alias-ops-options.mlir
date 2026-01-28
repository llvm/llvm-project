// RUN: mlir-opt --test-fold-memref-alias-options="exclude-pattern=load-subview" -split-input-file %s | FileCheck %s --check-prefix=EXCLUDE
// RUN: mlir-opt --test-fold-memref-alias-options="control-attr=no_fold" -split-input-file %s | FileCheck %s --check-prefix=CONTROL

// -----

// Excluding the load-subview pattern keeps the subview + load untouched.
func.func @exclude_load_subview(%arg0: memref<4xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %sv = memref.subview %arg0[0] [4] [1] : memref<4xf32> to memref<4xf32, strided<[1], offset: 0>>
  %v = memref.load %sv[%c0] : memref<4xf32, strided<[1], offset: 0>>
  return %v : f32
}

// EXCLUDE-LABEL: func.func @exclude_load_subview
// EXCLUDE: %[[SV:.*]] = memref.subview
// EXCLUDE: memref.load %[[SV]]
// EXCLUDE-NOT: memref.load %arg0

// -----

// Control callback rejects ops carrying the attribute; the plain load is still
// folded through the subview.
func.func @control_attr(%arg0: memref<4xf32>) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %sv = memref.subview %arg0[0] [4] [1] : memref<4xf32> to memref<4xf32, strided<[1], offset: 0>>
  %blocked = memref.load %sv[%c0] {no_fold} : memref<4xf32, strided<[1], offset: 0>>
  %folded = memref.load %sv[%c0] : memref<4xf32, strided<[1], offset: 0>>
  return %blocked, %folded : f32, f32
}

// CONTROL-LABEL: func.func @control_attr
// CONTROL: %[[SV:.*]] = memref.subview
// CONTROL: %[[A:.*]] = memref.load %[[SV]][%c0] {no_fold}
// CONTROL: %[[B:.*]] = memref.load %arg0[%c0]
