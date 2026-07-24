// Default (both min/max groups expanded): the ops become cmpf/cmpi + select.
// RUN: mlir-opt %s -arith-expand -split-input-file | FileCheck %s --check-prefix=EXPAND

// include-min-max-f=false: only the float ops are kept; integer ops still expand.
// RUN: mlir-opt %s -arith-expand="include-min-max-f=false" -split-input-file | FileCheck %s --check-prefix=KEEPF

// include-min-max-i=false: only the integer ops are kept; float ops still expand.
// RUN: mlir-opt %s -arith-expand="include-min-max-i=false" -split-input-file | FileCheck %s --check-prefix=KEEPI

// Both disabled: all min/max ops are kept.
// RUN: mlir-opt %s -arith-expand="include-min-max-f=false include-min-max-i=false" -split-input-file | FileCheck %s --check-prefix=KEEPALL

// EXPAND-LABEL:  func @maximumf
// KEEPF-LABEL:   func @maximumf
// KEEPI-LABEL:   func @maximumf
// KEEPALL-LABEL: func @maximumf
func.func @maximumf(%a: f32, %b: f32) -> f32 {
  // EXPAND: arith.cmpf ugt
  // EXPAND: arith.select
  // EXPAND-NOT: arith.maximumf
  // KEEPF: arith.maximumf
  // KEEPF-NOT: arith.select
  // KEEPI: arith.cmpf ugt
  // KEEPI-NOT: arith.maximumf
  // KEEPALL: arith.maximumf
  // KEEPALL-NOT: arith.select
  %result = arith.maximumf %a, %b : f32
  return %result : f32
}

// -----

// EXPAND-LABEL:  func @minnumf
// KEEPF-LABEL:   func @minnumf
// KEEPI-LABEL:   func @minnumf
// KEEPALL-LABEL: func @minnumf
func.func @minnumf(%a: f32, %b: f32) -> f32 {
  // EXPAND: arith.cmpf ult
  // EXPAND-NOT: arith.minnumf
  // KEEPF: arith.minnumf
  // KEEPI: arith.cmpf ult
  // KEEPALL: arith.minnumf
  %result = arith.minnumf %a, %b : f32
  return %result : f32
}

// -----

// EXPAND-LABEL:  func @maxsi
// KEEPF-LABEL:   func @maxsi
// KEEPI-LABEL:   func @maxsi
// KEEPALL-LABEL: func @maxsi
func.func @maxsi(%a: i32, %b: i32) -> i32 {
  // EXPAND: arith.cmpi sgt
  // EXPAND-NOT: arith.maxsi
  // KEEPF: arith.cmpi sgt
  // KEEPF-NOT: arith.maxsi
  // KEEPI: arith.maxsi
  // KEEPI-NOT: arith.select
  // KEEPALL: arith.maxsi
  // KEEPALL-NOT: arith.select
  %result = arith.maxsi %a, %b : i32
  return %result : i32
}

// -----

// EXPAND-LABEL:  func @minui
// KEEPF-LABEL:   func @minui
// KEEPI-LABEL:   func @minui
// KEEPALL-LABEL: func @minui
func.func @minui(%a: i32, %b: i32) -> i32 {
  // EXPAND: arith.cmpi ult
  // EXPAND-NOT: arith.minui
  // KEEPF: arith.cmpi ult
  // KEEPI: arith.minui
  // KEEPALL: arith.minui
  %result = arith.minui %a, %b : i32
  return %result : i32
}

// -----

// Even with min/max expansion disabled, the other expansions (here
// ceildivsi) still run.

// EXPAND-LABEL:  func @ceildivi_still_expands
// KEEPALL-LABEL: func @ceildivi_still_expands
func.func @ceildivi_still_expands(%a: i32, %b: i32) -> i32 {
  // EXPAND-NOT: arith.ceildivsi
  // KEEPALL-NOT: arith.ceildivsi
  %result = arith.ceildivsi %a, %b : i32
  return %result : i32
}
