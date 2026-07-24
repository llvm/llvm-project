// Default (include-min-max=true): the min/max ops are expanded into
// cmpf/cmpi + select sequences.
// RUN: mlir-opt %s -arith-expand -split-input-file | FileCheck %s --check-prefix=EXPAND

// include-min-max=false: the min/max ops are left untouched so that a later
// arith-to-llvm conversion can lower them to the single-instruction
// llvm.intr.maximum/minimum/... intrinsics.
// RUN: mlir-opt %s -arith-expand="include-min-max=false" -split-input-file | FileCheck %s --check-prefix=KEEP

// EXPAND-LABEL: func @maximumf
// KEEP-LABEL:   func @maximumf
func.func @maximumf(%a: f32, %b: f32) -> f32 {
  // EXPAND: arith.cmpf ugt
  // EXPAND: arith.select
  // EXPAND: arith.cmpf uno
  // EXPAND: arith.select
  // EXPAND-NOT: arith.maximumf
  // KEEP: arith.maximumf
  // KEEP-NOT: arith.select
  %result = arith.maximumf %a, %b : f32
  return %result : f32
}

// -----

// EXPAND-LABEL: func @minnumf
// KEEP-LABEL:   func @minnumf
func.func @minnumf(%a: f32, %b: f32) -> f32 {
  // EXPAND: arith.cmpf ult
  // EXPAND: arith.select
  // EXPAND-NOT: arith.minnumf
  // KEEP: arith.minnumf
  // KEEP-NOT: arith.select
  %result = arith.minnumf %a, %b : f32
  return %result : f32
}

// -----

// EXPAND-LABEL: func @maxsi
// KEEP-LABEL:   func @maxsi
func.func @maxsi(%a: i32, %b: i32) -> i32 {
  // EXPAND: arith.cmpi sgt
  // EXPAND: arith.select
  // EXPAND-NOT: arith.maxsi
  // KEEP: arith.maxsi
  // KEEP-NOT: arith.select
  %result = arith.maxsi %a, %b : i32
  return %result : i32
}

// -----

// Even with min/max expansion disabled, the other expansions (here
// ceildivsi) still run.

// EXPAND-LABEL: func @ceildivi_still_expands
// KEEP-LABEL:   func @ceildivi_still_expands
func.func @ceildivi_still_expands(%a: i32, %b: i32) -> i32 {
  // EXPAND-NOT: arith.ceildivsi
  // KEEP-NOT: arith.ceildivsi
  %result = arith.ceildivsi %a, %b : i32
  return %result : i32
}
