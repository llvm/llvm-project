// Default (min/max expansion disabled): the ops are kept as-is so a later
// arith-to-llvm lowering can map them to the min/max intrinsics.
// RUN: mlir-opt %s -arith-expand -split-input-file | FileCheck %s --check-prefix=DEFAULT

// include-min-max-f=true: only the float ops expand into cmpf + select; the
// integer ops are kept (integer expansion still off by default).
// RUN: mlir-opt %s -arith-expand="include-min-max-f=true" -split-input-file | FileCheck %s --check-prefix=EXPANDF

// include-min-max-i=true: only the integer ops expand into cmpi + select; the
// float ops are kept (float expansion still off by default).
// RUN: mlir-opt %s -arith-expand="include-min-max-i=true" -split-input-file | FileCheck %s --check-prefix=EXPANDI

// Both enabled: all min/max ops are expanded.
// RUN: mlir-opt %s -arith-expand="include-min-max-f=true include-min-max-i=true" -split-input-file | FileCheck %s --check-prefix=EXPANDALL

// DEFAULT-LABEL:   func @maximumf
// EXPANDF-LABEL:   func @maximumf
// EXPANDI-LABEL:   func @maximumf
// EXPANDALL-LABEL: func @maximumf
func.func @maximumf(%a: f32, %b: f32) -> f32 {
  // DEFAULT: arith.maximumf
  // DEFAULT-NOT: arith.select
  // EXPANDF: arith.cmpf ugt
  // EXPANDF: arith.select
  // EXPANDF-NOT: arith.maximumf
  // EXPANDI: arith.maximumf
  // EXPANDI-NOT: arith.select
  // EXPANDALL: arith.cmpf ugt
  // EXPANDALL: arith.select
  // EXPANDALL-NOT: arith.maximumf
  %result = arith.maximumf %a, %b : f32
  return %result : f32
}

// -----

// DEFAULT-LABEL:   func @minnumf
// EXPANDF-LABEL:   func @minnumf
// EXPANDI-LABEL:   func @minnumf
// EXPANDALL-LABEL: func @minnumf
func.func @minnumf(%a: f32, %b: f32) -> f32 {
  // DEFAULT: arith.minnumf
  // EXPANDF: arith.cmpf ult
  // EXPANDF-NOT: arith.minnumf
  // EXPANDI: arith.minnumf
  // EXPANDALL: arith.cmpf ult
  // EXPANDALL-NOT: arith.minnumf
  %result = arith.minnumf %a, %b : f32
  return %result : f32
}

// -----

// DEFAULT-LABEL:   func @maxsi
// EXPANDF-LABEL:   func @maxsi
// EXPANDI-LABEL:   func @maxsi
// EXPANDALL-LABEL: func @maxsi
func.func @maxsi(%a: i32, %b: i32) -> i32 {
  // DEFAULT: arith.maxsi
  // DEFAULT-NOT: arith.select
  // EXPANDF: arith.maxsi
  // EXPANDF-NOT: arith.select
  // EXPANDI: arith.cmpi sgt
  // EXPANDI-NOT: arith.maxsi
  // EXPANDALL: arith.cmpi sgt
  // EXPANDALL-NOT: arith.maxsi
  %result = arith.maxsi %a, %b : i32
  return %result : i32
}

// -----

// DEFAULT-LABEL:   func @minui
// EXPANDF-LABEL:   func @minui
// EXPANDI-LABEL:   func @minui
// EXPANDALL-LABEL: func @minui
func.func @minui(%a: i32, %b: i32) -> i32 {
  // DEFAULT: arith.minui
  // EXPANDF: arith.minui
  // EXPANDI: arith.cmpi ult
  // EXPANDI-NOT: arith.minui
  // EXPANDALL: arith.cmpi ult
  // EXPANDALL-NOT: arith.minui
  %result = arith.minui %a, %b : i32
  return %result : i32
}

// -----

// Regardless of the min/max gating, the other expansions (here ceildivsi)
// always run.

// DEFAULT-LABEL:   func @ceildivi_still_expands
// EXPANDALL-LABEL: func @ceildivi_still_expands
func.func @ceildivi_still_expands(%a: i32, %b: i32) -> i32 {
  // DEFAULT-NOT: arith.ceildivsi
  // EXPANDALL-NOT: arith.ceildivsi
  %result = arith.ceildivsi %a, %b : i32
  return %result : i32
}
