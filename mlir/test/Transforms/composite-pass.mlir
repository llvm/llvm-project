// RUN: mlir-opt %s --log-actions-to=- --test-composite-fixed-point-pass -split-input-file --dump-pass-pipeline 2>&1 | FileCheck %s --check-prefixes=CHECK,PIPELINE
// RUN: mlir-opt %s --log-actions-to=- --composite-fixed-point-pass='name=TestCompositePass pipeline=any(canonicalize,cse)' -split-input-file | FileCheck %s

// Ensure the composite pass correctly prints its options.
// PIPELINE:      builtin.module(composite-fixed-point-pass{max-iterations=10 name=TestCompositePass
// PIPELINE-SAME: pipeline=canonicalize{ max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},cse})

// CHECK-LABEL: running `TestCompositePass`
//       CHECK: running `Canonicalizer`
//       CHECK: running `CSE`
//   CHECK-NOT: running `Canonicalizer`
//   CHECK-NOT: running `CSE`
func.func @test() {
  return
}

// -----

// CHECK-LABEL: running `TestCompositePass`
//       CHECK: running `Canonicalizer`
//       CHECK: running `CSE`
//       CHECK: running `Canonicalizer`
//       CHECK: running `CSE`
//   CHECK-NOT: running `Canonicalizer`
//   CHECK-NOT: running `CSE`
func.func @test() {
// this constant will be canonicalized away, causing another pass iteration
  %0 = arith.constant 1.5 : f32
  return
}
