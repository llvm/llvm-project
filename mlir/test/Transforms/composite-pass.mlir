// RUN: mlir-opt %s --log-actions-to=- --test-composite-pass -split-input-file | FileCheck %s
// RUN: mlir-opt %s --log-actions-to=- --composite-pass='name=TestCompositePass pipeline=any(canonicalize,cse)' -split-input-file | FileCheck %s

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
