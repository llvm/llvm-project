// RUN: mlir-opt %s -remove-dead-values | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/242375.
// A nested function with no callers has all of its results considered
// non-live, so the pass erases them. Since `wasmssa.return` is return-like,
// its operands are erased in the same pass, keeping the IR valid.

// CHECK-LABEL: wasmssa.func @f() {
// CHECK: %{{.+}} = wasmssa.const 3.140000e+00 : f64
// CHECK: wasmssa.return
// CHECK: }
wasmssa.func @f() -> f64 {
  %0 = wasmssa.const 3.14 : f64
  wasmssa.return %0 : f64
}
