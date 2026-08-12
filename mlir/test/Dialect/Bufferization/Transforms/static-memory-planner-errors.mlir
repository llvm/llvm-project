// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(static-memory-planner-analysis))" \
// RUN:     -split-input-file -verify-diagnostics

// -----

// Test 1: Alloc with no dealloc should be an error (not silently skipped).
func.func @error_no_dealloc() {
  // expected-error @+1 {{no dealloc found; run the deallocation pipeline before this pass}}
  %alloc = memref.alloc() : memref<1024xf32>
  return
}

// -----

// Test 2: Alloc whose dealloc escapes to a sibling block via unstructured
// control flow (cf.br) should be an error.
func.func @error_escaping_dealloc(%cond: i1) {
  // expected-error @+1 {{unstructured control flow is not supported}}
  %alloc = memref.alloc() : memref<1024xf32>
  cf.br ^bb1
^bb1:
  memref.dealloc %alloc : memref<1024xf32>
  return
}
