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

// -----

// Test 3: When one alloc has no dealloc and another is valid, the pass errors
// on the first bad alloc and stops (WalkResult::interrupt). The error points
// precisely to the problematic alloc, not to subsequent valid ones.
func.func @error_first_bad_stops_walk() {
  // expected-error @+1 {{no dealloc found; run the deallocation pipeline before this pass}}
  %bad = memref.alloc() : memref<1024xf32>
  %ok = memref.alloc() : memref<512xf32>
  memref.dealloc %ok : memref<512xf32>
  return
}

// -----

// Test 4: cf.cond_br to a sibling block is also unstructured control flow and
// must be rejected, just like cf.br.
func.func @error_cond_br_dealloc(%cond: i1) {
  // expected-error @+1 {{unstructured control flow is not supported}}
  %alloc = memref.alloc() : memref<1024xf32>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  memref.dealloc %alloc : memref<1024xf32>
  return
^bb2:
  return
}
