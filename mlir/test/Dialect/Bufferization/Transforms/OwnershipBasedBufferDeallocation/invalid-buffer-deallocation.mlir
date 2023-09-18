// RUN: mlir-opt -verify-diagnostics -ownership-based-buffer-deallocation -split-input-file %s

func.func @free_effect() {
  %alloc = memref.alloc() : memref<2xi32>
  // expected-error @below {{memory free side-effect on MemRef value not supported!}}
  %new_alloc = memref.realloc %alloc : memref<2xi32> to memref<4xi32>
  return
}

// -----

func.func @free_effect() {
  %alloc = memref.alloc() : memref<2xi32>
  // expected-error @below {{memory free side-effect on MemRef value not supported!}}
  memref.dealloc %alloc : memref<2xi32>
  return
}

// -----

func.func @free_effect() {
  %true = arith.constant true
  %alloc = memref.alloc() : memref<2xi32>
  // expected-error @below {{No deallocation operations must be present when running this pass!}}
  bufferization.dealloc (%alloc : memref<2xi32>) if (%true)
  return
}
