// RUN: mlir-opt -ownership-based-buffer-deallocation -split-input-file %s | \
// RUN: mlir-opt -ownership-based-buffer-deallocation -split-input-file %s

// This should not be an error because the ownership based buffer deallocation introduces
// deallocs itself, so running it twice over (say when piping IR over different tools with
// their own pipelines) crashes the compiler on perfectly valid code.

func.func @free_effect() {
  %alloc = memref.alloc() : memref<2xi32>
  %new_alloc = memref.realloc %alloc : memref<2xi32> to memref<4xi32>
  return
}

// -----

func.func @free_effect() {
  %alloc = memref.alloc() : memref<2xi32>
  memref.dealloc %alloc : memref<2xi32>
  return
}
