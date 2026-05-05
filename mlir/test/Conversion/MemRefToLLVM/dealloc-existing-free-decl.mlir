// RUN: mlir-opt -finalize-memref-to-llvm %s | FileCheck %s

// Verify that memref.dealloc is lowered even when a func.func @free
// declaration already exists in the module. The conflicting declaration
// is erased and replaced with llvm.func @free before the conversion runs.

// CHECK: llvm.func @free(!llvm.ptr)
// CHECK-LABEL: func.func @dealloc_with_existing_free
// CHECK:         llvm.call @free(%{{.*}}) : (!llvm.ptr) -> ()
// CHECK:         return
// CHECK-NOT: func.func @free

func.func @dealloc_with_existing_free(%arg0: memref<?xf32>) {
  memref.dealloc %arg0 : memref<?xf32>
  return
}

func.func private @free(!llvm.ptr)
