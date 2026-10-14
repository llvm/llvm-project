// RUN: mlir-opt -finalize-memref-to-llvm -convert-func-to-llvm -finalize-memref-to-llvm %s | FileCheck %s

// Verify that memref.dealloc lowering does not create a duplicate @free
// declaration when a func.func @free already exists in the module (e.g. from
// a user-defined declaration). Both func-to-llvm and memref-to-llvm patterns
// run in the same conversion in convert-to-llvm, and the func.func @free may
// not yet be converted to llvm.func when the dealloc pattern runs.
// The pipeline used in the test is to simulate MemrefToLLVM and FuncToLLVM
// patterns used in a same pass.

// CHECK-LABEL: llvm.func @dealloc_with_user_free
// CHECK:         llvm.call @free(%{{.*}}) : (!llvm.ptr) -> ()
// CHECK:         llvm.return

// CHECK: llvm.func @free(!llvm.ptr)
// CHECK-NOT: llvm.func @free

func.func @dealloc_with_user_free(%arg0: memref<?xf32>) {
  memref.dealloc %arg0 : memref<?xf32>
  return
}

func.func private @free(!llvm.ptr)
