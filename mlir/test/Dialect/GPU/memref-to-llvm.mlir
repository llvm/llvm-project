// RUN: mlir-opt --convert-to-llvm %s | FileCheck %s

// Checking that malloc and free are declared in the proper module.

// CHECK: module attributes {gpu.container_module} {
// CHECK:   llvm.func @free(!llvm.ptr)
// CHECK:   llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:   gpu.module @kernels {
// CHECK:     llvm.func @free(!llvm.ptr)
// CHECK:     llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:     gpu.func @kernel_1
// CHECK:       llvm.call @malloc({{.*}}) : (i64) -> !llvm.ptr
// CHECK:       llvm.call @free({{.*}}) : (!llvm.ptr) -> ()
// CHECK:       gpu.return
// CHECK:     }
// CHECK:   }
// CHECK: }
module attributes {gpu.container_module} {

  gpu.module @kernels {
    gpu.func @kernel_1() kernel {
      %memref_a = memref.alloc() : memref<8x16xf32>
      memref.dealloc %memref_a : memref<8x16xf32>
      gpu.return
    }
  }

  func.func @main() {
    %memref_a = memref.alloc() : memref<8x16xf32>
    memref.dealloc %memref_a : memref<8x16xf32>
    return
  }
}
