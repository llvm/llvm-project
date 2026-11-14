// RUN: mlir-opt %s -convert-openmp-to-llvm | mlir-translate -mlir-to-llvmir | FileCheck %s

// Check SystemZ signext on i32 parameters of omp_target_alloc/free.

module attributes {llvm.target_triple = "s390x-unknown-linux-gnu"} {
  llvm.func @test_alloc_free() {
    %device = llvm.mlir.constant(0 : i32) : i32
    %1 = omp.target_allocmem %device : i32, i64
    omp.target_freemem %device, %1 : i32, i64
    llvm.return
  }
}

// CHECK-LABEL: define void @test_alloc_free()
// CHECK: call ptr @omp_target_alloc(i64 8, i32 signext 0)
// CHECK: call void @omp_target_free(ptr %{{.+}}, i32 signext 0)
// CHECK: declare ptr @omp_target_alloc(i64, i32 signext)
// CHECK: declare void @omp_target_free(ptr, i32 signext)
