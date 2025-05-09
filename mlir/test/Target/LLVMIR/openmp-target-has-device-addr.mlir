// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// integer :: x(:)
// omp target has_device_addr(x)

// CHECK-LABEL: ModuleID
// CHECK: @.offload_sizes = private unnamed_addr constant [1 x i64] [i64 48]
// CHECK: @.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 549]

module attributes { llvm.target_triple = "x86_64-unknown-linux-gnu", omp.target_triples = ["amdgcn-amd-amdhsa"], omp.version = #omp.version<version = 52>} {
  llvm.func @has_device_addr(%arg0: !llvm.ptr) attributes {target_cpu = "x86-64"} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %41 = omp.map.info var_ptr(%1 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(always, implicit, to) capture(ByRef) -> !llvm.ptr {name = "x"}
    omp.target has_device_addr(%41 -> %arg1 : !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}
