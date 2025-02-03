// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Not intended to be a functional example, the aim of this test is to verify
// omp.threadprivate does not crash on lowering during the OpenMP target device
// pass when used in conjunction with target code in the same module.

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true } {
  llvm.func @func() attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>} {
    %0 = llvm.mlir.addressof @_QFEpointer2 : !llvm.ptr
    %1 = omp.threadprivate %0 : !llvm.ptr -> !llvm.ptr
    %2 = omp.map.info var_ptr(%1 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>) map_clauses(implicit, to) capture(ByRef) -> !llvm.ptr
    omp.target map_entries(%2 -> %arg0 : !llvm.ptr) {
      %3 = llvm.mlir.constant(1 : i32) : i32
      %4 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
      llvm.store %3, %4 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
   llvm.mlir.global internal @_QFEpointer2() {addr_space = 0 : i32} : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    llvm.return %0 : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
  }
}

// CHECK: define weak_odr protected amdgpu_kernel void @{{.*}}(ptr %{{.*}}, ptr %[[ARG1:.*]]) #{{[0-9]+}} {
// CHECK:  %[[ALLOCA:.*]] = alloca ptr, align 8
// CHECK:  store ptr %[[ARG1]], ptr %[[ALLOCA]], align 8
// CHECK:  %[[LOAD_ALLOCA:.*]] = load ptr, ptr %[[ALLOCA]], align 8
// CHECK:  store i32 1, ptr %[[LOAD_ALLOCA]], align 4
