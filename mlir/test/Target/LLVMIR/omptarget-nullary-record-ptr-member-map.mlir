// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This test checks that we generate a select statement in cases where we're
// mapping a pointer, to select a size of 0 when the pointer is null and
// select the size of the mapped type when it is not null. Preventing a runtime
// mapping error in cases where we legally map null data to device.

module attributes {omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @test_select_gen(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) var_ptr_ptr(%arg1 : !llvm.ptr) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %1 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)>) map_clauses(to) capture(ByRef) members(%0 : [0] : !llvm.ptr) -> !llvm.ptr
    omp.target map_entries(%0 -> %arg2, %1 -> %arg3 : !llvm.ptr, !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: {{.*}}test_select_gen({{.*}}, ptr %[[ARG1:.*]]) {{.*}}
// CHECK: %[[LOAD_ARG1:.*]] = load ptr, ptr %[[ARG1]], align 8
// CHECK: %[[ICMP_ARG1:.*]] = icmp eq ptr %[[LOAD_ARG1]], null
// CHECK: %[[SEL_ARG1:.*]] = select i1 %[[ICMP_ARG1]], i64 0, i64 4
