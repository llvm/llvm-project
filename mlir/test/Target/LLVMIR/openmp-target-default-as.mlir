// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// This tests that we correctly use the default program AS from the data layout.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.program_memory_space", 4 : ui32>>, llvm.target_triple = "spirv64-intel", omp.is_target_device = true, omp.is_gpu = true} {

// CHECK: @[[IDENT:.*]] = private unnamed_addr constant %s{{.*}} { i32 0, i32 2, i32 0, i32 22, ptr addrspace(4) addrspacecast (ptr @{{.*}} to ptr addrspace(4)) }, align 8

 llvm.func @omp_target_region_() {
    %0 = llvm.mlir.constant(20 : i32) : i32
    %1 = llvm.mlir.constant(10 : i32) : i32
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.alloca %2 x i32 {bindc_name = "a", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFomp_target_regionEa"} : (i64) -> !llvm.ptr<5>
    %4 = llvm.addrspacecast %3 : !llvm.ptr<5> to !llvm.ptr
    llvm.store %1, %4 : i32, !llvm.ptr
    %map = omp.map.info var_ptr(%4 : !llvm.ptr, i32)   map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    omp.target map_entries(%map -> %arg : !llvm.ptr) {
      omp.terminator
    }
    llvm.return
  }
}
