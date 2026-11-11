// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  llvm.func @_QQmain(%arg0: !llvm.ptr) {
    %0 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%0 -> %arg1 : !llvm.ptr) {
      %1 = llvm.load %arg1 : !llvm.ptr -> i32 loc(#loc1)
      omp.parallel {
        %2 = llvm.add %1, %1 : i32 loc(#loc1)
        llvm.store %2, %arg1 : i32, !llvm.ptr loc(#loc1)
        omp.terminator
      } loc(#loc1)
      omp.terminator
    } loc(#loc3)
    llvm.return
  } loc(#loc2)
}
#file = #llvm.di_file<"target.f90" in "">
#cu = #llvm.di_compile_unit<id = distinct[0]<>,
 sourceLanguage = DW_LANG_Fortran95, file = #file, isOptimized = false,
 emissionKind = LineTablesOnly>
#sp_ty = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
#sp = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #cu, scope = #file,
 name = "_QQmain", file = #file, subprogramFlags = "Definition", type = #sp_ty>
#sp1 = #llvm.di_subprogram<id = distinct[2]<>, compileUnit = #cu, scope = #file,
 name = "__omp_offloading_target", file = #file, subprogramFlags = "Definition",
 type = #sp_ty>
#loc1 = loc("target.f90":12:5)
#loc2 = loc(fused<#sp>[#loc1])
#loc3 = loc(fused<#sp1>[#loc1])

// Both the aggregate holding the outlined region's arguments and the buffer
// forwarding the non-pointer value are allocated in device shared memory
// rather than on the stack, so the runtime calls that allocate and free them
// must carry a debug location, scoped to the correct function.

// CHECK: define {{.*}}@__omp_offloading_{{.*}} !dbg ![[SP:[0-9]+]] {
// CHECK: call {{.*}}@__kmpc_alloc_shared(i64 16), !dbg ![[LOC:[0-9]+]]
// CHECK: call {{.*}}@__kmpc_alloc_shared(i64 4), !dbg ![[LOC]]
// CHECK: call void @__kmpc_free_shared(ptr {{.*}}, i64 16), !dbg ![[LOC]]
// CHECK: call void @__kmpc_free_shared(ptr {{.*}}, i64 4), !dbg ![[LOC]]
// CHECK-DAG: ![[SP]] = distinct !DISubprogram(name: "__omp_offloading_target"
// CHECK-DAG: ![[LOC]] = !DILocation(line: 12, column: 5, scope: ![[SP]])
