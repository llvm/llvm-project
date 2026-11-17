// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// The __kmpc_target_init and __kmpc_target_deinit calls are emitted into the
// outlined kernel by OpenMPIRBuilder, which has no location of its own for
// them: the translation's current location is scoped to the parent function at
// that point. If they are left without a !dbg, the verifier rejects the module
// once a device runtime that carries debug info is linked in, because they
// become inlinable calls inside a function that has debug info. Check that both
// get a location scoped to the outlined function's subprogram.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true} {
  llvm.func @_QQmain() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr<5>
    %ascast = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    %9 = omp.map.info var_ptr(%ascast : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%9 -> %arg0 : !llvm.ptr) {
      %13 = llvm.mlir.constant(1 : i32) : i32
      llvm.store %13, %arg0 : i32, !llvm.ptr loc(#loc2)
      omp.terminator
    } loc(#loc4)
    llvm.return
  } loc(#loc3)
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
#loc2 = loc("target.f90":46:3)
#loc3 = loc(fused<#sp>[#loc1])
#loc4 = loc(fused<#sp1>[#loc1])

// CHECK: call i32 @__kmpc_target_init({{.*}}), !dbg ![[LOC:[0-9]+]]
// CHECK: call void @__kmpc_target_deinit(), !dbg ![[LOC]]
// CHECK-DAG: ![[SP:[0-9]+]] = distinct !DISubprogram(name: "__omp_offloading_target"
// CHECK-DAG: ![[LOC]] = !DILocation(line: 12, column: 5, scope: ![[SP]])

// -----

// The deinit call is emitted after the target body has been generated, so it
// picks up whatever debug location the body left behind. A body holding
// another construct leaves it empty, which used to cost the deinit its !dbg
// while the init, emitted before the body, kept its own.

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true} {
  llvm.func @_QQmain() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr<5>
    %ascast = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    %9 = omp.map.info var_ptr(%ascast : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) name("") -> !llvm.ptr
    omp.target kernel_type(generic) map_entries(%9 -> %arg0 : !llvm.ptr) {
      omp.parallel {
        %13 = llvm.mlir.constant(1 : i32) : i32
        llvm.store %13, %arg0 : i32, !llvm.ptr loc(#loc2)
        omp.terminator
      } loc(#loc4)
      omp.terminator
    } loc(#loc4)
    llvm.return
  } loc(#loc3)
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
#loc2 = loc("target.f90":46:3)
#loc3 = loc(fused<#sp>[#loc1])
#loc4 = loc(fused<#sp1>[#loc1])

// CHECK: call i32 @__kmpc_target_init({{.*}}), !dbg ![[PLOC:[0-9]+]]
// CHECK: call void @__kmpc_target_deinit(), !dbg ![[PLOC]]
// CHECK-DAG: ![[PSP:[0-9]+]] = distinct !DISubprogram(name: "__omp_offloading_target"
// CHECK-DAG: ![[PLOC]] = !DILocation(line: 12, column: 5, scope: ![[PSP]])
