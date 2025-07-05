// RUN: mlir-translate -mlir-to-llvmir %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true} {
  llvm.mlir.global external @_QMtest_0Esp() {addr_space = 0 : i32, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : i32 {
    %0 = llvm.mlir.constant(1 : i32) : i32 loc(#loc1)
    llvm.return %0 : i32 loc(#loc1)
  } loc(#loc1)
  llvm.func @_QQmain() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr<5> loc(#loc2)
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr loc(#loc2)
    %6 = llvm.mlir.addressof @_QMtest_0Esp : !llvm.ptr loc(#loc1)
    %7 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr loc(#loc3)
    %8 = omp.map.info var_ptr(%6 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr loc(#loc3)
    omp.target map_entries(%7 -> %arg0, %8 -> %arg1 : !llvm.ptr, !llvm.ptr) {
      %16 = llvm.load %arg1 : !llvm.ptr -> i32 loc(#loc5)
      llvm.store %16, %arg0 : i32, !llvm.ptr loc(#loc5)
      omp.terminator loc(#loc5)
    } loc(#loc16)
    llvm.return loc(#loc6)
  } loc(#loc15)
}
#di_file = #llvm.di_file<"target7.f90" in "">
#di_null_type = #llvm.di_null_type
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>,
 sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "flang",
 isOptimized = false, emissionKind = LineTablesOnly>
#di_subroutine_type = #llvm.di_subroutine_type<
  callingConvention = DW_CC_program, types = #di_null_type>
#di_subprogram = #llvm.di_subprogram<id = distinct[1]<>,
  compileUnit = #di_compile_unit, scope = #di_file, name = "main",
  file = #di_file, subprogramFlags = "Definition|MainSubprogram",
  type = #di_subroutine_type>
#di_subprogram1 = #llvm.di_subprogram<compileUnit = #di_compile_unit,
  name = "target", file = #di_file, subprogramFlags = "Definition",
  type = #di_subroutine_type>
#loc1 = loc("test.f90":3:18)
#loc2 = loc("test.f90":7:7)
#loc3 = loc("test.f90":9:18)
#loc5 = loc("test.f90":11:7)
#loc6 = loc("test.f90":12:7)
#loc15 = loc(fused<#di_subprogram>[#loc2])
#loc16 = loc(fused<#di_subprogram1>[#loc5])
