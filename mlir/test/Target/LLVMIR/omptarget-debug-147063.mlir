// RUN: mlir-translate -mlir-to-llvmir %s

module attributes {llvm.target_triple = "x86_64-unknown-linux-gnu", omp.is_gpu = false, omp.is_target_device = false, omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  omp.private {type = private} @_QFFfnEv_private_i32 : i32 loc(#loc1)
  llvm.func internal @_QFPfn() {
    %0 = llvm.mlir.constant(1 : i64) : i64 loc(#loc1)
    %1 = llvm.alloca %0 x i32 {bindc_name = "v"} : (i64) -> !llvm.ptr loc(#loc1)
    %2 = llvm.mlir.constant(1 : i32) : i32
    omp.parallel private(@_QFFfnEv_private_i32 %1 -> %arg0 : !llvm.ptr) {
      llvm.store %2, %arg0 : i32, !llvm.ptr loc(#loc2)
      %4 = omp.map.info var_ptr(%arg0 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr {name = "v"} loc(#loc2)
      omp.target map_entries(%4 -> %arg1 : !llvm.ptr) {
        %5 = llvm.mlir.constant(1 : i32) : i32
        %6 = llvm.load %arg1 : !llvm.ptr -> i32 loc(#loc3)
        %7 = llvm.add %6, %5 : i32 loc(#loc3)
        llvm.store %7, %arg1 : i32, !llvm.ptr loc(#loc3)
        omp.terminator loc(#loc3)
      } loc(#loc7)
      omp.terminator
    } loc(#loc4)
    llvm.return
  } loc(#loc6)
}

#di_file = #llvm.di_file<"target.f90" in "">
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

#loc1 = loc("test.f90":7:15)
#loc2 = loc("test.f90":1:7)
#loc3 = loc("test.f90":3:7)
#loc4 = loc("test.f90":16:7)
#loc6 = loc(fused<#di_subprogram>[#loc1])
#loc7 = loc(fused<#di_subprogram1>[#loc3])
