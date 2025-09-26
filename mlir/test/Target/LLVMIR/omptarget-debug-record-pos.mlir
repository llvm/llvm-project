// RUN: mlir-translate -mlir-to-llvmir %s

#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real",
 sizeInBits = 32, encoding = DW_ATE_float>
#file = #llvm.di_file<"test.f90" in "">
#di_null_type = #llvm.di_null_type
#cu = #llvm.di_compile_unit<id = distinct[0]<>,
 sourceLanguage = DW_LANG_Fortran95, file = #file, isOptimized = false,
 emissionKind = Full>
#sp_ty = #llvm.di_subroutine_type<callingConvention = DW_CC_program,
 types = #di_null_type>
#sp = #llvm.di_subprogram<compileUnit = #cu, name = "main", file=#file,
 subprogramFlags = "Definition", type = #sp_ty>
#sp1 = #llvm.di_subprogram<compileUnit = #cu, name = "target", file=#file,
 subprogramFlags = "Definition", type = #sp_ty>
#var1 = #llvm.di_local_variable<scope = #sp, name = "x", file = #file, line = 2,
 type = #di_basic_type>
#var2 = #llvm.di_local_variable<scope = #sp1, name = "x", file = #file,
 line = 2, type = #di_basic_type>

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true} {
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i64) : i64 loc(#loc2)
    %1 = llvm.alloca %0 x i1 : (i64) -> !llvm.ptr<5> loc(#loc2)
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr loc(#loc2)
    llvm.intr.dbg.declare #var1 = %2 : !llvm.ptr loc(#loc2)
    %4 = omp.map.info var_ptr(%2 : !llvm.ptr, f32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "x"} loc(#loc2)
    omp.target map_entries(%4 -> %arg0 : !llvm.ptr) {
      %5 = llvm.mlir.constant(1.000000e+00 : f32) : f32 loc(#loc3)
      llvm.intr.dbg.declare #var2 = %arg0 : !llvm.ptr loc(#loc3)
      %6 = llvm.load %arg0 : !llvm.ptr -> f32 loc(#loc3)
      %7 = llvm.fadd %6, %5 {fastmathFlags = #llvm.fastmath<contract>} : f32 loc(#loc3)
      llvm.store %7, %arg0 : f32, !llvm.ptr loc(#loc3)
      omp.terminator loc(#loc3)
    } loc(#loc4)
    llvm.return loc(#loc2)
  } loc(#loc5)
}

#loc2 = loc("test.f90":6:7)
#loc3 = loc("test.f90":8:7)
#loc4 = loc(fused<#sp1>[#loc3])
#loc5 = loc(fused<#sp>[#loc2])

// CHECK-LABEL: user_code.entry
// CHECK: %[[LOAD:.*]] = load ptr
// CHECK-NEXT:     #dbg_declare(ptr %[[LOAD]]{{.*}})
