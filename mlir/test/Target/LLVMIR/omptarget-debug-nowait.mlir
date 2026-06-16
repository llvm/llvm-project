// RUN: mlir-translate -mlir-to-llvmir %s

module attributes {omp.is_target_device = false} {
  llvm.func @main() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr
    %3 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    %6 = omp.map.info var_ptr(%1 : !llvm.ptr, f32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %7 = omp.map.info var_ptr(%3 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr
    omp.target nowait map_entries(%6 -> %arg0, %7 -> %arg1 : !llvm.ptr, !llvm.ptr) {
      %8 = llvm.mlir.constant(0 : i64) : i64
      %9 = llvm.mlir.constant(100 : i32) : i32
      llvm.br ^bb1(%9, %8 : i32, i64)
    ^bb1(%13: i32, %14: i64):  // 2 preds: ^bb0, ^bb2
      %15 = llvm.icmp "sgt" %14, %8 : i64
      llvm.cond_br %15, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      llvm.store %13, %arg1 : i32, !llvm.ptr
      llvm.br ^bb1(%13, %14 : i32, i64)
    ^bb3:  // pred: ^bb1
      llvm.store %13, %arg1 : i32, !llvm.ptr
      omp.terminator
    } loc(#loc3)
    llvm.return
  } loc(#loc2)
}

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

#loc1 = loc("test.f90":6:7)
#loc2 = loc(fused<#sp>[#loc1])
#loc3 = loc(fused<#sp1>[#loc1])

