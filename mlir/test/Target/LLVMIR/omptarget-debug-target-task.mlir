// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = false} {
  llvm.func @omp_target_depend_() {
    %0 = llvm.mlir.constant(39 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(40 : index) : i64
    %3 = omp.map.bounds lower_bound(%1 : i64) upper_bound(%0 : i64) extent(%2 : i64) stride(%1 : i64) start_idx(%1 : i64)
    %4 = llvm.mlir.addressof @_QFEa : !llvm.ptr
    %5 = omp.map.info var_ptr(%4 : !llvm.ptr, !llvm.array<40 x i32>) map_clauses(from) capture(ByRef) bounds(%3) -> !llvm.ptr {name = "a"}
    omp.target depend(taskdependin -> %4 : !llvm.ptr) map_entries(%5 -> %arg0 : !llvm.ptr) {
      %6 = llvm.mlir.constant(100 : index) : i32
      llvm.store %6, %arg0 : i32, !llvm.ptr
      omp.terminator
    } loc(#loc13)
    llvm.return
  } loc(#loc12)

  llvm.mlir.global internal @_QFEa() {addr_space = 0 : i32} : !llvm.array<40 x i32> {
    %0 = llvm.mlir.zero : !llvm.array<40 x i32>
    llvm.return %0 : !llvm.array<40 x i32>
  }
}

#loc1 = loc("test.f90":4:18)
#loc2 = loc("test.f90":8:7)

#di_file = #llvm.di_file<"test.f90" in "">
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


#loc12 = loc(fused<#di_subprogram>[#loc1])
#loc13 = loc(fused<#di_subprogram1>[#loc2])

// CHECK: define internal void @.omp_target_task_proxy_func
// CHECK-NOT: !dbg
// CHECK: }
