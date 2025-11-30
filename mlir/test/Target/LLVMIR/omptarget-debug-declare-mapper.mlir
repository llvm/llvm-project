// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  omp.declare_mapper @_QQFmy_testmy_mapper : !llvm.struct<"_QFmy_testTmy_type", (i32)> {
  ^bb0(%arg0: !llvm.ptr):
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFmy_testTmy_type", (i32)>
    %2 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "var%data"}
    %3 = omp.map.info var_ptr(%arg0 : !llvm.ptr, !llvm.struct<"_QFmy_testTmy_type", (i32)>) map_clauses(tofrom) capture(ByRef) members(%2 : [0] : !llvm.ptr) -> !llvm.ptr {name = "var", partial_map = true}
    omp.declare_mapper.info map_entries(%3, %2 : !llvm.ptr, !llvm.ptr)
  }

  llvm.func @_QPopenmp_target_data_mapper() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<"_QFmy_testTmy_type", (i32)> {bindc_name = "a"} : (i64) -> !llvm.ptr
    %2 = omp.map.info var_ptr(%1 : !llvm.ptr, !llvm.struct<"_QFmy_testTmy_type", (i32)>) map_clauses(tofrom) capture(ByRef) mapper(@_QQFmy_testmy_mapper) -> !llvm.ptr {name = "a"}
    omp.target_data map_entries(%2 : !llvm.ptr) {
      %3 = llvm.mlir.constant(10 : i32) : i32
      %4 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"_QFmy_testTmy_type", (i32)>
      llvm.store %3, %4 : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }  loc(#loc12)
} loc(#loc)

#loc = loc("test.f90":4:18)
#loc1 = loc("test.f90":4:18)

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

#loc12 = loc(fused<#di_subprogram>[#loc1])

// CHECK: define internal void @{{.*}}omp_mapper{{.*}}_QQFmy_testmy_mapper
// CHECK-NOT: !dbg
// CHECK: }

