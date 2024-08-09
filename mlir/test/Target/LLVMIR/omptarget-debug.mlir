// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = true} {
  llvm.func @_QQmain() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr<5>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<5> to !llvm.ptr
    %9 = omp.map.info var_ptr(%2 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    omp.target map_entries(%9 -> %arg0 : !llvm.ptr) {
    ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
      %13 = llvm.mlir.constant(1 : i32) : i32
      %14 = llvm.load %arg0 : !llvm.ptr -> i32 loc(#loc2)
      %15 = llvm.add %14, %13  : i32 loc(#loc2)
      llvm.store %15, %arg0 : i32, !llvm.ptr loc(#loc2)
      omp.terminator
    }
    llvm.return
  } loc(#loc5)
}
#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "void", encoding = DW_ATE_address>
#di_file = #llvm.di_file<"target.f90" in "">
#loc1 = loc("target.f90":1:1)
#loc2 = loc("target.f90":46:3)

#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, producer = "Flang", isOptimized = false, emissionKind = LineTablesOnly>
#di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal, types = #di_basic_type, #di_basic_type>
#di_subprogram = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "_QQmain", linkageName = "_QQmain", file = #di_file, line = 1, scopeLine = 1, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type>
#loc5 = loc(fused<#di_subprogram>[#loc1])

// 45:  !$omp target map(tofrom: a)
// 46:  a = a + 1
// 47:  !$omp end target

// CHECK: [[SP:.*]] = distinct !DISubprogram(name: "__omp_offloading_{{.*}}"{{.*}})
// CHECK-DAG: !DILocation(line: 46, column: 3, scope: [[SP]])
