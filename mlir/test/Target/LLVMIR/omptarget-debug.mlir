// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = true} {
  llvm.func @_QQmain() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
    %9 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    omp.target map_entries(%9 -> %arg0 : !llvm.ptr) {
      %13 = llvm.mlir.constant(1 : i32) : i32
      llvm.store %13, %arg0 : i32, !llvm.ptr loc(#loc2)
      omp.terminator
    }
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
#loc1 = loc("target.f90":1:1)
#loc2 = loc("target.f90":46:3)
#loc3 = loc(fused<#sp>[#loc1])

// CHECK-DAG: ![[SP:.*]] = {{.*}}!DISubprogram(name: "__omp_offloading_{{.*}}"{{.*}})
// CHECK-DAG: !DILocation(line: 46, column: 3, scope: ![[SP]])
