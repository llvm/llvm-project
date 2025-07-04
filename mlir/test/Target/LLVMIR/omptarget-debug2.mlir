// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// Same test as omptarget-debug.mlir but with is_target_device = false.
// Somehow test with omp.target don't work with -split-input-file.
module attributes {omp.is_target_device = false} {
  llvm.func @_QQmain() {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
    %9 = omp.map.info var_ptr(%1 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = ""}
    omp.target map_entries(%9 -> %arg0 : !llvm.ptr) {
      %13 = llvm.mlir.constant(1 : i32) : i32
      llvm.store %13, %arg0 : i32, !llvm.ptr loc(#loc2)
      omp.terminator
    }  loc(#loc4)
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
#loc1 = loc("target.f90":1:1)
#loc2 = loc("target.f90":46:3)
#loc3 = loc(fused<#sp>[#loc1])
#loc4 = loc(fused<#sp1>[#loc1])

// CHECK-DAG: ![[SP:.*]] = {{.*}}!DISubprogram(name: "__omp_offloading_target"{{.*}})
// CHECK-DAG: !DILocation(line: 46, column: 3, scope: ![[SP]])
