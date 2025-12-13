// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = false} {
  llvm.func @test() {
    omp.target {
      omp.terminator
    }  loc(#loc4)
    llvm.return
  }  loc(#loc3)
}
#file = #llvm.di_file<"target.f90" in "">
#cu = #llvm.di_compile_unit<id = distinct[0]<>,
 sourceLanguage = DW_LANG_Fortran95, file = #file, isOptimized = false,
 emissionKind = Full>
#sp_ty = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
#sp = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #cu, scope = #file,
 name = "_QQmain", file = #file, subprogramFlags = "Definition", type = #sp_ty>
#sp1 = #llvm.di_subprogram<id = distinct[2]<>, compileUnit = #cu, scope = #file,
 name = "__omp_offloading_target", file = #file, subprogramFlags = "Definition",
 type = #sp_ty>
#loc1 = loc("target.f90":1:1)
#loc2 = loc("target.f90":46:3)
#loc3 = loc(fused<#sp>[#loc1])
#loc4 = loc(fused<#sp1>[#loc2])

// CHECK: ![[SP:.*]] = {{.*}}!DISubprogram(name: "__omp_offloading_target"{{.*}})

