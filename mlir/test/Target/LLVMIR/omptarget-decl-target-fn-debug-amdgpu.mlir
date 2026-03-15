// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

#file = #llvm.di_file<"target.f90" in "">
#cu = #llvm.di_compile_unit<id = distinct[0]<>,
 sourceLanguage = DW_LANG_Fortran95, file = #file, isOptimized = false,
 emissionKind = LineTablesOnly>
#sp_ty = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
#sp = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #cu, scope = #file,
 name = "add", file = #file, subprogramFlags = "Definition", type = #sp_ty>
#ty = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real", sizeInBits = 32, encoding = DW_ATE_float>
#var_a = #llvm.di_local_variable<scope = #sp, name = "a", file = #file, line = 22, arg = 1, type = #ty>


module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_target_device = true, dlti.dl_spec = #dlti.dl_spec<"dlti.alloca_memory_space" = 5 : ui64>} {
  llvm.func @add(%arg0: !llvm.ptr) attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} {
    llvm.intr.dbg.declare #var_a = %arg0 : !llvm.ptr loc(#loc2)
    llvm.return
  } loc(#loc3)
}

#loc1 = loc("target.f90":1:1)
#loc2 = loc("target.f90":46:3)
#loc3 = loc(fused<#sp>[#loc1])

// CHECK: define{{.*}}@add(ptr %[[ARG:[0-9]+]]){{.*}}!dbg ![[SP:[0-9]+]] {
// CHECK: #dbg_declare(ptr %[[ARG]], ![[A:[0-9]+]], !DIExpression(DIOpArg(0, ptr), DIOpDeref(ptr)), !{{.*}})
// CHECK: }
// CHECK: ![[SP]] = {{.*}}!DISubprogram(name: "add"{{.*}})
// CHECK: ![[A]] = !DILocalVariable(name: "a", arg: 1, scope: ![[SP]]{{.*}})
