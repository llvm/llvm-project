// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

#int_ty = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "integer",
  sizeInBits = 32, encoding = DW_ATE_signed>
#real_ty = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "real",
  sizeInBits = 32, encoding = DW_ATE_float>
#file = #llvm.di_file<"target.f90" in "">
#di_null_type = #llvm.di_null_type
#cu = #llvm.di_compile_unit<id = distinct[0]<>,
  sourceLanguage = DW_LANG_Fortran95, file = #file, isOptimized = false,
  emissionKind = Full>
#array_ty = #llvm.di_composite_type<tag = DW_TAG_array_type,
  baseType = #int_ty, elements = #llvm.di_subrange<count = 10 : i64>>
#sp_ty = #llvm.di_subroutine_type<callingConvention = DW_CC_program,
  types = #di_null_type>
#g_var = #llvm.di_global_variable<scope = #cu, name = "arr",
  linkageName = "_QFEarr", file = #file, line = 4,
  type = #array_ty, isDefined = true>
#g_var_expr = #llvm.di_global_variable_expression<var = #g_var>
#sp = #llvm.di_subprogram<id = distinct[2]<>, compileUnit = #cu, scope = #file,
  name = "test", file = #file, subprogramFlags = "Definition", type = #sp_ty>
#var_arr = #llvm.di_local_variable<scope = #sp,
  name = "arr", file = #file, line = 4, type = #array_ty>
#var_i = #llvm.di_local_variable<scope = #sp,
  name = "i", file = #file, line = 13, type = #int_ty>
#var_x = #llvm.di_local_variable<scope = #sp,
 name = "x", file = #file, line = 12, type = #real_ty>

module attributes {omp.is_target_device = false} {
  llvm.func @test() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr
    %4 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    %6 = llvm.mlir.constant(9 : index) : i64
    %7 = llvm.mlir.constant(0 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.constant(10 : index) : i64
    %11 = llvm.mlir.addressof @_QFEarr : !llvm.ptr
    %14 = omp.map.info var_ptr(%1 : !llvm.ptr, f32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr
    %15 = omp.map.bounds lower_bound(%7 : i64) upper_bound(%6 : i64) extent(%10 : i64) stride(%8 : i64) start_idx(%8 : i64)
    %16 = omp.map.info var_ptr(%11 : !llvm.ptr, !llvm.array<10 x i32>) map_clauses(tofrom) capture(ByRef) bounds(%15) -> !llvm.ptr
    %17 = omp.map.info var_ptr(%4 : !llvm.ptr, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !llvm.ptr
    omp.target map_entries(%14 -> %arg0, %16 -> %arg1, %17 -> %arg2 : !llvm.ptr, !llvm.ptr, !llvm.ptr) {
      llvm.intr.dbg.declare #var_x = %arg0 : !llvm.ptr
      llvm.intr.dbg.declare #var_arr = %arg1 : !llvm.ptr
      llvm.intr.dbg.declare #var_i = %arg2 : !llvm.ptr
      omp.terminator
    }
    llvm.return
  } loc(#loc3)
  llvm.mlir.global internal @_QFEarr() {addr_space = 0 : i32, dbg_exprs = [#g_var_expr]} : !llvm.array<10 x i32> {
  } loc(#loc4)
}
#loc1 = loc("target.f90":4:7)
#loc2 = loc("target.f90":11:7)
#loc3 = loc(fused<#sp>[#loc2])
#loc4 = loc(fused<#g_var>[#loc1])

// CHECK: ![[SP:[0-9]+]] = distinct !DISubprogram(name: "__omp_offloading{{.*}}test{{.*}})
// CHECK: !DILocalVariable(name: "x", arg: 1, scope: ![[SP]]{{.*}})
// CHECK: !DILocalVariable(name: "arr", arg: 2, scope: ![[SP]]{{.*}})
// CHECK: !DILocalVariable(name: "i", arg: 3, scope: ![[SP]]{{.*}})
