// RUN: mlir-opt -emit-bytecode %s | mlir-translate --mlir-to-llvmir | FileCheck %s

#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>
#di_file = #llvm.di_file<"foo.c" in "/mlir/">
#di_file1 = #llvm.di_file<"foo.c" in "/mlir/">
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>, isRecSelf = true>
#di_compile_unit = #llvm.di_compile_unit<id = distinct[1]<>, sourceLanguage = DW_LANG_C11, file = #di_file, producer = "MLIR", isOptimized = true, emissionKind = Full, nameTableKind = None>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "a", file = #di_file1, line = 2, type = #di_basic_type>
#di_subroutine_type = #llvm.di_subroutine_type<types = #di_basic_type>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[0]<>, id = distinct[2]<>, compileUnit = #di_compile_unit, scope = #di_file1, name = "main", file = #di_file1, line = 1, scopeLine = 1, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type, retainedNodes = #di_local_variable>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram1, name = "a", file = #di_file1, line = 2, type = #di_basic_type>

module attributes {dlti.dl_spec = #dlti.dl_spec<i64 = dense<64> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>>, llvm.ident = "MLIR", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.module_flags [#llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>]

  // CHECK: define i32 @main
  llvm.func @main() -> i32 attributes {passthrough = ["noinline"]} {
    %0 = llvm.mlir.constant(0 : i32) : i32 loc(#loc3)
    llvm.intr.dbg.value #di_local_variable1 = %0 : i32 loc(#loc7)
    llvm.return %0 : i32 loc(#loc8)
  } loc(#loc6)
} loc(#loc)
#loc = loc("foo.c":0:0)
#loc1 = loc("main")
#loc2 = loc("foo.c":1:0)
#loc3 = loc(unknown)
#loc4 = loc("foo.c":0:0)
#loc5 = loc("foo.c":3:0)
#loc6 = loc(fused<#di_subprogram1>[#loc1, #loc2])
#loc7 = loc(fused<#di_subprogram1>[#loc4])
#loc8 = loc(fused<#di_subprogram1>[#loc5])

