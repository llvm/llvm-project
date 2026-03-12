// RUN: mlir-opt %s --mlir-print-debuginfo | FileCheck %s
//
// Basic parse/print test for DILocationAttr (no snapshots).
// Uses a single source file and exercises two DILocationAttr locations (module and function).

#di_file = #llvm.di_file<"source.c" in "/">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, isOptimized = false, emissionKind = None>
#di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "test", file = #di_file, subprogramFlags = Definition>

// CHECK-DAG: #llvm.di_location<{{.*}} in #di_subprogram>
// CHECK-DAG: #llvm.di_location<{{.*}} in #di_subprogram>
#loc_at_module = #llvm.di_location<loc("source.c":1:1) in #di_subprogram>
#loc_at_func = #llvm.di_location<loc("source.c":10:2) in #di_subprogram>

module {
  llvm.func @f() {} loc(#loc_at_func)
} loc(#loc_at_module)
