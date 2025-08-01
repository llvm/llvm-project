// RUN: mlir-opt %s --pass-pipeline="builtin.module(ensure-debug-info-scope-on-llvm-func)" --split-input-file --mlir-print-debuginfo | FileCheck %s
// RUN: mlir-opt %s --pass-pipeline="builtin.module(ensure-debug-info-scope-on-llvm-func{emission-kind=DebugDirectivesOnly})" --split-input-file --mlir-print-debuginfo | FileCheck --check-prefix=CHECK_OTHER_KIND %s 

// CHECK-LABEL: llvm.func @func_no_debug()
// CHECK: llvm.return loc(#loc
// CHECK: loc(#loc[[LOC:[0-9]+]])
// CHECK: #di_file = #llvm.di_file<"<unknown>" in "">
// CHECK: #di_subprogram = #llvm.di_subprogram<id = distinct[{{.*}}]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "func_no_debug", linkageName = "func_no_debug", file = #di_file, line = 1, scopeLine = 1, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type>
// CHECK: #loc[[LOC]] = loc(fused<#di_subprogram>
module {
  llvm.func @func_no_debug() {
    llvm.return loc(unknown)
  } loc(unknown)
} loc(unknown)

// -----

// Test that the declarations subprogram is not made distinct.
// CHECK-LABEL: llvm.func @func_decl_no_debug()
// CHECK: #di_subprogram = #llvm.di_subprogram<
// CHECK-NOT: id = distinct
module {
  llvm.func @func_decl_no_debug() loc(unknown)
} loc(unknown)

// -----

// Test that existing debug info is not overwritten.
// CHECK-LABEL: llvm.func @func_with_debug()
// CHECK: llvm.return loc(#loc
// CHECK: loc(#loc[[LOC:[0-9]+]])
// CHECK: #di_file = #llvm.di_file<"<unknown>" in "">
// CHECK: #di_subprogram = #llvm.di_subprogram<id = distinct[{{.*}}]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "func_with_debug", linkageName = "func_with_debug", file = #di_file, line = 42, scopeLine = 42, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type>
// CHECK: #loc[[LOC]] = loc(fused<#di_subprogram>
// CHECK_OTHER_KIND: emissionKind = DebugDirectivesOnly
module {
  llvm.func @func_with_debug() {
    llvm.return loc(#loc1)
  } loc(#loc2)
} loc(#loc)
#di_file = #llvm.di_file<"<unknown>" in "">
#di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
#loc = loc("foo":0:0)
#loc1 = loc(unknown)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, producer = "MLIR", isOptimized = true, emissionKind = LineTablesOnly>
#di_subprogram = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "func_with_debug", linkageName = "func_with_debug", file = #di_file, line = 42, scopeLine = 42, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type>
#loc2 = loc(fused<#di_subprogram>[#loc1])

// -----

// Test that a compile unit present on a module op is used for the function.
// CHECK-LABEL: llvm.func @propagate_compile_unit()
// CHECK: llvm.return loc(#loc
// CHECK: loc(#loc[[FUNCLOC:[0-9]+]])
// CHECK: loc(#loc[[MODULELOC:[0-9]+]])
// CHECK-DAG: #[[DI_FILE_MODULE:.+]] = #llvm.di_file<"bar.mlir" in "baz">
// CHECK-DAG: #[[DI_FILE_FUNC:.+]] = #llvm.di_file<"file.mlir" in ""> 
// CHECK-DAG: #loc[[FUNCFILELOC:[0-9]+]] = loc("file.mlir":9:8)
// CHECK-DAG: #di_compile_unit = #llvm.di_compile_unit<id = distinct[{{.*}}]<>, sourceLanguage = DW_LANG_C, file = #[[DI_FILE_MODULE]], producer = "MLIR", isOptimized = true, emissionKind = LineTablesOnly>
// CHECK-DAG: #di_subprogram = #llvm.di_subprogram<id = distinct[{{.*}}]<>, compileUnit = #di_compile_unit, scope = #[[DI_FILE_FUNC]], name = "propagate_compile_unit", linkageName = "propagate_compile_unit", file = #[[DI_FILE_FUNC]], line = 9, scopeLine = 9, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type>
// CHECK-DAG: #loc[[MODULELOC]] = loc(fused<#di_compile_unit>[#loc])
// CHECK-DAG: #loc[[FUNCLOC]] = loc(fused<#di_subprogram>[#loc[[FUNCFILELOC]]
module {
  llvm.func @propagate_compile_unit() {
    llvm.return loc(unknown)
  } loc("file.mlir":9:8)
} loc(#loc)
#di_file = #llvm.di_file<"bar.mlir" in "baz">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, producer = "MLIR", isOptimized = true, emissionKind = LineTablesOnly>
#loc = loc(fused<#di_compile_unit>["foo.mlir":2:1])

// -----

// Test that only one compile unit is created.
// CHECK-LABEL: module @multiple_funcs
// CHECK: llvm.di_compile_unit
// CHECK-NOT: llvm.di_compile_unit
module @multiple_funcs {
  llvm.func @func0() {
    llvm.return loc(unknown)
  } loc(unknown)
  llvm.func @func1() {
    llvm.return loc(unknown)
  } loc(unknown)
} loc(unknown)

// -----

// CHECK-LABEL: llvm.func @func_inlined()
// CHECK: #di_file = #llvm.di_file<"base.py" in "testing">
// CHECK: #di_file1 = #llvm.di_file<"gpu_test.py" in "">
// CHECK: #di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
// CHECK: #di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, producer = "MLIR", isOptimized = true, emissionKind = LineTablesOnly>
// CHECK: #loc3 = loc(callsite(#loc1 at #loc2))
// CHECK: #di_subprogram = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #di_compile_unit, scope = #di_file1, name = "func_inlined", linkageName = "func_inlined", file = #di_file1, line = 1150, scopeLine = 1150, subprogramFlags = "Definition|Optimized", type = #di_subroutine_type>
// CHECK: #di_lexical_block_file = #llvm.di_lexical_block_file<scope = #di_subprogram, file = #di_file, discriminator = 0>


#loc = loc("gpu_test.py":1150:34 to :43)
#loc1 = loc("testing/parameterized.py":321:17 to :53)
#loc2 = loc("testing/base.py":2904:19 to :56)
#loc_1 = loc(callsite(#loc at #loc1))
#loc21 = loc(callsite(#loc2 at #loc_1))

module {
  llvm.func @func_inlined() {
    llvm.return loc(#loc21)
  } loc(#loc)
} loc(#loc2)

// -----

// CHECK-LABEL: llvm.func @func_name_with_child()
// CHECK: #di_file = #llvm.di_file<"file" in "/tmp">
// CHECK: #di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
// CHECK: #di_subprogram = #llvm.di_subprogram<scope = #di_file, name = "func_name_with_child", linkageName = "func_name_with_child", file = #di_file, line = 100, scopeLine = 100, subprogramFlags = Optimized, type = #di_subroutine_type>

module {
  llvm.func @func_name_with_child() loc("foo"("/tmp/file":100))
} loc(unknown)

// -----

// CHECK-LABEL: llvm.func @func_fusion()
// CHECK: #di_file = #llvm.di_file<"file" in "/tmp">
// CHECK: #di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
// CHECK: #di_subprogram = #llvm.di_subprogram<scope = #di_file, name = "func_fusion", linkageName = "func_fusion", file = #di_file, line = 20, scopeLine = 20, subprogramFlags = Optimized, type = #di_subroutine_type>

module {
  llvm.func @func_fusion() loc(fused<"myPass">["foo", "/tmp/file":20])
} loc(unknown)

// -----

// CHECK-LABEL: llvm.func @func_callsiteloc()
// CHECK: #di_file = #llvm.di_file<"mysource.cc" in "">
// CHECK: #di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
// CHECK: #di_subprogram = #llvm.di_subprogram<scope = #di_file, name = "func_callsiteloc", linkageName = "func_callsiteloc", file = #di_file, line = 10, scopeLine = 10, subprogramFlags = Optimized, type = #di_subroutine_type>

module {
  llvm.func @func_callsiteloc() loc(callsite("foo" at "mysource.cc":10:8))
} loc(unknown)

