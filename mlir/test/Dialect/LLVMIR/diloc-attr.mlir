// RUN: mlir-opt %s --mlir-print-debuginfo -split-input-file | FileCheck %s
//
// Parse/print and verifier coverage for DILocationAttr. The verifier ensures
// the FileLineColLoc filename is consistent with the scope's DIFile (when
// reachable).

// -----

// Basename match: source filename equals DIFile.name (directory ignored).
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

// -----

// Directory-joined match: source filename equals "<DIFile.directory>/<DIFile.name>".
#di_file = #llvm.di_file<"source.c" in "/src">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, isOptimized = false, emissionKind = None>
#di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "test_joined", file = #di_file, subprogramFlags = Definition>

// CHECK: #llvm.di_location<{{.*}} in #di_subprogram>
#loc_joined = #llvm.di_location<loc("/src/source.c":1:1) in #di_subprogram>

llvm.func @g_joined() {} loc(#loc_joined)

// -----

// Scope chain walks DILexicalBlock -> DILexicalBlock -> DISubprogram to find
// the DIFile. Inner blocks omit the optional file field.
#di_file = #llvm.di_file<"source.c" in "/">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, isOptimized = false, emissionKind = None>
#di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "test_chain", file = #di_file, subprogramFlags = Definition>
#di_lb_outer = #llvm.di_lexical_block<scope = #di_subprogram, line = 5, column = 1>
#di_lb_inner = #llvm.di_lexical_block<scope = #di_lb_outer, line = 6, column = 3>

// CHECK: #llvm.di_location<{{.*}} in #di_lexical_block1>
#loc_in_inner = #llvm.di_location<loc("source.c":7:5) in #di_lb_inner>

llvm.func @h_chain() {} loc(#loc_in_inner)

// -----

// Type-as-scope: scope chain walks DISubprogram (no own $file) -> DICompositeType
// -> DIFile. Models a C++ member function whose enclosing class carries the file.
#di_file = #llvm.di_file<"class.cpp" in "/src">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, isOptimized = false, emissionKind = None>
#di_class = #llvm.di_composite_type<tag = DW_TAG_class_type, name = "MyClass", file = #di_file, line = 1, scope = #di_file>
// Intentionally no $file on the member subprogram, so findFileInScope must
// walk through #di_class to find the DIFile.
#di_member = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_class, name = "foo", subprogramFlags = Definition>

// CHECK: #llvm.di_location<{{.*}} in #di_subprogram>
#loc_in_member = #llvm.di_location<loc("/src/class.cpp":42:1) in #di_member>

llvm.func @member_func() {} loc(#loc_in_member)

// -----

// "When you have one" — DISubprogram without a file means there's no DIFile
// reachable in the scope chain, so the source filename is unconstrained.
#di_subprogram = #llvm.di_subprogram<recId = distinct[0]<>>

// CHECK: #llvm.di_location<{{.*}} in #di_subprogram>
#loc_no_file_in_scope = #llvm.di_location<loc("anything.c":1:1) in #di_subprogram>

llvm.func @no_file_scope() {} loc(#loc_no_file_in_scope)

// -----

// Empty source filename ("" from text, or programmatic "") is rewritten by
// MLIR's FileLineColRange builder to the "-" sentinel. The verifier treats
// "-" as the no-source-info placeholder and skips the consistency check —
// this covers the pass-generated "<unknown>" / empty-source pair used by
// DIScopeForLLVMFuncOp for ops with no source info.
#di_file = #llvm.di_file<"<unknown>" in "">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, isOptimized = false, emissionKind = None>
#di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "empty_source", file = #di_file, subprogramFlags = Definition>

// CHECK: #llvm.di_location<{{.*}} in #di_subprogram>
#loc_empty = #llvm.di_location<loc("":0:0) in #di_subprogram>

llvm.func @empty_source_filename() {} loc(#loc_empty)
