// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Verifier: DILocationAttr source filename must match the scope's DIFile
// (basename or directory-joined). See LLVMAttrDefs.td DILocationAttr description.

// -----

// Basename differs: source 'wrong.c' vs scope 'source.c'.
#di_file = #llvm.di_file<"source.c" in "/">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, isOptimized = false, emissionKind = None>
#di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "f", file = #di_file, subprogramFlags = Definition>

// expected-error @+1 {{DILocationAttr source filename 'wrong.c' is inconsistent with scope DIFile '/source.c'}}
#bad = #llvm.di_location<loc("wrong.c":1:1) in #di_subprogram>

llvm.func @f() {} loc(#bad)

// -----

// Directory mismatch: source '/build/source.c' vs scope DIFile rendered as
// '/src/source.c'. Source carries its own directory and it disagrees with
// the scope's, so neither the basename-equality match ('/build/source.c'
// != 'source.c') nor the directory-joined match ('/build/source.c' !=
// '/src/source.c') succeeds.
#di_file = #llvm.di_file<"source.c" in "/src">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, isOptimized = false, emissionKind = None>
#di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "f", file = #di_file, subprogramFlags = Definition>

// expected-error @+1 {{DILocationAttr source filename '/build/source.c' is inconsistent with scope DIFile '/src/source.c'}}
#bad_dir = #llvm.di_location<loc("/build/source.c":1:1) in #di_subprogram>

llvm.func @f_dir_mismatch() {} loc(#bad_dir)

// -----

// Scope chain walks DILexicalBlock -> DISubprogram; bad source name still
// rejected because the chain reaches a DIFile.
#di_file = #llvm.di_file<"source.c" in "/">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, isOptimized = false, emissionKind = None>
#di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "f", file = #di_file, subprogramFlags = Definition>
#di_lb = #llvm.di_lexical_block<scope = #di_subprogram, line = 5, column = 1>

// expected-error @+1 {{DILocationAttr source filename 'other.c' is inconsistent with scope DIFile '/source.c'}}
#bad_chain = #llvm.di_location<loc("other.c":6:1) in #di_lb>

llvm.func @f_chain() {} loc(#bad_chain)

// -----

// Type-as-scope chain: DISubprogram (no own $file) -> DICompositeType
// -> DIFile. Mismatch must still be flagged through the type-as-scope branch
// of findFileInScope.
#di_file = #llvm.di_file<"class.cpp" in "/src">
#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, isOptimized = false, emissionKind = None>
#di_class = #llvm.di_composite_type<tag = DW_TAG_class_type, name = "MyClass", file = #di_file, line = 1, scope = #di_file>
#di_member = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_class, name = "foo", subprogramFlags = Definition>

// expected-error @+1 {{DILocationAttr source filename 'other.cpp' is inconsistent with scope DIFile '/src/class.cpp'}}
#bad_type_chain = #llvm.di_location<loc("other.cpp":42:1) in #di_member>

llvm.func @member_func_bad() {} loc(#bad_type_chain)
