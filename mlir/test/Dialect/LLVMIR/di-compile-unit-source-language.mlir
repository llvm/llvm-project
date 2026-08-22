// RUN: mlir-opt %s --verify-roundtrip | FileCheck %s

#file = #llvm.di_file<"source.cpp" in "/test/">

// A plain DW_LANG retains the historical inline syntax.
// CHECK: sourceLanguage = DW_LANG_C_plus_plus_20
module attributes {test.language = #llvm.di_compile_unit<
  id = distinct[0]<>, sourceLanguage = DW_LANG_C_plus_plus_20, file = #file
>} {}

// CHECK: sourceLanguage = #llvm.di_source_language_name<name = DW_LNAME_C_plus_plus, version = 202002, dialect = DW_LLVM_LANG_DIALECT_simt>
module attributes {test.versioned = #llvm.di_compile_unit<
  id = distinct[1]<>, sourceLanguage = #llvm.di_source_language_name<
    name = DW_LNAME_C_plus_plus, version = 202002,
    dialect = DW_LLVM_LANG_DIALECT_simt>, file = #file
>} {}

// An explicitly set zero version selects a source language name.
// CHECK: #llvm.di_compile_unit<
// CHECK-SAME: sourceLanguage = #llvm.di_source_language_name<name = DW_LNAME_Rust, version = 0>
module attributes {test.unversioned_name = #llvm.di_compile_unit<
  id = distinct[2]<>, sourceLanguage = #llvm.di_source_language_name<
    version = 0, name = DW_LNAME_Rust>, file = #file
>} {}
