// RUN: mlir-opt %s -split-input-file -verify-diagnostics

#file = #llvm.di_file<"source.cpp" in "/test/">

// expected-error @+1 {{expected exactly one of language or name}}
module attributes {test.language = #llvm.di_source_language_name<>} {}

// -----

// expected-error @+1 {{expected exactly one of language or name}}
module attributes {test.language = #llvm.di_source_language_name<language = DW_LANG_C, name = DW_LNAME_C, version = 199901>} {}

// -----

// expected-error @+1 {{duplicate or unknown struct parameter name: sourceLanguage}}
module attributes {test.cu = #llvm.di_compile_unit<sourceLanguage = DW_LANG_C_plus_plus_20, sourceLanguage = #llvm.di_source_language_name<name = DW_LNAME_C_plus_plus, version = 202002>, file = #file>} {}

// -----

// expected-error @+1 {{sourceLanguage must be set}}
module attributes {test.cu = #llvm.di_compile_unit<>} {}

// -----

// expected-error @+1 {{DW_LANG cannot have a version}}
module attributes {test.cu = #llvm.di_compile_unit<sourceLanguage = #llvm.di_source_language_name<language = DW_LANG_C, version = 202002>>} {}

// -----

// expected-error @+1 {{DW_LNAME requires a version}}
module attributes {test.cu = #llvm.di_compile_unit<sourceLanguage = #llvm.di_source_language_name<name = DW_LNAME_Rust>>} {}
