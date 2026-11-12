// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{expected non-function type}}
module attributes {test.enum = #llvm.di_enumerator<>} {}

// -----

// expected-error @+1 {{expected positive signless integer width}}
module attributes {test.enum = #llvm.di_enumerator<i0, 0>} {}

// -----

// expected-error @+1 {{expected positive signless integer width}}
module attributes {test.enum = #llvm.di_enumerator<ui64, 0>} {}

// -----

// expected-error @+1 {{expected ','}}
module attributes {test.enum = #llvm.di_enumerator<i64, "A">} {}

// -----

// expected-error @+1 {{expected integer value}}
module attributes {test.enum = #llvm.di_enumerator<i64, "A",>} {}

// -----

// expected-error @+1 {{expected integer value}}
module attributes {test.enum = #llvm.di_enumerator<i64, true>} {}

// -----

// expected-error @+1 {{unsigned enumerator cannot have a negative value}}
module attributes {test.enum = #llvm.di_enumerator<i64, "neg", -1, isUnsigned>} {}

// -----

// expected-error @+1 {{integer value out of range for i8}}
module attributes {test.enum = #llvm.di_enumerator<i8, "too_big", 256, isUnsigned>} {}

// -----

// expected-error @+1 {{integer value out of range for i1}}
module attributes {test.enum = #llvm.di_enumerator<i1, "one_signed", 1>} {}

// -----

// expected-error @+1 {{expected 'isUnsigned'}}
module attributes {test.enum = #llvm.di_enumerator<i64, "bad_flag", 1, foo>} {}

