// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{negative integer cannot have an unsigned type}}
module attributes {test.enum = #llvm.di_enumerator<name = "neg", -1 : i8 unsigned>} {}

// -----

// expected-error @+1 {{integer value too large for the given type}}
module attributes {test.enum = #llvm.di_enumerator<name = "too_big", 256 : i8 unsigned>} {}

// -----

// expected-error @+1 {{integer value out of range for the given type}}
module attributes {test.enum = #llvm.di_enumerator<name = "too_big", 128 : i8>} {}

// -----

// expected-error @+1 {{integer value out of range for the given type}}
module attributes {test.enum = #llvm.di_enumerator<name = "too_neg", -129 : i8>} {}

// -----

// expected-error @+1 {{expected non-zero width signless integer type}}
module attributes {test.enum = #llvm.di_enumerator<0 : i0>} {}

// -----

// expected-error @+1 {{expected non-zero width signless integer type}}
module attributes {test.enum = #llvm.di_enumerator<0 : ui64>} {}

// -----

// expected-error @+1 {{expected non-zero width signless integer type}}
module attributes {test.enum = #llvm.di_enumerator<0 : si64>} {}

// -----

// expected-error @+1 {{expected ':'}}
module attributes {test.enum = #llvm.di_enumerator<0>} {}

// -----

// expected-error @+1 {{expected non-function type}}
module attributes {test.enum = #llvm.di_enumerator<0 :>} {}

// -----

// expected-error @+1 {{expected integer value}}
module attributes {test.enum = #llvm.di_enumerator<foo : i64>} {}

// -----

// expected-error @+1 {{expected integer value}}
module attributes {test.enum = #llvm.di_enumerator<"not_int" : i64>} {}

// -----

// expected-error @+1 {{expected '>'}}
module attributes {test.enum = #llvm.di_enumerator<0 : i64 unsigned unsigned>} {}

// -----

// expected-error @+1 {{expected '>'}}
module attributes {test.enum = #llvm.di_enumerator<0 : i64 foo>} {}

// -----

// expected-error @+2 {{invalid kind of attribute specified}}
// expected-error @+1 {{failed to parse LLVM_DIEnumeratorAttr parameter 'name' which is to be a `StringAttr`}}
module attributes {test.enum = #llvm.di_enumerator<name = 0 : i64>} {}
