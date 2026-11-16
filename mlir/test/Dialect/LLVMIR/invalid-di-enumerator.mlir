// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{struct is missing required parameter: value}}
module attributes {test.enum = #llvm.di_enumerator<name = "A">} {}

// -----

// expected-error @+2 {{expected valid keyword}}
// expected-error @+1 {{expected a parameter name in struct}}
module attributes {test.enum = #llvm.di_enumerator<>} {}

// -----

// expected-error @+1 {{duplicate or unknown struct parameter name: value}}
module attributes {test.enum = #llvm.di_enumerator<value = 0, value = 1>} {}

// -----

// expected-error @+1 {{duplicate or unknown struct parameter name: foo}}
module attributes {test.enum = #llvm.di_enumerator<foo = 0, value = 0>} {}

// -----

// expected-error @+2 {{invalid kind of attribute specified}}
// expected-error @+1 {{failed to parse LLVM_DIEnumeratorAttr parameter 'name' which is to be a `StringAttr`}}
module attributes {test.enum = #llvm.di_enumerator<name = 0, value = 0>} {}

// -----

// expected-error @+2 {{invalid kind of attribute specified}}
// expected-error @+1 {{failed to parse LLVM_DIEnumeratorAttr parameter 'value' which is to be a `IntegerAttr`}}
module attributes {test.enum = #llvm.di_enumerator<value = "foo">} {}

// -----

// expected-error @+2 {{integer constant out of range for attribute}}
// expected-error @+1 {{failed to parse LLVM_DIEnumeratorAttr parameter 'value' which is to be a `IntegerAttr`}}
module attributes {test.enum = #llvm.di_enumerator<value = 256 : i8>} {}

// -----

// expected-error @+2 {{expected integer value}}
// expected-error @+1 {{failed to parse LLVM_DIEnumeratorAttr parameter 'isUnsigned' which is to be a `bool`}}
module attributes {test.enum = #llvm.di_enumerator<value = 0, isUnsigned = foo>} {}

// -----

// expected-error @+1 {{expected signless integer type for 'value'}}
module attributes {test.enum = #llvm.di_enumerator<value = 0 : ui64>} {}

// -----

// expected-error @+1 {{expected signless integer type for 'value'}}
module attributes {test.enum = #llvm.di_enumerator<value = 0 : si64>} {}

// -----

// expected-error @+1 {{expected signless integer type for 'value'}}
module attributes {test.enum = #llvm.di_enumerator<value = 0 : index>} {}
