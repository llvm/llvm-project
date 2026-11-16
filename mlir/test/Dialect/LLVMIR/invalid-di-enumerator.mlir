// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{expected signless integer type for 'value'}}
module attributes {test.enum = #llvm.di_enumerator<value = 0 : ui64>} {}

// -----

// expected-error @+1 {{expected signless integer type for 'value'}}
module attributes {test.enum = #llvm.di_enumerator<value = 0 : si64>} {}

// -----

// expected-error @+1 {{expected signless integer type for 'value'}}
module attributes {test.enum = #llvm.di_enumerator<value = 0 : index>} {}
