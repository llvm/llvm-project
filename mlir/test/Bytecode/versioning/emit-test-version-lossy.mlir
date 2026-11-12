// RUN: mlir-opt %s -split-input-file -emit-bytecode -emit-bytecode-dialect-version=test=1.12 -verify-diagnostics
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-dialect-version=test=2.0 | mlir-opt | FileCheck %s

// CHECK: "test.versionedA"() <{dims = 123 : i64, modifier = true}>
// expected-error@+1 {{cannot emit test dialect versions before 2.0 with 'modifier = true'}}
"test.versionedA"() <{dims = 123 : i64, modifier = true}> : () -> ()

// -----

// CHECK: test.with_versioned_properties 1 | 2
// expected-error@+1 {{cannot emit test dialect versions before 2.0 with a nonzero 'value2' property}}
test.with_versioned_properties 1 | 2
