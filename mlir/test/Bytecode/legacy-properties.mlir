// RUN: mlir-opt %s -split-input-file -emit-bytecode -emit-bytecode-version=0 -verify-diagnostics
// RUN: mlir-opt %s -split-input-file -emit-bytecode -emit-bytecode-version=1 -verify-diagnostics
// RUN: mlir-opt %s -split-input-file -emit-bytecode -emit-bytecode-version=2 -verify-diagnostics
// RUN: mlir-opt %s -split-input-file -emit-bytecode -emit-bytecode-version=3 -verify-diagnostics
// RUN: mlir-opt %s -split-input-file -emit-bytecode -emit-bytecode-version=4 -verify-diagnostics
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-version=5 | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-version=6 | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode | mlir-opt | FileCheck %s

// Native properties cannot be merged into the legacy attribute dictionary.
// CHECK: test.with_versioned_properties 1 | 2
// expected-error@+1 {{cannot emit bytecode before version 5 without losing operation properties}}
test.with_versioned_properties 1 | 2

// -----

// A discardable attribute with the native property's name does not restore it.
// CHECK: test.with_versioned_properties 1 | 2 {prop = {value1 = 1 : i32, value2 = 2 : i32}}
// expected-error@+1 {{cannot emit bytecode before version 5 without losing operation properties}}
test.with_versioned_properties 1 | 2 {prop = {value1 = 1 : i32, value2 = 2 : i32}}

// -----

// Default native properties can still round-trip through a legacy format.
// CHECK: test.with_versioned_properties 0 | 0
test.with_versioned_properties 0 | 0

// -----

// Inherent attributes can be reconstructed from the legacy dictionary.
// CHECK: "test.versionedA"() <{dims = 123 : i64, modifier = true}>
"test.versionedA"() <{dims = 123 : i64, modifier = true}> : () -> ()
