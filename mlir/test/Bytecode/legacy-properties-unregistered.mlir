// RUN: mlir-opt %s -allow-unregistered-dialect -split-input-file -emit-bytecode -emit-bytecode-version=0 -verify-diagnostics
// RUN: mlir-opt %s -allow-unregistered-dialect -split-input-file -emit-bytecode -emit-bytecode-version=1 -verify-diagnostics
// RUN: mlir-opt %s -allow-unregistered-dialect -split-input-file -emit-bytecode -emit-bytecode-version=2 -verify-diagnostics
// RUN: mlir-opt %s -allow-unregistered-dialect -split-input-file -emit-bytecode -emit-bytecode-version=3 -verify-diagnostics
// RUN: mlir-opt %s -allow-unregistered-dialect -split-input-file -emit-bytecode -emit-bytecode-version=4 -verify-diagnostics
// RUN: mlir-opt %s -allow-unregistered-dialect -emit-bytecode -emit-bytecode-version=5 | mlir-opt -allow-unregistered-dialect | FileCheck %s
// RUN: mlir-opt %s -allow-unregistered-dialect -emit-bytecode -emit-bytecode-version=6 | mlir-opt -allow-unregistered-dialect | FileCheck %s
// RUN: mlir-opt %s -allow-unregistered-dialect -emit-bytecode | mlir-opt -allow-unregistered-dialect | FileCheck %s

// Unregistered operations without properties can still use legacy bytecode.
// CHECK: "unknown.no_properties"() {discardable = 42 : i64}
"unknown.no_properties"() {discardable = 42 : i64} : () -> ()

// -----

// Even an empty property dictionary differs from the initialized null value.
// CHECK: "unknown.empty_properties"() <{}>
// expected-error@+1 {{cannot emit bytecode before version 5 without losing operation properties}}
"unknown.empty_properties"() <{}> : () -> ()

// -----

// CHECK: "unknown.properties"() <{value = 42 : i64}>
// expected-error@+1 {{cannot emit bytecode before version 5 without losing operation properties}}
"unknown.properties"() <{value = 42 : i64}> : () -> ()
