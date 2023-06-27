// This file contains test cases related to the dialect post-parsing upgrade
// mechanism.

//===--------------------------------------------------------------------===//
// Test generic
//===--------------------------------------------------------------------===//

// COM: bytecode contains
// COM: module {
// COM:   version: 2.0
// COM:   "test.versionedA"() <{dims = 123 : i64, modifier = false}> : () -> ()
// COM: }
// RUN: mlir-opt %S/versioned-op-2.0.mlirbc 2>&1 | FileCheck %s --check-prefix=CHECK1
// CHECK1: "test.versionedA"() <{dims = 123 : i64, modifier = false}> : () -> ()

//===--------------------------------------------------------------------===//
// Test upgrade
//===--------------------------------------------------------------------===//

// COM: bytecode contains
// COM: module {
// COM:   version: 1.12
// COM:   "test.versionedA"() <{dimensions = 123 : i64}> : () -> ()
// COM: }
// RUN: mlir-opt %S/versioned-op-1.12.mlirbc 2>&1 | FileCheck %s --check-prefix=CHECK2
// CHECK2: "test.versionedA"() <{dims = 123 : i64, modifier = false}> : () -> ()

//===--------------------------------------------------------------------===//
// Test forbidden downgrade
//===--------------------------------------------------------------------===//

// COM: bytecode contains
// COM: module {
// COM:   version: 2.2
// COM:   "test.versionedA"() <{dims = 123 : i64, modifier = false}> : () -> ()
// COM: }
// RUN: not mlir-opt %S/versioned-op-2.2.mlirbc 2>&1 | FileCheck %s --check-prefix=ERR_NEW_VERSION
// ERR_NEW_VERSION: current test dialect version is 2.0, can't parse version: 2.2
