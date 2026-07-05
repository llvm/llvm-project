// This file contains a test case representative of a dialect parsing an
// attribute with versioned custom encoding.

//===--------------------------------------------------------------------===//
// Test attribute upgrade
//===--------------------------------------------------------------------===//

// COM: bytecode contains
// COM: module {
// COM:   version: 1.12
// COM:   "test.versionedB"() <{attribute = #test.attr_params<24, 42>}> : () -> ()
// COM: }
// RUN: mlir-opt %S/versioned-attr-1.12.mlirbc 2>&1 | FileCheck %s --check-prefix=CHECK1
// CHECK1: "test.versionedB"() <{attribute = #test.attr_params<42, 24>}> : () -> ()

//===--------------------------------------------------------------------===//
// Test attribute upgrade
//===--------------------------------------------------------------------===//

// COM: bytecode contains
// COM: module {
// COM:   version: 2.0
// COM:   "test.versionedB"() <{attribute = #test.attr_params<42, 24>}> : () -> ()
// COM: }
// RUN: mlir-opt %S/versioned-attr-2.0.mlirbc 2>&1 | FileCheck %s --check-prefix=CHECK2
// CHECK2: "test.versionedB"() <{attribute = #test.attr_params<42, 24>}> : () -> ()
