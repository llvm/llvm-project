// This file contains test cases related to the dialect post-parsing upgrade
// mechanism.
// COM: those tests parse bytecode that was generated before test dialect
//      adopted `usePropertiesFromAttributes`.

//===--------------------------------------------------------------------===//
// Test generic
//===--------------------------------------------------------------------===//

// COM: bytecode contains
// COM: module {
// COM:   version: 2.0
// COM:   test.with_versioned_properties 1 | 2
// COM: }
// RUN: mlir-opt %S/versioned-op-with-native-prop-2.0.mlirbc 2>&1 | FileCheck %s --check-prefix=CHECK1
// CHECK1: test.with_versioned_properties 1 | 2

//===--------------------------------------------------------------------===//
// Test upgrade
//===--------------------------------------------------------------------===//

// COM: bytecode contains
// COM: module {
// COM:   version: 1.12

// COM: }
// RUN: mlir-opt %S/versioned-op-with-native-prop-1.12.mlirbc 2>&1 | FileCheck %s --check-prefix=CHECK3
// CHECK3: test.with_versioned_properties 1 | 0
