// RUN: mlir-opt %s --test-bytecode-roundtrip="test-dialect-version=1.2 test-kind=6" -verify-diagnostics | FileCheck %s

module {
  "test.versionedA"() <{dims = 123 : i64, modifier = false}> : () -> ()
}

// COM: the property downgrader is executed twice: first for IR numbering and then for emission.
// CHECK: downgrading op...
// CHECK: downgrading op properties...
// CHECK: downgrading op properties...
