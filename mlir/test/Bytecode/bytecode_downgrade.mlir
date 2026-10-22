// RUN: mlir-opt %s --test-bytecode-roundtrip="test-dialect-version=1.2 test-kind=6" -verify-diagnostics | FileCheck %s

module {
  "test.versionedA"() <{dims = 123 : i64, modifier = false}> : () -> ()
}

// CHECK: downgrading op...
// CHECK: "test.versionedA"() <{dims = 123 : i64, modifier = false}>
