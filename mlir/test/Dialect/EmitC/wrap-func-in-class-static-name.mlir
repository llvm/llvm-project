// RUN: mlir-opt %s -wrap-emitc-func-in-class=class-name-format=StaticName | FileCheck %s

/// Tests that wrap-emitc-func-in-class works with a static class name format
/// that does not contain a placeholder for the function name.

emitc.func @foo() {
  emitc.return
}

// CHECK: emitc.class @StaticName {
// CHECK:   emitc.func @"operator()"() {
// CHECK:     return
// CHECK:   }
// CHECK: }
