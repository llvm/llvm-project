// RUN: mlir-opt %s -symbol-privatize=exclude="aap" | FileCheck %s

// CHECK-LABEL: module attributes {test.simple}
module attributes {test.simple} {
  // CHECK: func @aap
  func @aap() { return }

  // CHECK: func private @kat
  func @kat() { return }
}

