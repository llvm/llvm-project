// RUN: mlir-opt %s --load-pass-plugin=%standalone_libs/libStandalonePlugin.so --pass-pipeline="builtin.module(standalone-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @abar()
  func.func @abar() {
    return
  }
}
