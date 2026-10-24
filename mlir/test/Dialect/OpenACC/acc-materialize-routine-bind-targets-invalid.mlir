// RUN: not mlir-opt %s --pass-pipeline='builtin.module(func.func(acc-materialize-routine-bind-targets))' 2>&1 | FileCheck %s

module {
  func.func @test() {
    return
  }
}

// CHECK: error: root operation must have the SymbolTable trait
