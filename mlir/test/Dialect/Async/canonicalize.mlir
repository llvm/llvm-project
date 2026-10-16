// RUN: mlir-opt %s -split-input-file -canonicalize | FileCheck %s

// CHECK-NOT: async.execute

func.func @empty_execute() {
  %token = async.execute {
    async.yield
  }
  return
}
