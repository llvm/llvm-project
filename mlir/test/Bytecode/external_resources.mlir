// RUN: mlir-opt %s -emit-bytecode | mlir-opt | FileCheck %s

module {
}

{-#
  // CHECK: external_resources
  external_resources: {
    // CHECK-NEXT: mlir_reproducer
    mlir_reproducer: {
      // CHECK-NEXT: pipeline: "builtin.module(func.func(canonicalize,cse))",
      pipeline: "builtin.module(func.func(canonicalize,cse))",
      // CHECK-NEXT: disable_threading: true
      disable_threading: true,
      // CHECK-NEXT: verify_each: true
      verify_each: true
    }
  }
#-}
