// Verify that bytecode written before func.func and gpu.module gained a dlti
// property can still be read. The fixture was produced from this input by
// mlir-opt built at fc696f791e06 (the PR's parent) using -emit-bytecode.
// RUN: mlir-opt %S/dlti-query-op-interface-pre-dlti.mlirbc | FileCheck %s --check-prefix=OLD
// RUN: mlir-opt %S/dlti-query-op-interface-pre-dlti-gpu.mlirbc | FileCheck %s --check-prefix=OLD-GPU
// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s --check-prefix=NEW

// The GPU-only fixture uses the same parent revision and this input:
//   module attributes {gpu.container_module} {
//     gpu.module @legacy_gpu {}
//   }

// OLD: module attributes {gpu.container_module}
// OLD: func.func @legacy()
// OLD: gpu.module @legacy_gpu
// OLD-GPU: module attributes {gpu.container_module}
// OLD-GPU: gpu.module @legacy_gpu
// NEW: module attributes {gpu.container_module}
// NEW: func.func @legacy()
// NEW: gpu.module @legacy_gpu
module attributes {gpu.container_module} {
  func.func @legacy() {
    return
  }
  gpu.module @legacy_gpu {
  }
}
