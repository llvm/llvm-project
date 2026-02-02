// RUN: mlir-opt -xegpu-subgroup-distribute -split-input-file %s 2>&1 | FileCheck %s

// Test that the pass gracefully handles a GPU module without a target attribute
// instead of crashing with "UNREACHABLE executed".

// CHECK-LABEL: gpu.module @no_target_module
// The function body should remain unchanged since no target is attached.
// CHECK: gpu.func @simple_func
// CHECK: gpu.return
gpu.module @no_target_module {
  gpu.func @simple_func(%arg0: memref<8x16xf16>) {
    gpu.return
  }
}
