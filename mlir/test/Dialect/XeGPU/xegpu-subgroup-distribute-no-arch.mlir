// RUN: mlir-opt --xegpu-subgroup-distribute -split-input-file %s | FileCheck %s
// Regression test for https://github.com/llvm/llvm-project/issues/181531:
// Running --xegpu-subgroup-distribute without a chip target attribute used to
// call llvm_unreachable in getUArch(). The pass should now bail out gracefully.

// CHECK-LABEL: gpu.func @no_crash_without_chip_attr
// CHECK:       gpu.return
gpu.module @test_module {
  gpu.func @no_crash_without_chip_attr(%arg0: memref<8x16xf16>, %arg1: memref<8x16xf16>) {
    gpu.return
  }
}
