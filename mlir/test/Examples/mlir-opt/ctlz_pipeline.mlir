// RUN: mlir-opt --pass-pipeline='builtin.module(builtin.module(func.func(cse,canonicalize),convert-to-llvm))' %s | FileCheck %s

// CHECK-LABEL: llvm.func @func1
// CHECK-NEXT: llvm.add
// CHECK-NEXT: llvm.add
// CHECK-NEXT: llvm.return
module {
  module {
    func.func @func1(%arg0: i32) -> i32 {
      %0 = arith.addi %arg0, %arg0 : i32
      %1 = arith.addi %arg0, %arg0 : i32
      %2 = arith.addi %0, %1 : i32
      func.return %2 : i32
    }
  }

  // CHECK-LABEL: @gpu_module
  // CHECK-LABEL: gpu.func @func2
  // CHECK-COUNT-3: arith.addi
  // CHECK-NEXT: gpu.return
  gpu.module @gpu_module {
    gpu.func @func2(%arg0: i32) -> i32 {
      %0 = arith.addi %arg0, %arg0 : i32
      %1 = arith.addi %arg0, %arg0 : i32
      %2 = arith.addi %0, %1 : i32
      gpu.return %2 : i32
    }
  }
}

