// RUN: mlir-opt %s -inline -split-input-file | FileCheck %s

func.func @func() -> i32 {
  %0 = ub.poison : i32
  return %0 : i32
}

// CHECK-LABEL: func @test_inline
func.func @test_inline(%ptr : !llvm.ptr) -> i32 {
// CHECK-NOT: call
  %0 = call @func() : () -> i32
  return %0 : i32
}
