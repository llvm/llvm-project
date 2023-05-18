// REQUIRES: asserts
// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(mem2reg))' --split-input-file --mlir-pass-statistics 2>&1 >/dev/null | FileCheck %s

// CHECK: Mem2Reg
// CHECK-NEXT: (S) 0 new block args
// CHECK-NEXT: (S) 1 promoted slots
func.func @basic() -> i32 {
  %0 = arith.constant 5 : i32
  %1 = memref.alloca() : memref<i32>
  memref.store %0, %1[] : memref<i32>
  %2 = memref.load %1[] : memref<i32>
  return %2 : i32
}

// -----

// CHECK: Mem2Reg
// CHECK-NEXT: (S) 0 new block args
// CHECK-NEXT: (S) 0 promoted slots
func.func @no_alloca() -> i32 {
  %0 = arith.constant 5 : i32
  return %0 : i32
}

// -----

// CHECK: Mem2Reg
// CHECK-NEXT: (S) 2 new block args
// CHECK-NEXT: (S) 1 promoted slots
func.func @cycle(%arg0: i64, %arg1: i1, %arg2: i64) {
  %alloca = memref.alloca() : memref<i64>
  memref.store %arg2, %alloca[] : memref<i64>
  cf.cond_br %arg1, ^bb1, ^bb2
^bb1:
  %use = memref.load %alloca[] : memref<i64>
  call @use(%use) : (i64) -> ()
  memref.store %arg0, %alloca[] : memref<i64>
  cf.br ^bb2
^bb2:
  cf.br ^bb1
}

func.func @use(%arg: i64) { return }

// -----

// CHECK: Mem2Reg
// CHECK-NEXT: (S) 0 new block args
// CHECK-NEXT: (S) 3 promoted slots
func.func @recursive(%arg: i64) -> i64 {
  %alloca0 = memref.alloca() : memref<memref<memref<i64>>>
  %alloca1 = memref.alloca() : memref<memref<i64>>
  %alloca2 = memref.alloca() : memref<i64>
  memref.store %arg, %alloca2[] : memref<i64>
  memref.store %alloca2, %alloca1[] : memref<memref<i64>>
  memref.store %alloca1, %alloca0[] : memref<memref<memref<i64>>>
  %load0 = memref.load %alloca0[] : memref<memref<memref<i64>>>
  %load1 = memref.load %load0[] : memref<memref<i64>>
  %load2 = memref.load %load1[] : memref<i64>
  return %load2 : i64
}
