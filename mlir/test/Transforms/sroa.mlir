// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(sroa))' --split-input-file | FileCheck %s

// Verifies that allocators with multiple slots are handled properly.

// CHECK-LABEL: func.func @multi_slot_alloca
func.func @multi_slot_alloca() -> (i32, i32) {
  %0 = arith.constant 0 : index
  %1, %2 = test.multi_slot_alloca : () -> (memref<2xi32>, memref<4xi32>)
  // CHECK-COUNT-2: test.multi_slot_alloca : () -> memref<i32>
  %3 = memref.load %1[%0] {first}: memref<2xi32>
  %4 = memref.load %2[%0] {second} : memref<4xi32>
  return %3, %4 : i32, i32
}

// -----

// Verifies that a multi slot allocator can be partially destructured.

func.func private @consumer(memref<2xi32>)

// CHECK-LABEL: func.func @multi_slot_alloca_only_second
func.func @multi_slot_alloca_only_second() -> (i32, i32) {
  %0 = arith.constant 0 : index
  // CHECK: test.multi_slot_alloca : () -> memref<2xi32>
  // CHECK: test.multi_slot_alloca : () -> memref<i32>
  %1, %2 = test.multi_slot_alloca : () -> (memref<2xi32>, memref<4xi32>)
  func.call @consumer(%1) : (memref<2xi32>) -> ()
  %3 = memref.load %1[%0] : memref<2xi32>
  %4 = memref.load %2[%0] : memref<4xi32>
  return %3, %4 : i32, i32
}
