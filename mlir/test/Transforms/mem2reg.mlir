// RUN: mlir-opt %s --pass-pipeline='builtin.module(any(mem2reg))' --split-input-file | FileCheck %s

// Verifies that allocators with mutliple slots are handled properly.

// CHECK-LABEL: func.func @multi_slot_alloca
func.func @multi_slot_alloca() -> (i32, i32) {
  // CHECK-NOT: test.multi_slot_alloca
  %1, %2 = test.multi_slot_alloca : () -> (memref<i32>, memref<i32>)
  %3 = memref.load %1[] : memref<i32>
  %4 = memref.load %2[] : memref<i32>
  return %3, %4 : i32, i32
}

// -----

// Verifies that a multi slot allocator can be partially promoted.

func.func private @consumer(memref<i32>)

// CHECK-LABEL: func.func @multi_slot_alloca_only_second
func.func @multi_slot_alloca_only_second() -> (i32, i32) {
  // CHECK: %{{[[:alnum:]]+}} = test.multi_slot_alloca
  %1, %2 = test.multi_slot_alloca : () -> (memref<i32>, memref<i32>)
  func.call @consumer(%1) : (memref<i32>) -> ()
  %3 = memref.load %1[] : memref<i32>
  %4 = memref.load %2[] : memref<i32>
  return %3, %4 : i32, i32
}

// -----

// Checks that slots are not promoted if used in a graph region.

// CHECK-LABEL: test.isolated_graph_region
test.isolated_graph_region {
  // CHECK: %{{[[:alnum:]]+}} = test.multi_slot_alloca
  %slot = test.multi_slot_alloca : () -> (memref<i32>)
  memref.store %a, %slot[] : memref<i32>
  %a = memref.load %slot[] : memref<i32>
  "test.foo"() : () -> ()
}
