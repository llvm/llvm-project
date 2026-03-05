// RUN: mlir-opt %s --pass-pipeline='builtin.module(any(mem2reg))' --split-input-file | FileCheck %s

// Verifies that allocators with multiple slots are handled properly.

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

// -----

// Verifies that block arguments of merge points are not abusively treated as
// the newly created block arguments. Here, ^merge has a pre-existing block
// argument (%genuine) and mem2reg adds a second one for the promoted slot. The
// slot arg then serves as the reaching definition for the follow-up merge point
// ^final. If the unused merge point propagation logic identified merge points
// by block rather than by specific block argument, it would confuse %genuine
// for the slot argument to be removed and thus not eliminate the slot which
// is unused. In other words, the genuine block argument, which is used, would
// mask that the actual slot argument is unused.

// CHECK-LABEL: func.func @merge_point_arg_not_confused
// CHECK-SAME: (%[[COND:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32) -> i32
// CHECK: cf.cond_br %[[COND]], ^[[BB1:.*]], ^[[BB2:.*]]
// CHECK: ^[[BB1]]:
// CHECK:   cf.br ^[[MERGE:.*]](%[[A]] : i32)
// CHECK: ^[[BB2]]:
// CHECK:   cf.br ^[[MERGE]](%[[B]] : i32)
// CHECK: ^[[MERGE]](%[[GENUINE:.*]]: i32):
// CHECK:   cf.cond_br %[[COND]], ^[[BB3:.*]], ^[[BB4:.*]]
// CHECK: ^[[BB3]]:
// CHECK:   cf.br ^[[FINAL:.*]](%[[GENUINE]] : i32)
// CHECK: ^[[BB4]]:
// CHECK:   %[[DUMMY:.*]] = arith.constant 0 : i32
// CHECK:   cf.br ^[[FINAL]](%[[DUMMY]] : i32)
// CHECK: ^[[FINAL]](%[[FINAL_SLOT:.*]]: i32):
// CHECK:   return %[[FINAL_SLOT]] : i32
func.func @merge_point_arg_not_confused(%cond: i1, %a: i32, %b: i32) -> i32 {
  %alloca = memref.alloca() : memref<i32>
  memref.store %a, %alloca[] : memref<i32>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  memref.store %b, %alloca[] : memref<i32>
  cf.br ^merge(%a : i32)
^bb2:
  cf.br ^merge(%b : i32)
^merge(%genuine: i32):
  cf.cond_br %cond, ^bb3, ^bb4
^bb3:
  memref.store %genuine, %alloca[] : memref<i32>
  cf.br ^final
^bb4:
  %dummy = arith.constant 0 : i32
  memref.store %dummy, %alloca[] : memref<i32>
  cf.br ^final
^final:
  %load = memref.load %alloca[] : memref<i32>
  return %load : i32
}
