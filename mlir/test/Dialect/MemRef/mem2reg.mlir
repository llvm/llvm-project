// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(mem2reg{region-simplify=false}))' --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @basic
func.func @basic() -> i32 {
  // CHECK-NOT: = memref.alloca
  // CHECK: %[[RES:.*]] = arith.constant 5 : i32
  // CHECK-NOT: = memref.alloca
  %0 = arith.constant 5 : i32
  %1 = memref.alloca() : memref<i32>
  memref.store %0, %1[] : memref<i32>
  %2 = memref.load %1[] : memref<i32>
  // CHECK: return %[[RES]] : i32
  return %2 : i32
}

// -----

// CHECK-LABEL: func.func @basic_default
func.func @basic_default() -> i32 {
  // CHECK-NOT: = memref.alloca
  // CHECK: %[[RES:.*]] = arith.constant 0 : i32
  // CHECK-NOT: = memref.alloca
  %0 = arith.constant 5 : i32
  %1 = memref.alloca() : memref<i32>
  %2 = memref.load %1[] : memref<i32>
  // CHECK-NOT: memref.store
  memref.store %0, %1[] : memref<i32>
  // CHECK: return %[[RES]] : i32
  return %2 : i32
}

// -----

// CHECK-LABEL: func.func @basic_float
func.func @basic_float() -> f32 {
  // CHECK-NOT: = memref.alloca
  // CHECK: %[[RES:.*]] = arith.constant {{.*}} : f32
  %0 = arith.constant 5.2 : f32
  // CHECK-NOT: = memref.alloca
  %1 = memref.alloca() : memref<f32>
  memref.store %0, %1[] : memref<f32>
  %2 = memref.load %1[] : memref<f32>
  // CHECK: return %[[RES]] : f32
  return %2 : f32
}

// -----

// CHECK-LABEL: func.func @basic_ranked
func.func @basic_ranked() -> i32 {
  // CHECK-NOT: = memref.alloca
  // CHECK: %[[RES:.*]] = arith.constant 5 : i32
  // CHECK-NOT: = memref.alloca
  %0 = arith.constant 0 : index
  %1 = arith.constant 5 : i32
  %2 = memref.alloca() : memref<1x1xi32>
  memref.store %1, %2[%0, %0] : memref<1x1xi32>
  %3 = memref.load %2[%0, %0] : memref<1x1xi32>
  // CHECK: return %[[RES]] : i32
  return %3 : i32
}

// -----

// CHECK-LABEL: func.func @reject_multiple_elements
func.func @reject_multiple_elements() -> i32 {
  // CHECK: %[[INDEX:.*]] = arith.constant 0 : index
  %0 = arith.constant 0 : index
  // CHECK: %[[STORED:.*]] = arith.constant 5 : i32
  %1 = arith.constant 5 : i32
  // CHECK: %[[ALLOCA:.*]] = memref.alloca()
  %2 = memref.alloca() : memref<1x2xi32>
  // CHECK: memref.store %[[STORED]], %[[ALLOCA]][%[[INDEX]], %[[INDEX]]]
  memref.store %1, %2[%0, %0] : memref<1x2xi32>
  // CHECK: %[[RES:.*]] = memref.load %[[ALLOCA]][%[[INDEX]], %[[INDEX]]]
  %3 = memref.load %2[%0, %0] : memref<1x2xi32>
  // CHECK: return %[[RES]] : i32
  return %3 : i32
}

// -----

// CHECK-LABEL: func.func @cycle
// CHECK-SAME: (%[[ARG0:.*]]: i64, %[[ARG1:.*]]: i1, %[[ARG2:.*]]: i64)
func.func @cycle(%arg0: i64, %arg1: i1, %arg2: i64) {
  // CHECK-NOT: = memref.alloca
  %alloca = memref.alloca() : memref<i64>
  memref.store %arg2, %alloca[] : memref<i64>
  // CHECK: cf.cond_br %[[ARG1:.*]], ^[[BB1:.*]](%[[ARG2]] : i64), ^[[BB2:.*]](%[[ARG2]] : i64)
  cf.cond_br %arg1, ^bb1, ^bb2
// CHECK: ^[[BB1]](%[[USE:.*]]: i64):
^bb1:
  %use = memref.load %alloca[] : memref<i64>
  // CHECK: call @use(%[[USE]])
  func.call @use(%use) : (i64) -> ()
  memref.store %arg0, %alloca[] : memref<i64>
  // CHECK: cf.br ^[[BB2]](%[[ARG0]] : i64)
  cf.br ^bb2
// CHECK: ^[[BB2]](%[[FWD:.*]]: i64):
^bb2:
  // CHECK: cf.br ^[[BB1]](%[[FWD]] : i64)
  cf.br ^bb1
}

func.func @use(%arg: i64) { return }

// -----

// CHECK-LABEL: func.func @recursive
// CHECK-SAME: (%[[ARG:.*]]: i64)
func.func @recursive(%arg: i64) -> i64 {
  // CHECK-NOT: = memref.alloca()
  %alloca0 = memref.alloca() : memref<memref<memref<i64>>>
  %alloca1 = memref.alloca() : memref<memref<i64>>
  %alloca2 = memref.alloca() : memref<i64>
  memref.store %arg, %alloca2[] : memref<i64>
  memref.store %alloca2, %alloca1[] : memref<memref<i64>>
  memref.store %alloca1, %alloca0[] : memref<memref<memref<i64>>>
  %load0 = memref.load %alloca0[] : memref<memref<memref<i64>>>
  %load1 = memref.load %load0[] : memref<memref<i64>>
  %load2 = memref.load %load1[] : memref<i64>
  // CHECK: return %[[ARG]] : i64
  return %load2 : i64
}

// -----

// CHECK-LABEL: func.func @deny_store_of_alloca
// CHECK-SAME: (%[[ARG:.*]]: memref<memref<i32>>)
func.func @deny_store_of_alloca(%arg: memref<memref<i32>>) -> i32 {
  // CHECK: %[[VALUE:.*]] = arith.constant 5 : i32
  %0 = arith.constant 5 : i32
  // CHECK: %[[ALLOCA:.*]] = memref.alloca
  %1 = memref.alloca() : memref<i32>
  // Storing into the memref is allowed.
  // CHECK: memref.store %[[VALUE]], %[[ALLOCA]][]
  memref.store %0, %1[] : memref<i32>
  // Storing the memref itself is NOT allowed.
  // CHECK: memref.store %[[ALLOCA]], %[[ARG]][]
  memref.store %1, %arg[] : memref<memref<i32>>
  // CHECK: %[[RES:.*]] = memref.load %[[ALLOCA]][]
  %2 = memref.load %1[] : memref<i32>
  // CHECK: return %[[RES]] : i32
  return %2 : i32
}

// -----

// CHECK-LABEL: func.func @promotable_nonpromotable_intertwined
func.func @promotable_nonpromotable_intertwined() -> i32 {
  // CHECK: %[[NON_PROMOTED:.*]] = memref.alloca() : memref<i32>
  %0 = memref.alloca() : memref<i32>
  // CHECK-NOT: = memref.alloca() : memref<memref<i32>>
  %1 = memref.alloca() : memref<memref<i32>>
  memref.store %0, %1[] : memref<memref<i32>>
  %2 = memref.load %1[] : memref<memref<i32>>
  // CHECK: call @use(%[[NON_PROMOTED]])
  call @use(%0) : (memref<i32>) -> ()
  // CHECK: %[[RES:.*]] = memref.load %[[NON_PROMOTED]][]
  %3 = memref.load %0[] : memref<i32>
  // CHECK: return %[[RES]] : i32
  return %3 : i32
}

func.func @use(%arg: memref<i32>) { return }
