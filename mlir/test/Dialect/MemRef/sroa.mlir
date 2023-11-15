// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(sroa))" --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @basic
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32)
func.func @basic(%arg0: i32, %arg1: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT: = memref.alloca()
  // CHECK-COUNT-2: = memref.alloca() : memref<i32>
  // CHECK-NOT: = memref.alloca()
  %alloca = memref.alloca() : memref<2xi32>
  // CHECK: memref.store %[[ARG0]], %[[ALLOCA0:.*]][]
  memref.store %arg0, %alloca[%c0] : memref<2xi32>
  // CHECK: memref.store %[[ARG1]], %[[ALLOCA1:.*]][]
  memref.store %arg1, %alloca[%c1] : memref<2xi32>
  // CHECK: %[[RES:.*]] = memref.load %[[ALLOCA0]][]
  %res = memref.load %alloca[%c0] : memref<2xi32>
  // CHECK: return %[[RES]] : i32
  return %res : i32
}

// -----

// CHECK-LABEL: func.func @basic_high_dimensions
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
func.func @basic_high_dimensions(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT: = memref.alloca()
  // CHECK-COUNT-3: = memref.alloca() : memref<i32>
  // CHECK-NOT: = memref.alloca()
  %alloca = memref.alloca() : memref<2x2xi32>
  // CHECK: memref.store %[[ARG0]], %[[ALLOCA0:.*]][]
  memref.store %arg0, %alloca[%c0, %c0] : memref<2x2xi32>
  // CHECK: memref.store %[[ARG1]], %[[ALLOCA1:.*]][]
  memref.store %arg1, %alloca[%c0, %c1] : memref<2x2xi32>
  // CHECK: memref.store %[[ARG2]], %[[ALLOCA2:.*]][]
  memref.store %arg2, %alloca[%c1, %c0] : memref<2x2xi32>
  // CHECK: %[[RES:.*]] = memref.load %[[ALLOCA1]][]
  %res = memref.load %alloca[%c0, %c1] : memref<2x2xi32>
  // CHECK: return %[[RES]] : i32
  return %res : i32
}

// -----

// CHECK-LABEL: func.func @resolve_alias
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32)
func.func @resolve_alias(%arg0: i32, %arg1: i32) -> i32 {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: = memref.alloca()
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<i32>
  // CHECK-NOT: = memref.alloca()
  %alloca = memref.alloca() : memref<2xi32>
  // CHECK: memref.store %[[ARG0]], %[[ALLOCA]][]
  memref.store %arg0, %alloca[%c0] : memref<2xi32>
  // CHECK: memref.store %[[ARG1]], %[[ALLOCA]][]
  memref.store %arg1, %alloca[%c0] : memref<2xi32>
  // CHECK: %[[RES:.*]] = memref.load %[[ALLOCA]][]
  %res = memref.load %alloca[%c0] : memref<2xi32>
  // CHECK: return %[[RES]] : i32
  return %res : i32
}

// -----

// CHECK-LABEL: func.func @no_direct_use
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32)
func.func @no_direct_use(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT: = memref.alloca()
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<2xi32>
  // CHECK-NOT: = memref.alloca()
  %alloca = memref.alloca() : memref<2xi32>
  // CHECK: memref.store %[[ARG0]], %[[ALLOCA]][%[[C0]]]
  memref.store %arg0, %alloca[%c0] : memref<2xi32>
  // CHECK: memref.store %[[ARG1]], %[[ALLOCA]][%[[C1]]]
  memref.store %arg1, %alloca[%c1] : memref<2xi32>
  // CHECK: %[[RES:.*]] = memref.load %[[ALLOCA]][%[[C0]]]
  %res = memref.load %alloca[%c0] : memref<2xi32>
  call @use(%alloca) : (memref<2xi32>) -> ()
  // CHECK: return %[[RES]] : i32
  return %res : i32
}

func.func @use(%foo: memref<2xi32>) { return }

// -----

// CHECK-LABEL: func.func @no_dynamic_indexing
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[INDEX:.*]]: index)
func.func @no_dynamic_indexing(%arg0: i32, %arg1: i32, %index: index) -> i32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NOT: = memref.alloca()
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<2xi32>
  // CHECK-NOT: = memref.alloca()
  %alloca = memref.alloca() : memref<2xi32>
  // CHECK: memref.store %[[ARG0]], %[[ALLOCA]][%[[C0]]]
  memref.store %arg0, %alloca[%c0] : memref<2xi32>
  // CHECK: memref.store %[[ARG1]], %[[ALLOCA]][%[[INDEX]]]
  memref.store %arg1, %alloca[%index] : memref<2xi32>
  // CHECK: %[[RES:.*]] = memref.load %[[ALLOCA]][%[[C0]]]
  %res = memref.load %alloca[%c0] : memref<2xi32>
  // CHECK: return %[[RES]] : i32
  return %res : i32
}

// -----

// CHECK-LABEL: func.func @no_dynamic_shape
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32)
func.func @no_dynamic_shape(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT: = memref.alloca()
  // CHECK: %[[ALLOCA:.*]] = memref.alloca(%[[C1]]) : memref<?x2xi32>
  // CHECK-NOT: = memref.alloca()
  %alloca = memref.alloca(%c1) : memref<?x2xi32>
  // CHECK: memref.store %[[ARG0]], %[[ALLOCA]][%[[C0]], %[[C0]]]
  memref.store %arg0, %alloca[%c0, %c0] : memref<?x2xi32>
  // CHECK: %[[RES:.*]] = memref.load %[[ALLOCA]][%[[C0]], %[[C0]]]
  %res = memref.load %alloca[%c0, %c0] : memref<?x2xi32>
  // CHECK: return %[[RES]] : i32
  return %res : i32
}

// -----

// CHECK-LABEL: func.func @no_out_of_bound_write
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32)
func.func @no_out_of_bound_write(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK: %[[C100:.*]] = arith.constant 100 : index
  %c100 = arith.constant 100 : index
  // CHECK-NOT: = memref.alloca()
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<2xi32>
  // CHECK-NOT: = memref.alloca()
  %alloca = memref.alloca() : memref<2xi32>
  // CHECK: memref.store %[[ARG0]], %[[ALLOCA]][%[[C0]]]
  memref.store %arg0, %alloca[%c0] : memref<2xi32>
  // CHECK: memref.store %[[ARG1]], %[[ALLOCA]][%[[C100]]]
  memref.store %arg1, %alloca[%c100] : memref<2xi32>
  // CHECK: %[[RES:.*]] = memref.load %[[ALLOCA]][%[[C0]]]
  %res = memref.load %alloca[%c0] : memref<2xi32>
  // CHECK: return %[[RES]] : i32
  return %res : i32
}

// -----

// CHECK-LABEL: func.func @no_out_of_bound_load
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32)
func.func @no_out_of_bound_load(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK: %[[C100:.*]] = arith.constant 100 : index
  %c100 = arith.constant 100 : index
  // CHECK-NOT: = memref.alloca()
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<2xi32>
  // CHECK-NOT: = memref.alloca()
  %alloca = memref.alloca() : memref<2xi32>
  // CHECK: memref.store %[[ARG0]], %[[ALLOCA]][%[[C0]]]
  memref.store %arg0, %alloca[%c0] : memref<2xi32>
  // CHECK: %[[RES:.*]] = memref.load %[[ALLOCA]][%[[C100]]]
  %res = memref.load %alloca[%c100] : memref<2xi32>
  // CHECK: return %[[RES]] : i32
  return %res : i32
}
