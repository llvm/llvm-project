// RUN: mlir-opt %s -allow-unregistered-dialect -test-linalg-transform-patterns=test-vector-transfer-forwarding-patterns | FileCheck %s

// CHECK-LABEL: testAllocRead
//  CHECK-SAME: %[[ARG0:[0-9a-zA-Z]*]]: memref
//   CHECK-NOT: linalg.fill
//   CHECK-NOT: linalg.copy
//       CHECK: %[[ALLOC:.*]] = memref.alloc
//       CHECK: vector.transfer_read %[[ARG0]]
//   CHECK-NOT: in_bounds
func @testAllocRead(%in: memref<? x f32>) -> vector<32 x f32> {
  %c0 = constant 0: index
  %f0 = constant 0.0: f32
  %alloc = memref.alloc() : memref<32 x f32>
  %subview = memref.subview %alloc[0][16][1] : memref<32 x f32> to memref<16 x f32>
  linalg.copy(%in, %subview): memref<? x f32>, memref<16 x f32>
  %0 = vector.transfer_read %alloc[%c0], %f0 {in_bounds = [true]} : memref<32 x f32>, vector<32 x f32>
  memref.dealloc %alloc : memref<32 x f32>
  return %0: vector<32 x f32>
}

// CHECK-LABEL: testAllocFillRead
//  CHECK-SAME: %[[ARG0:[0-9a-zA-Z]*]]: memref
//   CHECK-NOT: linalg.fill
//   CHECK-NOT: linalg.copy
//       CHECK: %[[ALLOC:.*]] = memref.alloc
//       CHECK: vector.transfer_read %[[ARG0]]
//   CHECK-NOT: in_bounds
func @testAllocFillRead(%in: memref<? x f32>) -> vector<32 x f32> {
  %c0 = constant 0: index
  %f0 = constant 0.0: f32
  %alloc = memref.alloc() : memref<32 x f32>
  linalg.fill(%alloc, %f0): memref<32 x f32>, f32
  %subview = memref.subview %alloc[0][16][1] : memref<32 x f32> to memref<16 x f32>
  linalg.copy(%in, %subview): memref<? x f32>, memref<16 x f32>
  %0 = vector.transfer_read %alloc[%c0], %f0 {in_bounds = [true]} : memref<32 x f32>, vector<32 x f32>
  memref.dealloc %alloc : memref<32 x f32>
  return %0: vector<32 x f32>
}

// CHECK-LABEL: testViewRead
//  CHECK-SAME: %[[ARG0:[0-9a-zA-Z]*]]: memref
//   CHECK-NOT: linalg.fill
//   CHECK-NOT: linalg.copy
//       CHECK: %[[ALLOC:.*]] = memref.alloc
//       CHECK: vector.transfer_read %[[ARG0]]
//   CHECK-NOT: in_bounds
func @testViewRead(%in: memref<? x f32>) -> vector<32 x f32> {
  %c0 = constant 0: index
  %f0 = constant 0.0: f32
  %alloc = memref.alloc() : memref<128 x i8>
  %view = memref.view %alloc[%c0][] : memref<128 x i8> to memref<32 x f32>
  %subview = memref.subview %view[0][16][1] : memref<32 x f32> to memref<16 x f32>
  linalg.copy(%in, %subview): memref<? x f32>, memref<16 x f32>
  %0 = vector.transfer_read %view[%c0], %f0 {in_bounds = [true]} : memref<32 x f32>, vector<32 x f32>
  memref.dealloc %alloc : memref<128 x i8>
  return %0: vector<32 x f32>
}

// CHECK-LABEL: testViewFillRead
//  CHECK-SAME: %[[ARG0:[0-9a-zA-Z]*]]: memref
//   CHECK-NOT: linalg.fill
//   CHECK-NOT: linalg.copy
//       CHECK: %[[ALLOC:.*]] = memref.alloc
//       CHECK: vector.transfer_read %[[ARG0]]
//   CHECK-NOT: in_bounds
func @testViewFillRead(%in: memref<? x f32>) -> vector<32 x f32> {
  %c0 = constant 0: index
  %f0 = constant 0.0: f32
  %alloc = memref.alloc() : memref<128 x i8>
  %view = memref.view %alloc[%c0][] : memref<128 x i8> to memref<32 x f32>
  %subview = memref.subview %view[0][16][1] : memref<32 x f32> to memref<16 x f32>
  linalg.fill(%view, %f0): memref<32 x f32>, f32
  linalg.copy(%in, %subview): memref<? x f32>, memref<16 x f32>
  %0 = vector.transfer_read %view[%c0], %f0 {in_bounds = [true]} : memref<32 x f32>, vector<32 x f32>
  memref.dealloc %alloc : memref<128 x i8>
  return %0: vector<32 x f32>
}

// CHECK-LABEL: testAllocWrite
//  CHECK-SAME: %[[ARG0:[0-9a-zA-Z]*]]: vector
//  CHECK-SAME: %[[ARG1:[0-9a-zA-Z]*]]: memref
//   CHECK-NOT: linalg.copy
//       CHECK: %[[ALLOC:.*]] = memref.alloc
//       CHECK: vector.transfer_write %[[ARG0]], %[[ARG1]]
//   CHECK-NOT: in_bounds
func @testAllocWrite(%vec: vector<32 x f32>, %out: memref<? x f32>) {
  %c0 = constant 0: index
  %f0 = constant 0.0: f32
  %alloc = memref.alloc() : memref<32 x f32>
  %subview = memref.subview %alloc[0][16][1] : memref<32 x f32> to memref<16 x f32>
  vector.transfer_write %vec, %alloc[%c0] {in_bounds = [true]} : vector<32 x f32>, memref<32 x f32>
  linalg.copy(%subview, %out): memref<16 x f32>, memref<? x f32>
  memref.dealloc %alloc : memref<32 x f32>
  return
}

// CHECK-LABEL: testViewWrite
//  CHECK-SAME: %[[ARG0:[0-9a-zA-Z]*]]: vector
//  CHECK-SAME: %[[ARG1:[0-9a-zA-Z]*]]: memref
//   CHECK-NOT: linalg.copy
//       CHECK: %[[ALLOC:.*]] = memref.alloc
//       CHECK: vector.transfer_write %[[ARG0]], %[[ARG1]]
//   CHECK-NOT: in_bounds
func @testViewWrite(%vec: vector<32 x f32>, %out: memref<? x f32>) {
  %c0 = constant 0: index
  %f0 = constant 0.0: f32
  %alloc = memref.alloc() : memref<128 x i8>
  %view = memref.view %alloc[%c0][] : memref<128 x i8> to memref<32 x f32>
  %subview = memref.subview %view[0][16][1] : memref<32 x f32> to memref<16 x f32>
  vector.transfer_write %vec, %view[%c0] {in_bounds = [true]} : vector<32 x f32>, memref<32 x f32>
  linalg.copy(%subview, %out): memref<16 x f32>, memref<? x f32>
  memref.dealloc %alloc : memref<128 x i8>
  return
}

///===--------------------------------------------------------------------===///
// Negative tests
///===--------------------------------------------------------------------===///

// This should fail the rewrite due to mismatching fill and transfer read value.
// CHECK-LABEL: failAllocFillRead
//  CHECK-SAME: %[[ARG0:[0-9a-zA-Z]*]]: memref
//   CHECK-NOT: vector.transfer_read %[[ARG0]]
//       CHECK: %[[ALLOC:.*]] = memref.alloc
//       CHECK: linalg.copy
//       CHECK: vector.transfer_read %[[ALLOC]]
func @failAllocFillRead(%in: memref<? x f32>) -> vector<32 x f32> {
  %c0 = constant 0: index
  %f0 = constant 0.0: f32
  %f1 = constant 1.0: f32
  %alloc = memref.alloc() : memref<32 x f32>
  linalg.fill(%alloc, %f0): memref<32 x f32>, f32
  %subview = memref.subview %alloc[0][16][1] : memref<32 x f32> to memref<16 x f32>
  linalg.copy(%in, %subview): memref<? x f32>, memref<16 x f32>
  "some_interleaved_use"(%subview) : (memref<16 x f32>) -> ()
  %0 = vector.transfer_read %alloc[%c0], %f1: memref<32 x f32>, vector<32 x f32>
  memref.dealloc %alloc : memref<32 x f32>
  return %0: vector<32 x f32>
}

// This should fail the rewrite due to some interleaved use.
// CHECK-LABEL: failAllocWrite
//  CHECK-SAME: %[[ARG0:[0-9a-zA-Z]*]]: vector
//  CHECK-SAME: %[[ARG1:[0-9a-zA-Z]*]]: memref
//   CHECK-NOT: vector.transfer_write %[[ARG0]], %[[ARG1]]
//       CHECK: %[[ALLOC:.*]] = memref.alloc
//       CHECK: vector.transfer_write %[[ARG0]], %[[ALLOC]]
//       CHECK: linalg.copy
func @failAllocWrite(%vec: vector<32 x f32>, %out: memref<? x f32>) {
  %c0 = constant 0: index
  %f0 = constant 0.0: f32
  %alloc = memref.alloc() : memref<32 x f32>
  %subview = memref.subview %alloc[0][16][1] : memref<32 x f32> to memref<16 x f32>
  vector.transfer_write %vec, %alloc[%c0] : vector<32 x f32>, memref<32 x f32>
  "some_interleaved_use"(%subview) : (memref<16 x f32>) -> ()
  linalg.copy(%subview, %out): memref<16 x f32>, memref<? x f32>
  memref.dealloc %alloc : memref<32 x f32>
  return
}
