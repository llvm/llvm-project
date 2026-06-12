// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries" -split-input-file | FileCheck %s
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries test-analysis-only" -split-input-file | FileCheck %s -check-prefix=CHECK-ANALYSIS

// CHECK-LABEL: func @mask(
//  CHECK-SAME:     %[[t0:.*]]: memref<?xf32, strided<[?], offset: ?>>
func.func @mask(%t0: tensor<?xf32>, %val: vector<16xf32>, %idx: index, %m0: vector<16xi1>) -> tensor<?xf32> {
  // CHECK-NOT: alloc
  // CHECK-NOT: copy
  //     CHECK: vector.mask %{{.*}} { vector.transfer_write %{{.*}}, %[[t0]][%{{.*}}] : vector<16xf32>, memref<?xf32, strided<[?], offset: ?>> } : vector<16xi1>
  %0 = vector.mask %m0 { vector.transfer_write %val, %t0[%idx] : vector<16xf32>, tensor<?xf32> } : vector<16xi1> -> tensor<?xf32>
  //     CHECK: return %[[t0]]
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @mask_scalable(
//  CHECK-SAME:     %[[t0:.*]]: memref<?xf32, strided<[?], offset: ?>>
func.func @mask_scalable(%t0: tensor<?xf32>, %val: vector<[16]xf32>, %idx: index, %m0: vector<[16]xi1>) -> tensor<?xf32> {
  // CHECK-NOT: alloc
  // CHECK-NOT: copy
  //     CHECK: vector.mask %{{.*}} { vector.transfer_write %{{.*}}, %[[t0]][%{{.*}}] : vector<[16]xf32>, memref<?xf32, strided<[?], offset: ?>> } : vector<[16]xi1>
  %0 = vector.mask %m0 { vector.transfer_write %val, %t0[%idx] : vector<[16]xf32>, tensor<?xf32> } : vector<[16]xi1> -> tensor<?xf32>
  //     CHECK: return %[[t0]]
  return %0 : tensor<?xf32>
}

// -----

// CHECK-ANALYSIS-LABEL: func @non_reading_xfer_write(
//  CHECK-ANALYSIS-SAME: tensor<5x10xf32> {bufferization.access = "write"}
func.func @non_reading_xfer_write(%t: tensor<5x10xf32>, %v: vector<6x11xf32>) -> tensor<5x10xf32> {
  %c0 = arith.constant 0 : index
  %1 = vector.transfer_write %v, %t[%c0, %c0] : vector<6x11xf32>, tensor<5x10xf32>
  return %1 : tensor<5x10xf32>
}
// -----

// CHECK-ANALYSIS-LABEL: func @reading_xfer_write(
//  CHECK-ANALYSIS-SAME: tensor<5x10xf32> {bufferization.access = "read-write"}
func.func @reading_xfer_write(%t: tensor<5x10xf32>, %v: vector<4x11xf32>) -> tensor<5x10xf32> {
  %c0 = arith.constant 0 : index
  %1 = vector.transfer_write %v, %t[%c0, %c0] : vector<4x11xf32>, tensor<5x10xf32>
  return %1 : tensor<5x10xf32>
}
