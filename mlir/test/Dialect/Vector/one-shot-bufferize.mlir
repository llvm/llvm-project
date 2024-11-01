// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries" -split-input-file | FileCheck %s

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
