// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(cse))' | FileCheck %s

// Verify that acc.set (which writes CurrentDeviceIdResource, a root disjoint
// from DefaultResource) does not block CSE of identical memref.load operations
// (which read DefaultResource). The two resources are disjoint, so the write
// cannot conflict with the loads.

// CHECK-LABEL: @cse_across_acc_set
func.func @cse_across_acc_set(%a: memref<10xf32>, %i: index) -> (f32, f32) {
  %v1 = memref.load %a[%i] : memref<10xf32>
  %c42 = arith.constant 42 : i32
  acc.set device_num(%c42 : i32)
  %v2 = memref.load %a[%i] : memref<10xf32>
  // CHECK: %[[V:.*]] = memref.load
  // CHECK-NOT: memref.load
  // CHECK: return %[[V]], %[[V]] : f32, f32
  return %v1, %v2 : f32, f32
}
