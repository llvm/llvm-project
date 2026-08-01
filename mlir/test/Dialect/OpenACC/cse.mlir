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

// -----

// Two identical acc.gpu_shared_memory ops must not be CSE'd: each reserves a
// distinct workgroup-memory slot.
// CHECK-LABEL: @cse_gpu_shared_memory_not_merged
func.func @cse_gpu_shared_memory_not_merged() -> (memref<8xf32, #gpu.address_space<workgroup>>, memref<8xf32, #gpu.address_space<workgroup>>) {
  %0 = acc.gpu_shared_memory {num_copies = 1 : i64, static_upper_bound_bytes = 256 : i64}
      : () -> memref<8xf32, #gpu.address_space<workgroup>>
  %1 = acc.gpu_shared_memory {num_copies = 1 : i64, static_upper_bound_bytes = 256 : i64}
      : () -> memref<8xf32, #gpu.address_space<workgroup>>
  // CHECK: acc.gpu_shared_memory
  // CHECK: acc.gpu_shared_memory
  return %0, %1 : memref<8xf32, #gpu.address_space<workgroup>>, memref<8xf32, #gpu.address_space<workgroup>>
}
