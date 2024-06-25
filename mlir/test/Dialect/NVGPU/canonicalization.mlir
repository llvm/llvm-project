// RUN: mlir-opt %s | mlir-opt -canonicalize -cse | FileCheck %s

gpu.module @main_kernel {

// CHECK-LABEL: @main_kernel(
//  CHECK-SAME: %[[arg0:.*]]: !nvgpu.tensormap.descriptor
  gpu.func @main_kernel(%arg0: !nvgpu.tensormap.descriptor<
        tensor = memref<128x32xf32, 3>, swizzle = none, l2promo = none, 
        oob = zero, interleave = none>) kernel attributes 
        { gpu.known_block_size = array<i32: 128, 1, 1>, 
          gpu.known_grid_size = array<i32: 1, 1, 1>
        } 
  {
    // CHECK: %[[c0:.+]] = arith.constant 0 : index 
    // CHECK: %[[S0:.+]] = gpu.thread_id  x
    // CHECK: %[[S1:.+]] = arith.cmpi eq, %[[S0]], %[[c0]] : index
    // CHECK: %[[S2:.+]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[S3:.+]] = memref.view %[[S2]][%[[c0]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x32xf32, #gpu.address_space<workgroup>>
    // CHECK: nvgpu.tma.async.store %[[S3]] to %[[arg0]][%[[c0]], %[[c0]]], predicate = %[[S1]] : memref<128x32xf32, #gpu.address_space<workgroup>> -> <tensor = memref<128x32xf32, 3>, swizzle = none, l2promo = none, oob = zero, interleave = none>
    %c0 = arith.constant 0 : index
    %0 = gpu.thread_id  x
    %1 = arith.cmpi eq, %0, %c0 : index
    %2 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
    %view = memref.view %2[%c0][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x32xf32, #gpu.address_space<workgroup>>
    nvgpu.tma.async.store %view to %arg0[%c0, %c0], predicate = %1 : memref<128x32xf32, #gpu.address_space<workgroup>> -> <tensor = memref<128x32xf32, 3>, swizzle = none, l2promo = none, oob = zero, interleave = none>
    nvvm.cp.async.bulk.commit.group
    nvvm.cp.async.bulk.wait_group 0
    gpu.return
  }
}