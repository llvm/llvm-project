// RUN: mlir-opt --test-gpu-rewrite -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @subgroupId
// CHECK-SAME: (%[[SZ:.*]]: index, %[[MEM:.*]]: memref<index, 1>) {
func.func @subgroupId(%sz : index, %mem: memref<index, 1>) {
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %sz, %grid_y = %sz, %grid_z = %sz)
             threads(%tx, %ty, %tz) in (%block_x = %sz, %block_y = %sz, %block_z = %sz) {
    // CHECK: %[[DIMX:.*]] = gpu.block_dim  x
    // CHECK-NEXT: %[[DIMY:.*]] = gpu.block_dim  y
    // CHECK-NEXT: %[[TIDX:.*]] = gpu.thread_id  x
    // CHECK-NEXT: %[[TIDY:.*]] = gpu.thread_id  y
    // CHECK-NEXT: %[[TIDZ:.*]] = gpu.thread_id  z
    // CHECK-NEXT: %[[T0:.*]] = arith.muli %[[DIMY]], %[[TIDZ]] : index
    // CHECK-NEXT: %[[T1:.*]] = arith.addi %[[T0]], %[[TIDY]] : index
    // CHECK-NEXT: %[[T2:.*]] = arith.muli %[[DIMX]], %[[T1]] : index
    // CHECK-NEXT: %[[T3:.*]] = arith.addi %[[TIDX]], %[[T2]] : index
    // CHECK-NEXT: %[[T4:.*]] = gpu.subgroup_size : index
    // CHECK-NEXT: %[[T5:.*]] = arith.divui %[[T3]], %[[T4]] : index
    %idz = gpu.subgroup_id : index
    memref.store %idz, %mem[] : memref<index, 1>
    gpu.terminator
  }
  return
}
