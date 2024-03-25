// RUN: mlir-opt --test-gpu-rewrite -split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @globalId
  // CHECK-SAME: (%[[SZ:.*]]: index, %[[MEM:.*]]: memref<index, 1>) {
  func.func @globalId(%sz : index, %mem: memref<index, 1>) {
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %sz, %grid_y = %sz, %grid_z = %sz)
               threads(%tx, %ty, %tz) in (%block_x = %sz, %block_y = %sz, %block_z = %sz) {
      // CHECK: %[[BIDY:.*]] = gpu.block_id x
      // CHECK-NEXT: %[[BDIMY:.*]] = gpu.block_dim x
      // CHECK-NEXT: %[[TMPY:.*]] = index.mul %[[BIDY]], %[[BDIMY]]
      // CHECK-NEXT: %[[TIDX:.*]] = gpu.thread_id x
      // CHECK-NEXT: %[[GIDX:.*]] = index.add %[[TIDX]], %[[TMPY]]
      %idx = gpu.global_id x
      // CHECK: memref.store %[[GIDX]], %[[MEM]][] : memref<index, 1>
      memref.store %idx, %mem[] : memref<index, 1>
  
      // CHECK: %[[BIDY:.*]] = gpu.block_id y
      // CHECK-NEXT: %[[BDIMY:.*]] = gpu.block_dim y
      // CHECK-NEXT: %[[TMPY:.*]] = index.mul %[[BIDY]], %[[BDIMY]]
      // CHECK-NEXT: %[[TIDY:.*]] = gpu.thread_id y
      // CHECK-NEXT: %[[GIDY:.*]] = index.add %[[TIDY]], %[[TMPY]]
      %idy = gpu.global_id y
      // CHECK: memref.store %[[GIDY]], %[[MEM]][] : memref<index, 1>
      memref.store %idy, %mem[] : memref<index, 1>
  
      // CHECK: %[[BIDZ:.*]] = gpu.block_id z
      // CHECK-NEXT: %[[BDIMZ:.*]] = gpu.block_dim z
      // CHECK-NEXT: %[[TMPZ:.*]] = index.mul %[[BIDZ]], %[[BDIMZ]]
      // CHECK-NEXT: %[[TIDZ:.*]] = gpu.thread_id z
      // CHECK-NEXT: %[[GIDZ:.*]] = index.add %[[TIDZ]], %[[TMPZ]]
      %idz = gpu.global_id z
      // CHECK: memref.store %[[GIDZ]], %[[MEM]][] : memref<index, 1>
      memref.store %idz, %mem[] : memref<index, 1>
      gpu.terminator
    }
    return
  }
}
