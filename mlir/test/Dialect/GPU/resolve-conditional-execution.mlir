// RUN: mlir-opt %s --gpu-resolve-conditional-execution -split-input-file | FileCheck %s
// RUN: mlir-opt %s --gpu-kernel-outlining --gpu-resolve-conditional-execution -split-input-file | FileCheck --check-prefix=LAUNCH %s

// CHECK-LABEL:func.func @conditional_execution_host
// CHECK: (%[[DEV:.*]]: index, %[[HOST:.*]]: index)
func.func @conditional_execution_host(%dev : index, %host : index) {
  // CHECK: %{{.*}} = scf.execute_region -> index {
  // CHECK-NEXT: scf.yield %[[HOST]] : index
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  // Test that it returns %host.
  %v = gpu.conditional_execution device {
    gpu.yield %dev: index
  } host {
    gpu.yield %host: index
  } -> index
  return
}

// -----

// CHECK-LABEL:func.func @conditional_execution_host
func.func @conditional_execution_host(%memref: memref<f32>) {
  // CHECK-NEXT: return
  // CHECK-NEXT: }
  // Test that the operation gets erased.
  gpu.conditional_execution device {
    %c1 = arith.constant 1.0 : f32
    memref.store %c1, %memref[] : memref<f32>
    gpu.yield
  }
  return
}

// -----

gpu.module @conditional_execution_dev {
// CHECK-LABEL:gpu.func @kernel
// CHECK: (%[[DEV:.*]]: index, %[[HOST:.*]]: index)
  gpu.func @kernel(%dev : index, %host : index) kernel {
    // CHECK: %{{.*}} = scf.execute_region -> index {
    // CHECK-NEXT: scf.yield %[[DEV]] : index
    // CHECK-NEXT: }
    // CHECK-NEXT: return
    // Test that it returns %dev.
    %v = gpu.conditional_execution device {
      gpu.yield %dev: index
    } host {
      gpu.yield %host: index
    } -> index
    gpu.return
  }
}

// -----

// CHECK-LABEL:func.func @conditional_execution_dev
// CHECK: (%[[MEMREF:.*]]: memref<f32>, %[[DEV:.*]]: f32, %[[HOST:.*]]: f32)
func.func @conditional_execution_dev(%memref: memref<f32>, %fdev: f32, %fhost: f32) {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%sbx = %c1, %sby = %c1, %sbz = %c1)
             threads(%tx, %ty, %tz) in (%stx = %c1, %sty = %c1, %stz = %c1) {
    // CHECK: scf.execute_region {
    // CHECK-NEXT: memref.store %[[DEV]], %[[MEMREF]][] : memref<f32>
    // CHECK-NEXT: scf.yield
    // CHECK-NEXT: }
    // CHECK-NEXT: gpu.terminator
    // Test that it uses %fdev.
    gpu.conditional_execution device {
      memref.store %fdev, %memref[] : memref<f32>
      gpu.yield
    } host {
      memref.store %fhost, %memref[] : memref<f32>
      gpu.yield
    }
    gpu.terminator
  }
  return
}

// -----

// LAUNCH-LABEL: func.func @thread_id() -> index
// LAUNCH: %[[HOST_ID:.*]] = arith.constant 0 : index
// LAUNCH-NEXT: %[[HOST_RES:.*]] =  scf.execute_region -> index {
// LAUNCH-NEXT: scf.yield %[[HOST_ID]] : index
// LAUNCH-NEXT: }
// LAUNCH-NEXT: return %[[HOST_RES]] : index
func.func @thread_id() -> index {
  %val = gpu.conditional_execution device {
    %id = gpu.thread_id x
    gpu.yield %id: index
  } host {
    %id = arith.constant 0 : index
    gpu.yield %id: index
  } -> index
  return %val : index
}
// LAUNCH-LABEL: func.func @launch()
// LAUNCH: gpu.launch_func
// LAUNCH-NEXT: %{{.*}} = call @thread_id() : () -> index
// LAUNCH-NEXT: return
func.func @launch() {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1,
                                       %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1,
                                        %block_z = %c1) {
    %id = func.call @thread_id() : () -> index
    gpu.terminator
  }
  %id = func.call @thread_id() : () -> index
  return
}
// LAUNCH: gpu.module @[[LAUNCH_ID:.*]] {
// LAUNCH: gpu.func @[[LAUNCH_ID]]
// LAUNCH: %{{.*}} = func.call @thread_id() : () -> index
// LAUNCH-NEXT: gpu.return
// LAUNCH-LABEL: func.func @thread_id() -> index
// LAUNCH-NEXT: %[[DEV_RES:.*]] =  scf.execute_region -> index {
// LAUNCH-NEXT: %[[DEV_ID:.*]] = gpu.thread_id  x
// LAUNCH-NEXT: scf.yield %[[DEV_ID]] : index
// LAUNCH-NEXT: }
// LAUNCH-NEXT: return %[[DEV_RES]] : index
