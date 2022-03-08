// RUN: mlir-opt -pass-pipeline="func.func(convert-affine-for-to-gpu{gpu-block-dims=0 gpu-thread-dims=1})" %s | FileCheck --check-prefix=CHECK-THREADS %s
// RUN: mlir-opt -pass-pipeline="func.func(convert-affine-for-to-gpu{gpu-block-dims=1 gpu-thread-dims=0})" %s | FileCheck --check-prefix=CHECK-BLOCKS %s

// CHECK-THREADS-LABEL: @one_d_loop
// CHECK-BLOCKS-LABEL: @one_d_loop
func @one_d_loop(%A : memref<?xf32>, %B : memref<?xf32>) {
  // Bounds of the loop, its range and step.
  // CHECK-THREADS-NEXT: %{{.*}} = arith.constant 0 : index
  // CHECK-THREADS-NEXT: %{{.*}} = arith.constant 42 : index
  // CHECK-THREADS-NEXT: %[[BOUND:.*]] = arith.subi %{{.*}}, %{{.*}} : index
  // CHECK-THREADS-NEXT: %{{.*}} = arith.constant 1 : index
  // CHECK-THREADS-NEXT: %[[ONE:.*]] = arith.constant 1 : index
  //
  // CHECK-BLOCKS-NEXT: %{{.*}} = arith.constant 0 : index
  // CHECK-BLOCKS-NEXT: %{{.*}} = arith.constant 42 : index
  // CHECK-BLOCKS-NEXT: %[[BOUND:.*]] = arith.subi %{{.*}}, %{{.*}} : index
  // CHECK-BLOCKS-NEXT: %{{.*}} = arith.constant 1 : index
  // CHECK-BLOCKS-NEXT: %[[ONE:.*]] = arith.constant 1 : index

  // CHECK-THREADS-NEXT: gpu.launch blocks(%[[B0:.*]], %[[B1:.*]], %[[B2:.*]]) in (%{{.*}} = %[[ONE]], %{{.*}} = %[[ONE]], %{{.*}}0 = %[[ONE]]) threads(%[[T0:.*]], %[[T1:.*]], %[[T2:.*]]) in (%{{.*}} = %[[BOUND]], %{{.*}} = %[[ONE]], %{{.*}} = %[[ONE]])
  // CHECK-BLOCKS-NEXT: gpu.launch blocks(%[[B0:.*]], %[[B1:.*]], %[[B2:.*]]) in (%{{.*}} = %[[BOUND]], %{{.*}} = %[[ONE]], %{{.*}}0 = %[[ONE]]) threads(%[[T0:.*]], %[[T1:.*]], %[[T2:.*]]) in (%{{.*}} = %[[ONE]], %{{.*}} = %[[ONE]], %{{.*}} = %[[ONE]])
  affine.for %i = 0 to 42 {
  // CHECK-THREADS-NEXT: %[[INDEX:.*]] = arith.addi %{{.*}}, %[[T0]]
  // CHECK-THREADS-NEXT: memref.load %{{.*}}[%[[INDEX]]]
  // CHECK-BLOCKS-NEXT: %[[INDEX:.*]] = arith.addi %{{.*}}, %[[B0]]
  // CHECK-BLOCKS-NEXT: memref.load %{{.*}}[%[[INDEX]]]
    %0 = memref.load %A[%i] : memref<?xf32>
    memref.store %0, %B[%i] : memref<?xf32>
    // CHECK-THREADS: gpu.terminator
    // CHECK-BLOCKS: gpu.terminator
  }
  return
}

