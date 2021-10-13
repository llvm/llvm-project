// RUN: mlir-opt -convert-affine-for-to-gpu="gpu-block-dims=1 gpu-thread-dims=1" %s | FileCheck --check-prefix=CHECK-11 %s
// RUN: mlir-opt -convert-affine-for-to-gpu="gpu-block-dims=2 gpu-thread-dims=2" %s | FileCheck --check-prefix=CHECK-22 %s

// CHECK-11-LABEL: @step_1
// CHECK-22-LABEL: @step_1
func @step_1(%A : memref<?x?x?x?xf32>, %B : memref<?x?x?x?xf32>) {
  // Bounds of the loop, its range and step.
  // CHECK-11-NEXT: %{{.*}} = arith.constant 0 : index
  // CHECK-11-NEXT: %{{.*}} = arith.constant 42 : index
  // CHECK-11-NEXT: %{{.*}} = arith.subi %{{.*}}, %{{.*}} : index
  // CHECK-11-NEXT: %{{.*}} = arith.constant 1 : index
  //
  // CHECK-22-NEXT: %{{.*}} = arith.constant 0 : index
  // CHECK-22-NEXT: %{{.*}} = arith.constant 42 : index
  // CHECK-22-NEXT: %{{.*}} = arith.subi %{{.*}}, %{{.*}} : index
  // CHECK-22-NEXT: %{{.*}} = arith.constant 1 : index
  affine.for %i = 0 to 42 {

    // Bounds of the loop, its range and step.
    // CHECK-11-NEXT: %{{.*}} = arith.constant 0 : index
    // CHECK-11-NEXT: %{{.*}} = arith.constant 10 : index
    // CHECK-11-NEXT: %{{.*}} = arith.subi %{{.*}}, %{{.*}} : index
    // CHECK-11-NEXT: %{{.*}} = arith.constant 1 : index
    //
    // CHECK-22-NEXT: %{{.*}} = arith.constant 0 : index
    // CHECK-22-NEXT: %{{.*}} = arith.constant 10 : index
    // CHECK-22-NEXT: %{{.*}} = arith.subi %{{.*}}, %{{.*}} : index
    // CHECK-22-NEXT: %{{.*}} = arith.constant 1 : index
    affine.for %j = 0 to 10 {
    // CHECK-11: gpu.launch
    // CHECK-11-SAME: blocks
    // CHECK-11-SAME: threads

      // Remapping of the loop induction variables.
      // CHECK-11:        %[[i:.*]] = arith.addi %{{.*}}, %{{.*}} : index
      // CHECK-11-NEXT:   %[[j:.*]] = arith.addi %{{.*}}, %{{.*}} : index

      // This loop is not converted if mapping to 1, 1 dimensions.
      // CHECK-11-NEXT: affine.for %[[ii:.*]] = 2 to 16
      //
      // Bounds of the loop, its range and step.
      // CHECK-22-NEXT: %{{.*}} = arith.constant 2 : index
      // CHECK-22-NEXT: %{{.*}} = arith.constant 16 : index
      // CHECK-22-NEXT: %{{.*}} = arith.subi %{{.*}}, %{{.*}} : index
      // CHECK-22-NEXT: %{{.*}} = arith.constant 1 : index
      affine.for %ii = 2 to 16 {
        // This loop is not converted if mapping to 1, 1 dimensions.
        // CHECK-11-NEXT: affine.for %[[jj:.*]] = 5 to 17
        //
        // Bounds of the loop, its range and step.
        // CHECK-22-NEXT: %{{.*}} = arith.constant 5 : index
        // CHECK-22-NEXT: %{{.*}} = arith.constant 17 : index
        // CHECK-22-NEXT: %{{.*}} = arith.subi %{{.*}}, %{{.*}} : index
        // CHECK-22-NEXT: %{{.*}} = arith.constant 1 : index
        affine.for %jj = 5 to 17 {
        // CHECK-22: gpu.launch
        // CHECK-22-SAME: blocks
        // CHECK-22-SAME: threads

          // Remapping of the loop induction variables in the last mapped scf.
          // CHECK-22:        %[[i:.*]] = arith.addi %{{.*}}, %{{.*}} : index
          // CHECK-22-NEXT:   %[[j:.*]] = arith.addi %{{.*}}, %{{.*}} : index
          // CHECK-22-NEXT:   %[[ii:.*]] = arith.addi %{{.*}}, %{{.*}} : index
          // CHECK-22-NEXT:   %[[jj:.*]] = arith.addi %{{.*}}, %{{.*}} : index

          // Using remapped values instead of loop iterators.
          // CHECK-11:        {{.*}} = memref.load %{{.*}}[%[[i]], %[[j]], %[[ii]], %[[jj]]] : memref<?x?x?x?xf32>
          // CHECK-22:        {{.*}} = memref.load %{{.*}}[%[[i]], %[[j]], %[[ii]], %[[jj]]] : memref<?x?x?x?xf32>
          %0 = memref.load %A[%i, %j, %ii, %jj] : memref<?x?x?x?xf32>
          // CHECK-11-NEXT:   memref.store {{.*}}, %{{.*}}[%[[i]], %[[j]], %[[ii]], %[[jj]]] : memref<?x?x?x?xf32>
          // CHECK-22-NEXT:   memref.store {{.*}}, %{{.*}}[%[[i]], %[[j]], %[[ii]], %[[jj]]] : memref<?x?x?x?xf32>
          memref.store %0, %B[%i, %j, %ii, %jj] : memref<?x?x?x?xf32>

          // CHECK-11: gpu.terminator
          // CHECK-22: gpu.terminator
        }
      }
    }
  }
  return
}

