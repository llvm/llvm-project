// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=32,64,256 test-fastest-varying=2,1,0" | FileCheck %s

func.func @vec3d(%A : memref<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = memref.dim %A, %c0 : memref<?x?x?xf32>
  %1 = memref.dim %A, %c1 : memref<?x?x?xf32>
  %2 = memref.dim %A, %c2 : memref<?x?x?xf32>
  // CHECK: affine.for %{{.*}} = 0 to %{{.*}} {
  // CHECK:   affine.for %{{.*}} = 0 to %{{.*}} {
  // CHECK:     affine.for %{{.*}} = 0 to %{{.*}} step 32 {
  // CHECK:       affine.for %{{.*}} = 0 to %{{.*}} step 64 {
  // CHECK:         affine.for %{{.*}} = 0 to %{{.*}} step 256 {
  // CHECK:           %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}} : memref<?x?x?xf32>, vector<32x64x256xf32>
  affine.for %t0 = 0 to %0 {
    affine.for %t1 = 0 to %0 {
      affine.for %i0 = 0 to %0 {
        affine.for %i1 = 0 to %1 {
          affine.for %i2 = 0 to %2 {
            %a2 = affine.load %A[%i0, %i1, %i2] : memref<?x?x?xf32>
          }
        }
      }
    }
  }
  return
}
