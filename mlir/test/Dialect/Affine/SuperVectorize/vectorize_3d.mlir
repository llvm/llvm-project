// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=32,64,256 test-fastest-varying=2,1,0" | FileCheck %s

func @vec3d(%A : memref<?x?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %0 = dim %A, %c0 : memref<?x?x?xf32>
  %1 = dim %A, %c1 : memref<?x?x?xf32>
  %2 = dim %A, %c2 : memref<?x?x?xf32>
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
