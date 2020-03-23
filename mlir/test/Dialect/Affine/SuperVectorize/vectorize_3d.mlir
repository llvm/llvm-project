// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=32,64,256 test-fastest-varying=2,1,0" | FileCheck %s

// Permutation maps used in vectorization.
// CHECK: #[[map_proj_d0d1d2_d0d1d2:map[0-9]+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func @vec3d(%A : memref<?x?x?xf32>) {
   %0 = dim %A, 0 : memref<?x?x?xf32>
   %1 = dim %A, 1 : memref<?x?x?xf32>
   %2 = dim %A, 2 : memref<?x?x?xf32>
   // CHECK: affine.for %{{.*}} = 0 to %{{.*}} {
   // CHECK:   affine.for %{{.*}} = 0 to %{{.*}} {
   // CHECK:     affine.for %{{.*}} = 0 to %{{.*}} step 32 {
   // CHECK:       affine.for %{{.*}} = 0 to %{{.*}} step 64 {
   // CHECK:         affine.for %{{.*}} = 0 to %{{.*}} step 256 {
   // CHECK:           %{{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}} {permutation_map = #[[map_proj_d0d1d2_d0d1d2]]} : memref<?x?x?xf32>, vector<32x64x256xf32>
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
