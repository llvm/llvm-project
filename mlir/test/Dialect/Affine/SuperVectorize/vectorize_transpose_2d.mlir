// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=32,256 test-fastest-varying=0,1" | FileCheck %s

// Permutation maps used in vectorization.
// CHECK-DAG: #[[map_proj_d0d1d2_d2d1:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$map_proj_d0d1_d1d0:map[0-9]*]] = affine_map<(d0, d1) -> (d1, d0)>

func.func @vec2d(%A : memref<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %M = memref.dim %A, %c0 : memref<?x?x?xf32>
  %N = memref.dim %A, %c1 : memref<?x?x?xf32>
  %P = memref.dim %A, %c2 : memref<?x?x?xf32>
  // CHECK: for  {{.*}} = 0 to %{{.*}} {
  // CHECK:   for  {{.*}} = 0 to %{{.*}} {
  // CHECK:     for  {{.*}} = 0 to %{{.*}} {
  // For the case: --test-fastest-varying=0 --test-fastest-varying=1 no
  // vectorization happens because of loop nesting order.
  affine.for %i0 = 0 to %M {
    affine.for %i1 = 0 to %N {
      affine.for %i2 = 0 to %P {
        %a2 = affine.load %A[%i0, %i1, %i2] : memref<?x?x?xf32>
      }
    }
  }
  // CHECK: affine.for %{{.*}} = 0 to %{{.*}} step 32
  // CHECK:   affine.for %{{.*}} = 0 to %{{.*}} {
  // CHECK:     affine.for %{{.*}} = 0 to %{{.*}} step 256
  // CHECK:       {{.*}} = vector.mask %{{.*}} { vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true], permutation_map = #[[map_proj_d0d1d2_d2d1]]} : memref<?x?x?xf32>, vector<32x256xf32> }
  affine.for %i3 = 0 to %M {
    affine.for %i4 = 0 to %N {
      affine.for %i5 = 0 to %P {
        %a5 = affine.load %A[%i4, %i5, %i3] : memref<?x?x?xf32>
      }
    }
  }
  return
}

func.func @vec2d_imperfectly_nested(%A : memref<?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = memref.dim %A, %c0 : memref<?x?x?xf32>
  %1 = memref.dim %A, %c1 : memref<?x?x?xf32>
  %2 = memref.dim %A, %c2 : memref<?x?x?xf32>
  // CHECK: affine.for %{{.*}} = 0 to %{{.*}} step 32 {
  // CHECK:   affine.for %{{.*}} = 0 to %{{.*}} step 256 {
  // CHECK:     affine.for %{{.*}} = 0 to %{{.*}} {
  // CHECK:       %{{.*}} = vector.mask %{{.*}} { vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true], permutation_map = #[[map_proj_d0d1d2_d2d1]]} : memref<?x?x?xf32>, vector<32x256xf32> }
  // CHECK:   affine.for %{{.*}} = 0 to %{{.*}} {
  // CHECK:     affine.for %{{.*}} = 0 to %{{.*}} step 256 {
  // CHECK:       %{{.*}} = vector.mask %{{.*}} { vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true], permutation_map = #[[map_proj_d0d1d2_d2d1]]} : memref<?x?x?xf32>, vector<32x256xf32> }
  // CHECK:     affine.for %{{.*}} = 0 to %{{.*}} step 256 {
  // CHECK:       %{{.*}} = vector.mask %{{.*}} { vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true], permutation_map = #[[map_proj_d0d1d2_d2d1]]} : memref<?x?x?xf32>, vector<32x256xf32> }
  affine.for %i0 = 0 to %0 {
    affine.for %i1 = 0 to %1 {
      affine.for %i2 = 0 to %2 {
        %a2 = affine.load %A[%i2, %i1, %i0] : memref<?x?x?xf32>
      }
    }
    affine.for %i3 = 0 to %1 {
      affine.for %i4 = 0 to %2 {
        %a4 = affine.load %A[%i3, %i4, %i0] : memref<?x?x?xf32>
      }
      affine.for %i5 = 0 to %2 {
        %a5 = affine.load %A[%i3, %i5, %i0] : memref<?x?x?xf32>
      }
    }
  }
  return
}

// The map selects the memref dims out of order, but the mask dims are in
// increasing order.

// CHECK-LABEL: func @transposed_store
// CHECK:         affine.for %[[I:.*]] = 0 to 128 step 32
// CHECK:           affine.for %[[J:.*]] = 0 to 512 step 256
// CHECK:             %[[SIZE0:.*]] = arith.subi %{{.*}}, %[[J]] : index
// CHECK:             %[[SIZE1:.*]] = arith.subi %{{.*}}, %[[I]] : index
// CHECK:             %[[MASK:.*]] = vector.create_mask %[[SIZE0]], %[[SIZE1]] : vector<256x32xi1>
// CHECK:             vector.mask %[[MASK]] { vector.transfer_write %{{.*}}, %{{.*}}[%[[J]], %[[I]]] {{.*}}permutation_map = #[[$map_proj_d0d1_d1d0]]{{.*}} : vector<32x256xf32>
func.func @transposed_store(%A: memref<512x128xf32>, %f: f32) {
  affine.for %i = 0 to 128 {
    affine.for %j = 0 to 512 {
      affine.store %f, %A[%j, %i] : memref<512x128xf32>
    }
  }
  return
}
