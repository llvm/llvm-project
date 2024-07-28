// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=4,8" | FileCheck %s -check-prefix=VECT
// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=32,256 test-fastest-varying=1,0" | FileCheck %s

// Permutation maps used in vectorization.
// CHECK-DAG: #[[$map_id1:map[0-9]*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$map_proj_d0d1_zerod1:map[0-9]*]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-DAG: #[[$map_proj_d0d1_d0zero:map[0-9]*]] = affine_map<(d0, d1) -> (d0, 0)>
// VECT-DAG: #[[$map_id1:map[0-9]*]] = affine_map<(d0) -> (d0)>
// VECT-DAG: #[[$map_proj_d0d1_zerod1:map[0-9]*]] = affine_map<(d0, d1) -> (0, d1)>
// VECT-DAG: #[[$map_proj_d0d1_d0zero:map[0-9]*]] = affine_map<(d0, d1) -> (d0, 0)>

func.func @vec2d(%A : memref<?x?x?xf32>) {
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?x?xf32>
   %P = memref.dim %A, %c2 : memref<?x?x?xf32>
   // CHECK: for  {{.*}} = 0 to %{{.*}} {
   // CHECK:   for {{.*}} = 0 to %{{.*}} step 32
   // CHECK:     for {{.*}} = 0 to %{{.*}} step 256
   // Example:
   // affine.for %{{.*}} = 0 to %{{.*}} {
   //   affine.for %{{.*}} = 0 to %{{.*}} step 32 {
   //     affine.for %{{.*}} = 0 to %{{.*}} step 256 {
   //       %{{.*}} = "vector.transfer_read"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?xf32>, index, index, index) -> vector<32x256xf32>
   affine.for %i0 = 0 to %M {
     affine.for %i1 = 0 to %N {
       affine.for %i2 = 0 to %P {
         %a2 = affine.load %A[%i0, %i1, %i2] : memref<?x?x?xf32>
       }
     }
   }
   // CHECK: for  {{.*}} = 0 to %{{.*}} {
   // CHECK:   for  {{.*}} = 0 to %{{.*}} {
   // CHECK:     for  {{.*}} = 0 to %{{.*}} {
   // For the case: --test-fastest-varying=1 --test-fastest-varying=0 no
   // vectorization happens because of loop nesting order .
   affine.for %i3 = 0 to %M {
     affine.for %i4 = 0 to %N {
       affine.for %i5 = 0 to %P {
         %a5 = affine.load %A[%i4, %i5, %i3] : memref<?x?x?xf32>
       }
     }
   }
   return
}

func.func @vector_add_2d(%M : index, %N : index) -> f32 {
  %A = memref.alloc (%M, %N) : memref<?x?xf32, 0>
  %B = memref.alloc (%M, %N) : memref<?x?xf32, 0>
  %C = memref.alloc (%M, %N) : memref<?x?xf32, 0>
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  affine.for %i0 = 0 to %M {
    affine.for %i1 = 0 to %N {
      // CHECK: [[C1:%.*]] = arith.constant dense<1.000000e+00> : vector<32x256xf32>
      // CHECK: vector.transfer_write [[C1]], {{.*}} : vector<32x256xf32>, memref<?x?xf32>
      // non-scoped %f1
      affine.store %f1, %A[%i0, %i1] : memref<?x?xf32, 0>
    }
  }
  affine.for %i2 = 0 to %M {
    affine.for %i3 = 0 to %N {
      // CHECK: [[C3:%.*]] = arith.constant dense<2.000000e+00> : vector<32x256xf32>
      // CHECK: vector.transfer_write [[C3]], {{.*}}  : vector<32x256xf32>, memref<?x?xf32>
      // non-scoped %f2
      affine.store %f2, %B[%i2, %i3] : memref<?x?xf32, 0>
    }
  }
  affine.for %i4 = 0 to %M {
    affine.for %i5 = 0 to %N {
      // CHECK: [[SPLAT2:%.*]] = arith.constant dense<2.000000e+00> : vector<32x256xf32>
      // CHECK: [[SPLAT1:%.*]] = arith.constant dense<1.000000e+00> : vector<32x256xf32>
      // CHECK: [[A5:%.*]] = vector.transfer_read %{{.*}}[{{.*}}], %{{.*}} : memref<?x?xf32>, vector<32x256xf32>
      // CHECK: [[B5:%.*]] = vector.transfer_read %{{.*}}[{{.*}}], %{{.*}} : memref<?x?xf32>, vector<32x256xf32>
      // CHECK: [[S5:%.*]] = arith.addf [[A5]], [[B5]] : vector<32x256xf32>
      // CHECK: [[S6:%.*]] = arith.addf [[S5]], [[SPLAT1]] : vector<32x256xf32>
      // CHECK: [[S7:%.*]] = arith.addf [[S5]], [[SPLAT2]] : vector<32x256xf32>
      // CHECK: [[S8:%.*]] = arith.addf [[S7]], [[S6]] : vector<32x256xf32>
      // CHECK: vector.transfer_write [[S8]], {{.*}} : vector<32x256xf32>, memref<?x?xf32>
      //
      %a5 = affine.load %A[%i4, %i5] : memref<?x?xf32, 0>
      %b5 = affine.load %B[%i4, %i5] : memref<?x?xf32, 0>
      %s5 = arith.addf %a5, %b5 : f32
      // non-scoped %f1
      %s6 = arith.addf %s5, %f1 : f32
      // non-scoped %f2
      %s7 = arith.addf %s5, %f2 : f32
      // diamond dependency.
      %s8 = arith.addf %s7, %s6 : f32
      affine.store %s8, %C[%i4, %i5] : memref<?x?xf32, 0>
    }
  }
  %c7 = arith.constant 7 : index
  %c42 = arith.constant 42 : index
  %res = affine.load %C[%c7, %c42] : memref<?x?xf32, 0>
  return %res : f32
}

// VECT-LABEL: func @vectorize_matmul
func.func @vectorize_matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %arg0, %c0 : memref<?x?xf32>
  %K = memref.dim %arg0, %c1 : memref<?x?xf32>
  %N = memref.dim %arg2, %c1 : memref<?x?xf32>
  //      VECT: %[[C0:.*]] = arith.constant 0 : index
  // VECT-NEXT: %[[C1:.*]] = arith.constant 1 : index
  // VECT-NEXT: %[[M:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
  // VECT-NEXT: %[[K:.*]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
  // VECT-NEXT: %[[N:.*]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
  //      VECT: {{.*}} #[[$map_id1]](%[[M]]) step 4 {
  // VECT-NEXT:   {{.*}} #[[$map_id1]](%[[N]]) step 8 {
  //      VECT:     %[[VC0:.*]] = arith.constant dense<0.000000e+00> : vector<4x8xf32>
  // VECT-NEXT:     vector.transfer_write %[[VC0]], %{{.*}}[%{{.*}}, %{{.*}}] : vector<4x8xf32>, memref<?x?xf32>
  affine.for %i0 = affine_map<(d0) -> (d0)>(%c0) to affine_map<(d0) -> (d0)>(%M) {
    affine.for %i1 = affine_map<(d0) -> (d0)>(%c0) to affine_map<(d0) -> (d0)>(%N) {
      %cst = arith.constant 0.000000e+00 : f32
      affine.store %cst, %arg2[%i0, %i1] : memref<?x?xf32>
    }
  }
  //      VECT:  affine.for %[[I2:.*]] = #[[$map_id1]](%[[C0]]) to #[[$map_id1]](%[[M]]) step 4 {
  // VECT-NEXT:    affine.for %[[I3:.*]] = #[[$map_id1]](%[[C0]]) to #[[$map_id1]](%[[N]]) step 8 {
  // VECT-NEXT:      affine.for %[[I4:.*]] = #[[$map_id1]](%[[C0]]) to #[[$map_id1]](%[[K]]) {
  //      VECT:        %[[A:.*]] = vector.transfer_read %{{.*}}[%[[I4]], %[[I3]]], %{{.*}} {in_bounds = [true, false], permutation_map = #[[$map_proj_d0d1_zerod1]]} : memref<?x?xf32>, vector<4x8xf32>
  //      VECT:        %[[B:.*]] = vector.transfer_read %{{.*}}[%[[I2]], %[[I4]]], %{{.*}} {in_bounds = [false, true], permutation_map = #[[$map_proj_d0d1_d0zero]]} : memref<?x?xf32>, vector<4x8xf32>
  // VECT-NEXT:        %[[C:.*]] = arith.mulf %[[B]], %[[A]] : vector<4x8xf32>
  //      VECT:        %[[D:.*]] = vector.transfer_read %{{.*}}[%[[I2]], %[[I3]]], %{{.*}} : memref<?x?xf32>, vector<4x8xf32>
  // VECT-NEXT:        %[[E:.*]] = arith.addf %[[D]], %[[C]] : vector<4x8xf32>
  //      VECT:        vector.transfer_write %[[E]], %{{.*}}[%[[I2]], %[[I3]]] : vector<4x8xf32>, memref<?x?xf32>
  affine.for %i2 = affine_map<(d0) -> (d0)>(%c0) to affine_map<(d0) -> (d0)>(%M) {
    affine.for %i3 = affine_map<(d0) -> (d0)>(%c0) to affine_map<(d0) -> (d0)>(%N) {
      affine.for %i4 = affine_map<(d0) -> (d0)>(%c0) to affine_map<(d0) -> (d0)>(%K) {
        %6 = affine.load %arg1[%i4, %i3] : memref<?x?xf32>
        %7 = affine.load %arg0[%i2, %i4] : memref<?x?xf32>
        %8 = arith.mulf %7, %6 : f32
        %9 = affine.load %arg2[%i2, %i3] : memref<?x?xf32>
        %10 = arith.addf %9, %8 : f32
        affine.store %10, %arg2[%i2, %i3] : memref<?x?xf32>
      }
    }
  }
  return
}
