// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=128 test-fastest-varying=0" -split-input-file | FileCheck %s

// CHECK-DAG: #[[$map_id1:map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$map_proj_d0d1_0:map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> (0)>

// CHECK-LABEL: func @vec1d_1
func.func @vec1d_1(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C2]] : memref<?x?x?xf32>
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK: for {{.*}} step 128
// CHECK-NEXT: %{{.*}} = affine.apply #[[$map_id1]](%[[C0]])
// CHECK-NEXT: %{{.*}} = affine.apply #[[$map_id1]](%[[C0]])
// CHECK-NEXT: %{{.*}} = arith.constant 0.0{{.*}}: f32
// CHECK-NEXT: {{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true], permutation_map = #[[$map_proj_d0d1_0]]} : memref<?x?xf32>, vector<128xf32>
   affine.for %i0 = 0 to %M { // vectorized due to scalar -> vector
     %a0 = affine.load %A[%c0, %c0] : memref<?x?xf32>
   }
   return
}

// -----

// CHECK-LABEL: func @vec1d_2
func.func @vec1d_2(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C2]] : memref<?x?x?xf32>
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK:for [[IV3:%[a-zA-Z0-9]+]] = 0 to [[ARG_M]] step 128
// CHECK-NEXT: %[[CST:.*]] = arith.constant 0.0{{.*}}: f32
// CHECK-NEXT: {{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %[[CST]] : memref<?x?xf32>, vector<128xf32>
   affine.for %i3 = 0 to %M { // vectorized
     %a3 = affine.load %A[%c0, %i3] : memref<?x?xf32>
   }
   return
}

// -----

// CHECK-LABEL: func @vec1d_3
func.func @vec1d_3(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %arg0, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %arg0, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %arg1, %[[C2]] : memref<?x?x?xf32>
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK:for [[IV8:%[0-9a-zA-Z_]+]] = 0 to [[ARG_M]] step 128
// CHECK-NEXT:   for [[IV9:%[0-9a-zA-Z_]*]] = 0 to [[ARG_N]] {
// CHECK-NEXT:   %[[APP9_0:[0-9a-zA-Z_]+]] = affine.apply {{.*}}([[IV9]], [[IV8]])
// CHECK-NEXT:   %[[APP9_1:[0-9a-zA-Z_]+]] = affine.apply {{.*}}([[IV9]], [[IV8]])
// CHECK-NEXT:   %[[CST:.*]] = arith.constant 0.0{{.*}}: f32
// CHECK-NEXT:   {{.*}} = vector.transfer_read %{{.*}}[%[[APP9_0]], %[[APP9_1]]], %[[CST]] : memref<?x?xf32>, vector<128xf32>
   affine.for %i8 = 0 to %M { // vectorized
     affine.for %i9 = 0 to %N {
       %a9 = affine.load %A[%i9, %i8 + %i9] : memref<?x?xf32>
     }
   }
   return
}

// -----

// CHECK-LABEL: func @vector_add_2d
func.func @vector_add_2d(%M : index, %N : index) -> f32 {
  %A = memref.alloc (%M, %N) : memref<?x?xf32, 0>
  %B = memref.alloc (%M, %N) : memref<?x?xf32, 0>
  %C = memref.alloc (%M, %N) : memref<?x?xf32, 0>
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  affine.for %i0 = 0 to %M {
    affine.for %i1 = 0 to %N {
      // CHECK: %[[C1:.*]] = arith.constant dense<1.000000e+00> : vector<128xf32>
      // CHECK: vector.transfer_write %[[C1]], {{.*}} : vector<128xf32>, memref<?x?xf32>
      // non-scoped %f1
      affine.store %f1, %A[%i0, %i1] : memref<?x?xf32, 0>
    }
  }
  affine.for %i2 = 0 to %M {
    affine.for %i3 = 0 to %N {
      // CHECK: %[[C3:.*]] = arith.constant dense<2.000000e+00> : vector<128xf32>
      // CHECK: vector.transfer_write %[[C3]], {{.*}} : vector<128xf32>, memref<?x?xf32>
      // non-scoped %f2
      affine.store %f2, %B[%i2, %i3] : memref<?x?xf32, 0>
    }
  }
  affine.for %i4 = 0 to %M {
    affine.for %i5 = 0 to %N {
      // CHECK: %[[SPLAT2:.*]] = arith.constant dense<2.000000e+00> : vector<128xf32>
      // CHECK: %[[SPLAT1:.*]] = arith.constant dense<1.000000e+00> : vector<128xf32>
      // CHECK: %[[A5:.*]] = vector.transfer_read %{{.*}}[{{.*}}], %{{[a-zA-Z0-9_]*}} : memref<?x?xf32>, vector<128xf32>
      // CHECK: %[[B5:.*]] = vector.transfer_read %{{.*}}[{{.*}}], %{{[a-zA-Z0-9_]*}} : memref<?x?xf32>, vector<128xf32>
      // CHECK: %[[S5:.*]] = arith.addf %[[A5]], %[[B5]] : vector<128xf32>
      // CHECK: %[[S6:.*]] = arith.addf %[[S5]], %[[SPLAT1]] : vector<128xf32>
      // CHECK: %[[S7:.*]] = arith.addf %[[S5]], %[[SPLAT2]] : vector<128xf32>
      // CHECK: %[[S8:.*]] = arith.addf %[[S7]], %[[S6]] : vector<128xf32>
      // CHECK: vector.transfer_write %[[S8]], {{.*}} : vector<128xf32>, memref<?x?xf32>
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

// -----

// CHECK-LABEL: func @vec_constant_with_two_users
func.func @vec_constant_with_two_users(%M : index, %N : index) -> (f32, f32) {
  %A = memref.alloc (%M, %N) : memref<?x?xf32, 0>
  %B = memref.alloc (%M) : memref<?xf32, 0>
  %f1 = arith.constant 1.0 : f32
  affine.for %i0 = 0 to %M { // vectorized
    // CHECK:      %[[C1:.*]] = arith.constant dense<1.000000e+00> : vector<128xf32>
    // CHECK-NEXT: affine.for
    // CHECK-NEXT:   vector.transfer_write %[[C1]], {{.*}} : vector<128xf32>, memref<?x?xf32>
    affine.for %i1 = 0 to %N {
      affine.store %f1, %A[%i1, %i0] : memref<?x?xf32, 0>
    }
    // CHECK: vector.transfer_write %[[C1]], {{.*}} : vector<128xf32>, memref<?xf32>
    affine.store %f1, %B[%i0] : memref<?xf32, 0>
  }
  %c12 = arith.constant 12 : index
  %res1 = affine.load %A[%c12, %c12] : memref<?x?xf32, 0>
  %res2 = affine.load %B[%c12] : memref<?xf32, 0>
  return %res1, %res2 : f32, f32
}

// -----

// CHECK-LABEL: func @vec_block_arg
func.func @vec_block_arg(%A : memref<32x512xi32>) {
  // CHECK:      affine.for %[[IV0:[0-9a-zA-Z_]+]] = 0 to 512 step 128 {
  // CHECK-NEXT:   affine.for %[[IV1:[0-9a-zA-Z_]+]] = 0 to 32 {
  // CHECK-NEXT:     %[[BROADCAST:.*]] = vector.broadcast %[[IV1]] : index to vector<128xindex>
  // CHECK-NEXT:     %[[CAST:.*]] = arith.index_cast %[[BROADCAST]] : vector<128xindex> to vector<128xi32>
  // CHECK-NEXT:     vector.transfer_write %[[CAST]], {{.*}}[%[[IV1]], %[[IV0]]] : vector<128xi32>, memref<32x512xi32>
  affine.for %i = 0 to 512 {  // vectorized
    affine.for %j = 0 to 32 {
      %idx = arith.index_cast %j : index to i32
      affine.store %idx, %A[%j, %i] : memref<32x512xi32>
    }
  }
  return
}

// -----

// CHECK-DAG: #[[$map0:map[0-9a-zA-Z_]*]] = affine_map<(d0, d1, d2) -> (d0 * 2 + d1 - 1)>
// CHECK-DAG: #[[$map1:map[0-9a-zA-Z_]*]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL: func @vec_block_arg_2
func.func @vec_block_arg_2(%A : memref<?x512xindex>) {
  %c0 = arith.constant 0 : index
  %N = memref.dim %A, %c0 : memref<?x512xindex>
  // CHECK:      affine.for %[[IV0:[0-9a-zA-Z_]+]] = 0 to %{{.*}} {
  // CHECK-NEXT:   %[[BROADCAST1:.*]] = vector.broadcast %[[IV0]] : index to vector<128xindex>
  // CHECK-NEXT:   affine.for %[[IV1:[0-9a-zA-Z_]+]] = 0 to 512 step 128 {
  // CHECK-NOT:      vector.broadcast %[[IV1]]
  // CHECK:          affine.for %[[IV2:[0-9a-zA-Z_]+]] = 0 to 2 {
  // CHECK-NEXT:       %[[BROADCAST2:.*]] = vector.broadcast %[[IV2]] : index to vector<128xindex>
  // CHECK-NEXT:       %[[INDEX1:.*]] = affine.apply #[[$map0]](%[[IV0]], %[[IV2]], %[[IV1]])
  // CHECK-NEXT:       %[[INDEX2:.*]] = affine.apply #[[$map1]](%[[IV0]], %[[IV2]], %[[IV1]])
  // CHECK:            %[[LOAD:.*]] = vector.transfer_read %{{.*}}[%[[INDEX1]], %[[INDEX2]]], %{{.*}} : memref<?x512xindex>, vector<128xindex>
  // CHECK-NEXT:       arith.muli %[[BROADCAST1]], %[[LOAD]] : vector<128xindex>
  // CHECK-NEXT:       arith.addi %{{.*}}, %[[BROADCAST2]] : vector<128xindex>
  // CHECK:          }
  affine.for %i0 = 0 to %N {
    affine.for %i1 = 0 to 512 { // vectorized
      affine.for %i2 = 0 to 2 {
        %0 = affine.load %A[%i0 * 2 + %i2 - 1, %i1] : memref<?x512xindex>
        %mul = arith.muli %i0, %0 : index
        %add = arith.addi %mul, %i2 : index
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: func @vec_rejected_1
func.func @vec_rejected_1(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C2]] : memref<?x?x?xf32>
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK:for {{.*}} [[ARG_M]] {
   affine.for %i1 = 0 to %M { // not vectorized
     %a1 = affine.load %A[%i1, %i1] : memref<?x?xf32>
   }
   return
}

// -----

// CHECK-LABEL: func @vec_rejected_2
func.func @vec_rejected_2(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C2]] : memref<?x?x?xf32>
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK:   affine.for %{{.*}}{{[0-9a-zA-Z_]*}} = 0 to [[ARG_M]] {
   affine.for %i2 = 0 to %M { // not vectorized, would vectorize with --test-fastest-varying=1
     %a2 = affine.load %A[%i2, %c0] : memref<?x?xf32>
   }
   return
}

// -----

// CHECK-LABEL: func @vec_rejected_3
func.func @vec_rejected_3(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C2]] : memref<?x?x?xf32>
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK:for [[IV4:%[0-9a-zA-Z_]+]] = 0 to [[ARG_M]] step 128 {
// CHECK-NEXT:   for [[IV5:%[0-9a-zA-Z_]*]] = 0 to [[ARG_N]] {
// CHECK-NEXT:     %{{.*}} = arith.constant 0.0{{.*}}: f32
// CHECK-NEXT:     {{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{[a-zA-Z0-9_]*}} : memref<?x?xf32>, vector<128xf32>
   affine.for %i4 = 0 to %M { // vectorized
     affine.for %i5 = 0 to %N { // not vectorized, would vectorize with --test-fastest-varying=1
       %a5 = affine.load %A[%i5, %i4] : memref<?x?xf32>
     }
   }
   return
}

// -----

// CHECK-LABEL: func @vec_rejected_4
func.func @vec_rejected_4(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C2]] : memref<?x?x?xf32>
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK: for [[IV6:%[0-9a-zA-Z_]*]] = 0 to [[ARG_M]] {
// CHECK-NEXT:   for [[IV7:%[0-9a-zA-Z_]*]] = 0 to [[ARG_N]] {
   affine.for %i6 = 0 to %M { // not vectorized, would vectorize with --test-fastest-varying=1
     affine.for %i7 = 0 to %N { // not vectorized, can never vectorize
       %a7 = affine.load %A[%i6 + %i7, %i6] : memref<?x?xf32>
     }
   }
   return
}

// -----

// CHECK-LABEL: func @vec_rejected_5
func.func @vec_rejected_5(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C2]] : memref<?x?x?xf32>
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK: for [[IV10:%[0-9a-zA-Z_]*]] = 0 to %{{[0-9a-zA-Z_]*}} {
// CHECK:   for [[IV11:%[0-9a-zA-Z_]*]] = 0 to %{{[0-9a-zA-Z_]*}} {
   affine.for %i10 = 0 to %M { // not vectorized, need per load transposes
     affine.for %i11 = 0 to %N { // not vectorized, need per load transposes
       %a11 = affine.load %A[%i10, %i11] : memref<?x?xf32>
       affine.store %a11, %A[%i11, %i10] : memref<?x?xf32>
     }
   }
   return
}

// -----

// CHECK-LABEL: func @vec_rejected_6
func.func @vec_rejected_6(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C2]] : memref<?x?x?xf32>
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK: for [[IV12:%[0-9a-zA-Z_]*]] = 0 to %{{[0-9a-zA-Z_]*}} {
// CHECK:   for [[IV13:%[0-9a-zA-Z_]*]] = 0 to %{{[0-9a-zA-Z_]*}} {
// CHECK:     for [[IV14:%[0-9a-zA-Z_]+]] = 0 to [[ARG_P]] step 128
   affine.for %i12 = 0 to %M { // not vectorized, can never vectorize
     affine.for %i13 = 0 to %N { // not vectorized, can never vectorize
       affine.for %i14 = 0 to %P { // vectorized
         %a14 = affine.load %B[%i13, %i12 + %i13, %i12 + %i14] : memref<?x?x?xf32>
       }
     }
   }
   return
}

// -----

// CHECK-LABEL: func @vec_rejected_7
func.func @vec_rejected_7(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C2]] : memref<?x?x?xf32>
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK:  affine.for %{{.*}}{{[0-9a-zA-Z_]*}} = 0 to %{{[0-9a-zA-Z_]*}} {
   affine.for %i16 = 0 to %M { // not vectorized, can't vectorize a vector load
     %a16 = memref.alloc(%M) : memref<?xvector<2xf32>>
     %l16 = affine.load %a16[%i16] : memref<?xvector<2xf32>>
   }
   return
}

// -----

// CHECK-DAG: #[[$map_id1:map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$map_proj_d0d1_0:map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> (0)>

// CHECK-LABEL: func @vec_rejected_8
func.func @vec_rejected_8(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C2]] : memref<?x?x?xf32>
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK: affine.for %{{.*}}{{[0-9a-zA-Z_]*}} = 0 to %{{[0-9a-zA-Z_]*}} {
// CHECK:   for [[IV18:%[a-zA-Z0-9]+]] = 0 to [[ARG_M]] step 128
// CHECK:     %{{.*}} = affine.apply #[[$map_id1]](%{{.*}})
// CHECK:     %{{.*}} = affine.apply #[[$map_id1]](%{{.*}})
// CHECK:     %{{.*}} = arith.constant 0.0{{.*}}: f32
// CHECK:     {{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true], permutation_map = #[[$map_proj_d0d1_0]]} : memref<?x?xf32>, vector<128xf32>
   affine.for %i17 = 0 to %M { // not vectorized, the 1-D pattern that matched %{{.*}} in DFS post-order prevents vectorizing %{{.*}}
     affine.for %i18 = 0 to %M { // vectorized due to scalar -> vector
       %a18 = affine.load %A[%c0, %c0] : memref<?x?xf32>
     }
   }
   return
}

// -----

// CHECK-DAG: #[[$map_id1:map[0-9a-zA-Z_]*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$map_proj_d0d1_0:map[0-9a-zA-Z_]*]] = affine_map<(d0, d1) -> (0)>

// CHECK-LABEL: func @vec_rejected_9
func.func @vec_rejected_9(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C2]] : memref<?x?x?xf32>
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK: affine.for %{{.*}}{{[0-9a-zA-Z_]*}} = 0 to %{{[0-9a-zA-Z_]*}} {
// CHECK:   for [[IV18:%[a-zA-Z0-9]+]] = 0 to [[ARG_M]] step 128
// CHECK:      %{{.*}} = affine.apply #[[$map_id1]](%{{.*}})
// CHECK-NEXT: %{{.*}} = affine.apply #[[$map_id1]](%{{.*}})
// CHECK-NEXT: %{{.*}} = arith.constant 0.0{{.*}}: f32
// CHECK-NEXT: {{.*}} = vector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true], permutation_map = #[[$map_proj_d0d1_0]]} : memref<?x?xf32>, vector<128xf32>
   affine.for %i17 = 0 to %M { // not vectorized, the 1-D pattern that matched %i18 in DFS post-order prevents vectorizing %{{.*}}
     affine.for %i18 = 0 to %M { // vectorized due to scalar -> vector
       %a18 = affine.load %A[%c0, %c0] : memref<?x?xf32>
     }
   }
   return
}

// -----

#set0 = affine_set<(i) : (i >= 0)>

// CHECK-LABEL: func @vec_rejected_10
func.func @vec_rejected_10(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
// CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C2]] : memref<?x?x?xf32>
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %M = memref.dim %A, %c0 : memref<?x?xf32>
   %N = memref.dim %A, %c1 : memref<?x?xf32>
   %P = memref.dim %B, %c2 : memref<?x?x?xf32>

// CHECK:  affine.for %{{.*}}{{[0-9a-zA-Z_]*}} = 0 to %{{[0-9a-zA-Z_]*}} {
   affine.for %i15 = 0 to %M { // not vectorized due to condition below
     affine.if #set0(%i15) {
       %a15 = affine.load %A[%c0, %c0] : memref<?x?xf32>
     }
   }
   return
}

// -----

// CHECK-LABEL: func @vec_rejected_11
func.func @vec_rejected_11(%A : memref<?x?xf32>, %B : memref<?x?x?xf32>) {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: [[ARG_M:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C0]] : memref<?x?xf32>
  // CHECK-DAG: [[ARG_N:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C1]] : memref<?x?xf32>
  // CHECK-DAG: [[ARG_P:%[0-9a-zA-Z_]+]] = memref.dim %{{.*}}, %[[C2]] : memref<?x?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %M = memref.dim %A, %c0 : memref<?x?xf32>
  %N = memref.dim %A, %c1 : memref<?x?xf32>
  %P = memref.dim %B, %c2 : memref<?x?x?xf32>

  // CHECK: for [[IV10:%[0-9a-zA-Z_]*]] = 0 to %{{[0-9a-zA-Z_]*}} {
  // CHECK:   for [[IV11:%[0-9a-zA-Z_]*]] = 0 to %{{[0-9a-zA-Z_]*}} {
  // This is similar to vec_rejected_5, but the order of indices is different.
  affine.for %i10 = 0 to %M { // not vectorized
    affine.for %i11 = 0 to %N { // not vectorized
      %a11 = affine.load %A[%i11, %i10] : memref<?x?xf32>
      affine.store %a11, %A[%i10, %i11] : memref<?x?xf32>
    }
  }
  return
}

// -----

// This should not vectorize due to the sequential dependence in the loop.
// CHECK-LABEL: @vec_rejected_sequential
func.func @vec_rejected_sequential(%A : memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %N = memref.dim %A, %c0 : memref<?xf32>
  affine.for %i = 0 to %N {
    // CHECK-NOT: vector
    %a = affine.load %A[%i] : memref<?xf32>
    // CHECK-NOT: vector
    affine.store %a, %A[%i + 1] : memref<?xf32>
  }
  return
}

// -----

// CHECK-LABEL: @vec_no_load_store_ops
func.func @vec_no_load_store_ops(%a: f32, %b: f32) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 128 {
   %add = arith.addf %a, %b : f32
 }
 // CHECK-DAG:  %[[bc1:.*]] = vector.broadcast
 // CHECK-DAG:  %[[bc0:.*]] = vector.broadcast
 // CHECK:      affine.for %{{.*}} = 0 to 128 step
 // CHECK-NEXT:   [[add:.*]] arith.addf %[[bc0]], %[[bc1]]

 return
}

// -----

// This should not be vectorized due to the unsupported block argument (%i).
// Support for operands with linear evolution is needed.
// CHECK-LABEL: @vec_rejected_unsupported_block_arg
func.func @vec_rejected_unsupported_block_arg(%A : memref<512xi32>) {
  affine.for %i = 0 to 512 {
    // CHECK-NOT: vector
    %idx = arith.index_cast %i : index to i32
    affine.store %idx, %A[%i] : memref<512xi32>
  }
  return
}

// -----

// '%i' loop is vectorized, including the inner reduction over '%j'.

func.func @vec_non_vecdim_reduction(%in: memref<128x256xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to 128 iter_args(%red_iter = %cst) -> (f32) {
     %ld = affine.load %in[%j, %i] : memref<128x256xf32>
     %add = arith.addf %red_iter, %ld : f32
     affine.yield %add : f32
   }
   affine.store %final_red, %out[%i] : memref<256xf32>
 }
 return
}

// CHECK-LABEL: @vec_non_vecdim_reduction
// CHECK:       affine.for %{{.*}} = 0 to 256 step 128 {
// CHECK:         %[[vzero:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:         %[[final_red:.*]] = affine.for %{{.*}} = 0 to 128 iter_args(%[[red_iter:.*]] = %[[vzero]]) -> (vector<128xf32>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<128x256xf32>, vector<128xf32>
// CHECK:           %[[add:.*]] = arith.addf %[[red_iter]], %[[ld]] : vector<128xf32>
// CHECK:           affine.yield %[[add]] : vector<128xf32>
// CHECK:         }
// CHECK:         vector.transfer_write %[[final_red]], %{{.*}} : vector<128xf32>, memref<256xf32>
// CHECK:       }

// -----

// '%i' loop is vectorized, including the inner reductions over '%j'.

func.func @vec_non_vecdim_reductions(%in0: memref<128x256xf32>, %in1: memref<128x256xi32>,
                                %out0: memref<256xf32>, %out1: memref<256xi32>) {
 %zero = arith.constant 0.000000e+00 : f32
 %one = arith.constant 1 : i32
 affine.for %i = 0 to 256 {
   %red0, %red1 = affine.for %j = 0 to 128
     iter_args(%red_iter0 = %zero, %red_iter1 = %one) -> (f32, i32) {
     %ld0 = affine.load %in0[%j, %i] : memref<128x256xf32>
     %add = arith.addf %red_iter0, %ld0 : f32
     %ld1 = affine.load %in1[%j, %i] : memref<128x256xi32>
     %mul = arith.muli %red_iter1, %ld1 : i32
     affine.yield %add, %mul : f32, i32
   }
   affine.store %red0, %out0[%i] : memref<256xf32>
   affine.store %red1, %out1[%i] : memref<256xi32>
 }
 return
}

// CHECK-LABEL: @vec_non_vecdim_reductions
// CHECK:       affine.for %{{.*}} = 0 to 256 step 128 {
// CHECK:         %[[vone:.*]] = arith.constant dense<1> : vector<128xi32>
// CHECK:         %[[vzero:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:         %[[reds:.*]]:2 = affine.for %{{.*}} = 0 to 128
// CHECK-SAME:      iter_args(%[[red_iter0:.*]] = %[[vzero]], %[[red_iter1:.*]] = %[[vone]]) -> (vector<128xf32>, vector<128xi32>) {
// CHECK:           %[[ld0:.*]] = vector.transfer_read %{{.*}} : memref<128x256xf32>, vector<128xf32>
// CHECK:           %[[add:.*]] = arith.addf %[[red_iter0]], %[[ld0]] : vector<128xf32>
// CHECK:           %[[ld1:.*]] = vector.transfer_read %{{.*}} : memref<128x256xi32>, vector<128xi32>
// CHECK:           %[[mul:.*]] = arith.muli %[[red_iter1]], %[[ld1]] : vector<128xi32>
// CHECK:           affine.yield %[[add]], %[[mul]] : vector<128xf32>, vector<128xi32>
// CHECK:         }
// CHECK:         vector.transfer_write %[[reds]]#0, %{{.*}} : vector<128xf32>, memref<256xf32>
// CHECK:         vector.transfer_write %[[reds]]#1, %{{.*}} : vector<128xi32>, memref<256xi32>
// CHECK:       }

// -----

// '%i' loop is vectorized, including the inner last value computation over '%j'.

func.func @vec_no_vecdim_last_value(%in: memref<128x256xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %last_val = affine.for %j = 0 to 128 iter_args(%last_iter = %cst) -> (f32) {
     %ld = affine.load %in[%j, %i] : memref<128x256xf32>
     affine.yield %ld : f32
   }
   affine.store %last_val, %out[%i] : memref<256xf32>
 }
 return
}

// CHECK-LABEL: @vec_no_vecdim_last_value
// CHECK:       affine.for %{{.*}} = 0 to 256 step 128 {
// CHECK:         %[[vzero:.*]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK:         %[[last_val:.*]] = affine.for %{{.*}} = 0 to 128 iter_args(%[[last_iter:.*]] = %[[vzero]]) -> (vector<128xf32>) {
// CHECK:           %[[ld:.*]] = vector.transfer_read %{{.*}} : memref<128x256xf32>, vector<128xf32>
// CHECK:           affine.yield %[[ld]] : vector<128xf32>
// CHECK:         }
// CHECK:         vector.transfer_write %[[last_val]], %{{.*}} : vector<128xf32>, memref<256xf32>
// CHECK:       }

// -----

// The inner reduction loop '%j' is not vectorized if we do not request
// reduction vectorization.

func.func @vec_vecdim_reduction_rejected(%in: memref<256x512xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %final_red = affine.for %j = 0 to 512 iter_args(%red_iter = %cst) -> (f32) {
     %ld = affine.load %in[%i, %j] : memref<256x512xf32>
     %add = arith.addf %red_iter, %ld : f32
     affine.yield %add : f32
   }
   affine.store %final_red, %out[%i] : memref<256xf32>
 }
 return
}

// CHECK-LABEL: @vec_vecdim_reduction_rejected
// CHECK-NOT: vector
