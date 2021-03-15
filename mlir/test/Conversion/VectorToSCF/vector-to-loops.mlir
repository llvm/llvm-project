// RUN: mlir-opt %s -convert-vector-to-scf -split-input-file -allow-unregistered-dialect | FileCheck %s
// RUN: mlir-opt %s -convert-vector-to-scf=full-unroll=true -split-input-file -allow-unregistered-dialect | FileCheck %s --check-prefix=FULL-UNROLL

// CHECK-LABEL: func @materialize_read_1d() {
func @materialize_read_1d() {
  %f0 = constant 0.0: f32
  %A = alloc () : memref<7x42xf32>
  affine.for %i0 = 0 to 7 step 4 {
    affine.for %i1 = 0 to 42 step 4 {
      %f1 = vector.transfer_read %A[%i0, %i1], %f0 {permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<7x42xf32>, vector<4xf32>
      %ip1 = affine.apply affine_map<(d0) -> (d0 + 1)> (%i1)
      %f2 = vector.transfer_read %A[%i0, %ip1], %f0 {permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<7x42xf32>, vector<4xf32>
      %ip2 = affine.apply affine_map<(d0) -> (d0 + 2)> (%i1)
      %f3 = vector.transfer_read %A[%i0, %ip2], %f0 {permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<7x42xf32>, vector<4xf32>
      %ip3 = affine.apply affine_map<(d0) -> (d0 + 3)> (%i1)
      %f4 = vector.transfer_read %A[%i0, %ip3], %f0 {permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<7x42xf32>, vector<4xf32>
      // Both accesses in the load must be clipped otherwise %i1 + 2 and %i1 + 3 will go out of bounds.
      // CHECK: scf.if
      // CHECK-NEXT: load
      // CHECK-NEXT: vector.insertelement
      // CHECK-NEXT: store
      // CHECK-NEXT: else
      // CHECK-NEXT: vector.insertelement
      // CHECK-NEXT: store
      // Add a dummy use to prevent dead code elimination from removing transfer
      // read ops.
      "dummy_use"(%f1, %f2, %f3, %f4) : (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) -> ()
    }
  }
  return
}

// -----

// CHECK-LABEL: func @materialize_read_1d_partially_specialized
func @materialize_read_1d_partially_specialized(%dyn1 : index, %dyn2 : index, %dyn4 : index) {
  %f0 = constant 0.0: f32
  %A = alloc (%dyn1, %dyn2, %dyn4) : memref<7x?x?x42x?xf32>
  affine.for %i0 = 0 to 7 {
    affine.for %i1 = 0 to %dyn1 {
      affine.for %i2 = 0 to %dyn2 {
        affine.for %i3 = 0 to 42 step 2 {
          affine.for %i4 = 0 to %dyn4 {
            %f1 = vector.transfer_read %A[%i0, %i1, %i2, %i3, %i4], %f0 {permutation_map = affine_map<(d0, d1, d2, d3, d4) -> (d3)>} : memref<7x?x?x42x?xf32>, vector<4xf32>
            %i3p1 = affine.apply affine_map<(d0) -> (d0 + 1)> (%i3)
            %f2 = vector.transfer_read %A[%i0, %i1, %i2, %i3p1, %i4], %f0 {permutation_map = affine_map<(d0, d1, d2, d3, d4) -> (d3)>} : memref<7x?x?x42x?xf32>, vector<4xf32>
            // Add a dummy use to prevent dead code elimination from removing
            // transfer read ops.
            "dummy_use"(%f1, %f2) : (vector<4xf32>, vector<4xf32>) -> ()
          }
        }
      }
    }
  }
  // CHECK: %[[tensor:[0-9]+]] = alloc
  // CHECK-NOT: {{.*}} dim %[[tensor]], %c0
  // CHECK-NOT: {{.*}} dim %[[tensor]], %c3
  return
}

// -----

// CHECK: #[[$ADD:map.*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: func @materialize_read(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
func @materialize_read(%M: index, %N: index, %O: index, %P: index) {
  %f0 = constant 0.0: f32
  // CHECK-DAG:  %[[ALLOC:.*]] = alloca() : memref<5x4xvector<3xf32>>
  // CHECK-DAG:  %[[C0:.*]] = constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = constant 1 : index
  // CHECK-DAG:  %[[C3:.*]] = constant 3 : index
  // CHECK-DAG:  %[[C4:.*]] = constant 4 : index
  // CHECK-DAG:  %[[C5:.*]] = constant 5 : index
  //     CHECK:  %{{.*}} = alloc(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?x?x?xf32>
  // CHECK-NEXT:  affine.for %[[I0:.*]] = 0 to %{{.*}} step 3 {
  // CHECK-NEXT:    affine.for %[[I1:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:      affine.for %[[I2:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:        affine.for %[[I3:.*]] = 0 to %{{.*}} step 5 {
  // CHECK-NEXT:          scf.for %[[I4:.*]] = %[[C0]] to %[[C3]] step %[[C1]] {
  // CHECK-NEXT:            scf.for %[[I5:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
  // CHECK-NEXT:              scf.for %[[I6:.*]] = %[[C0]] to %[[C5]] step %[[C1]] {
  // CHECK:                     %[[VIDX:.*]] = index_cast %[[I4]]
  // CHECK:                     %[[VEC:.*]] = load %[[ALLOC]][%[[I6]], %[[I5]]] : memref<5x4xvector<3xf32>>
  // CHECK:                     %[[L0:.*]] = affine.apply #[[$ADD]](%[[I0]], %[[I4]])
  // CHECK:                     %[[L3:.*]] = affine.apply #[[$ADD]](%[[I3]], %[[I6]])
  // CHECK-NEXT:                scf.if
  // CHECK-NEXT:                  %[[SCAL:.*]] = load %{{.*}}[%[[L0]], %[[I1]], %[[I2]], %[[L3]]] : memref<?x?x?x?xf32>
  // CHECK-NEXT:                  %[[RVEC:.*]] = vector.insertelement %[[SCAL]], %[[VEC]][%[[VIDX]] : i32] : vector<3xf32>
  // CHECK-NEXT:                  store %[[RVEC]], %[[ALLOC]][%[[I6]], %[[I5]]] : memref<5x4xvector<3xf32>>
  // CHECK-NEXT:                } else {
  // CHECK-NEXT:                  %[[CVEC:.*]] = vector.insertelement
  // CHECK-NEXT:                  store %[[CVEC]], %[[ALLOC]][%[[I6]], %[[I5]]] : memref<5x4xvector<3xf32>>
  // CHECK-NEXT:                }
  // CHECK-NEXT:              }
  // CHECK-NEXT:            }
  // CHECK-NEXT:          }
  // CHECK-NEXT:          %[[ALLOC_CAST:.*]] = vector.type_cast %[[ALLOC]] : memref<5x4xvector<3xf32>> to memref<vector<5x4x3xf32>>
  // CHECK-NEXT:          %[[LD:.*]] = load %[[ALLOC_CAST]][] : memref<vector<5x4x3xf32>>
  // CHECK-NEXT:          "dummy_use"(%[[LD]]) : (vector<5x4x3xf32>) -> ()
  // CHECK-NEXT:        }
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  // CHECK-NEXT:}

  // Check that I0 + I4 (of size 3) read from first index load(L0, ...) and write into last index store(..., I4)
  // Check that I3 + I6 (of size 5) read from last index load(..., L3) and write into first index store(I6, ...)
  // Other dimensions are just accessed with I1, I2 resp.
  %A = alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
  affine.for %i0 = 0 to %M step 3 {
    affine.for %i1 = 0 to %N {
      affine.for %i2 = 0 to %O {
        affine.for %i3 = 0 to %P step 5 {
          %f = vector.transfer_read %A[%i0, %i1, %i2, %i3], %f0 {permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, 0, d0)>} : memref<?x?x?x?xf32>, vector<5x4x3xf32>
          // Add a dummy use to prevent dead code elimination from removing
          // transfer read ops.
          "dummy_use"(%f) : (vector<5x4x3xf32>) -> ()
        }
      }
    }
  }
  return
}

// -----

// CHECK: #[[$ADD:map.*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL:func @materialize_write(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
func @materialize_write(%M: index, %N: index, %O: index, %P: index) {
  // CHECK-DAG:  %[[ALLOC:.*]] = alloca() : memref<5x4xvector<3xf32>>
  // CHECK-DAG:  %{{.*}} = constant dense<1.000000e+00> : vector<5x4x3xf32>
  // CHECK-DAG:  %[[C0:.*]] = constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = constant 1 : index
  // CHECK-DAG:  %[[C3:.*]] = constant 3 : index
  // CHECK-DAG:  %[[C4:.*]] = constant 4 : index
  // CHECK-DAG:  %[[C5:.*]] = constant 5 : index
  //     CHECK:  %{{.*}} = alloc(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?x?x?xf32>
  // CHECK-NEXT:  affine.for %[[I0:.*]] = 0 to %{{.*}} step 3 {
  // CHECK-NEXT:    affine.for %[[I1:.*]] = 0 to %{{.*}} step 4 {
  // CHECK-NEXT:      affine.for %[[I2:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:        affine.for %[[I3:.*]] = 0 to %{{.*}} step 5 {
  // CHECK-NEXT:          %[[VECTOR_VIEW:.*]] = vector.type_cast {{.*}} : memref<5x4xvector<3xf32>>
  //      CHECK:          store %{{.*}}, {{.*}} : memref<vector<5x4x3xf32>>
  // CHECK-NEXT:          scf.for %[[I4:.*]] = %[[C0]] to %[[C3]] step %[[C1]] {
  // CHECK-NEXT:            scf.for %[[I5:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
  // CHECK-NEXT:              scf.for %[[I6:.*]] = %[[C0]] to %[[C5]] step %[[C1]] {
  // CHECK:                     %[[VIDX:.*]] = index_cast %[[I4]]
  // CHECK:                     %[[S0:.*]] = affine.apply #[[$ADD]](%[[I0]], %[[I4]])
  // CHECK:                     %[[S1:.*]] = affine.apply #[[$ADD]](%[[I1]], %[[I5]])
  // CHECK:                     %[[S3:.*]] = affine.apply #[[$ADD]](%[[I3]], %[[I6]])
  // CHECK-NEXT:                scf.if
  // CHECK-NEXT:                  %[[VEC:.*]] = load {{.*}}[%[[I6]], %[[I5]]] : memref<5x4xvector<3xf32>>
  // CHECK-NEXT:                  %[[SCAL:.*]] = vector.extractelement %[[VEC]][%[[VIDX]] : i32] : vector<3xf32>
  //      CHECK:                  store %[[SCAL]], {{.*}}[%[[S0]], %[[S1]], %[[I2]], %[[S3]]] : memref<?x?x?x?xf32>
  // CHECK-NEXT:                }
  // CHECK-NEXT:              }
  // CHECK-NEXT:            }
  // CHECK-NEXT:          }
  // CHECK-NEXT:        }
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
  // CHECK-NEXT:  return
  // CHECK-NEXT:}
  //
  // Check that I0 + I4 (of size 3) read from last index load(..., I4) and write into first index store(S0, ...)
  // Check that I1 + I5 (of size 4) read from second index load(..., I5, ...) and write into second index store(..., S1, ...)
  // Check that I3 + I6 (of size 5) read from first index load(I6, ...) and write into last index store(..., S3)
  // Other dimension is just accessed with I2.
  %A = alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
  %f1 = constant dense<1.000000e+00> : vector<5x4x3xf32>
  affine.for %i0 = 0 to %M step 3 {
    affine.for %i1 = 0 to %N step 4 {
      affine.for %i2 = 0 to %O {
        affine.for %i3 = 0 to %P step 5 {
          vector.transfer_write %f1, %A[%i0, %i1, %i2, %i3] {permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d1, d0)>} : vector<5x4x3xf32>, memref<?x?x?x?xf32>
        }
      }
    }
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// FULL-UNROLL-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 + 1)>
// FULL-UNROLL-DAG: #[[$MAP2:.*]] = affine_map<()[s0] -> (s0 + 2)>


// CHECK-LABEL: transfer_read_progressive(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[base:[a-zA-Z0-9]+]]: index

// FULL-UNROLL-LABEL: transfer_read_progressive(
//  FULL-UNROLL-SAME:   %[[A:[a-zA-Z0-9]+]]: memref<?x?xf32>,
//  FULL-UNROLL-SAME:   %[[base:[a-zA-Z0-9]+]]: index

func @transfer_read_progressive(%A : memref<?x?xf32>, %base: index) -> vector<3x15xf32> {
  %f7 = constant 7.0: f32
  // CHECK-DAG: %[[C0:.*]] = constant 0 : index
  // CHECK-DAG: %[[splat:.*]] = constant dense<7.000000e+00> : vector<15xf32>
  // CHECK-DAG: %[[alloc:.*]] = alloca() : memref<3xvector<15xf32>>
  // CHECK: %[[cst:.*]] = constant 7.000000e+00 : f32
  // CHECK-DAG: %[[dim:.*]] = dim %[[A]], %[[C0]] : memref<?x?xf32>
  // CHECK: affine.for %[[I:.*]] = 0 to 3 {
  // CHECK:   %[[add:.*]] = affine.apply #[[$MAP0]](%[[I]])[%[[base]]]
  // CHECK:   %[[cond1:.*]] = cmpi slt, %[[add]], %[[dim]] : index
  // CHECK:   scf.if %[[cond1]] {
  // CHECK:     %[[vec_1d:.*]] = vector.transfer_read %[[A]][%[[add]], %[[base]]], %[[cst]] : memref<?x?xf32>, vector<15xf32>
  // CHECK:     store %[[vec_1d]], %[[alloc]][%[[I]]] : memref<3xvector<15xf32>>
  // CHECK:   } else {
  // CHECK:     store %[[splat]], %[[alloc]][%[[I]]] : memref<3xvector<15xf32>>
  // CHECK:   }
  // CHECK: %[[vmemref:.*]] = vector.type_cast %[[alloc]] : memref<3xvector<15xf32>> to memref<vector<3x15xf32>>
  // CHECK: %[[cst:.*]] = load %[[vmemref]][] : memref<vector<3x15xf32>>

  // FULL-UNROLL: %[[VEC0:.*]] = constant dense<7.000000e+00> : vector<3x15xf32>
  // FULL-UNROLL: %[[C0:.*]] = constant 0 : index
  // FULL-UNROLL: %[[SPLAT:.*]] = constant dense<7.000000e+00> : vector<15xf32>
  // FULL-UNROLL: %[[pad:.*]] = constant 7.000000e+00 : f32
  // FULL-UNROLL: %[[DIM:.*]] = dim %[[A]], %[[C0]] : memref<?x?xf32>
  // FULL-UNROLL: cmpi slt, %[[base]], %[[DIM]] : index
  // FULL-UNROLL: %[[VEC1:.*]] = scf.if %{{.*}} -> (vector<3x15xf32>) {
  // FULL-UNROLL:   vector.transfer_read %[[A]][%[[base]], %[[base]]], %[[pad]] : memref<?x?xf32>, vector<15xf32>
  // FULL-UNROLL:   vector.insert %{{.*}}, %[[VEC0]] [0] : vector<15xf32> into vector<3x15xf32>
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: } else {
  // FULL-UNROLL:   vector.insert %{{.*}}, %[[VEC0]] [0] : vector<15xf32> into vector<3x15xf32>
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: }
  // FULL-UNROLL: affine.apply #[[$MAP1]]()[%[[base]]]
  // FULL-UNROLL: cmpi slt, %{{.*}}, %[[DIM]] : index
  // FULL-UNROLL: %[[VEC2:.*]] = scf.if %{{.*}} -> (vector<3x15xf32>) {
  // FULL-UNROLL:   vector.transfer_read %[[A]][%{{.*}}, %[[base]]], %[[pad]] : memref<?x?xf32>, vector<15xf32>
  // FULL-UNROLL:   vector.insert %{{.*}}, %[[VEC1]] [1] : vector<15xf32> into vector<3x15xf32>
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: } else {
  // FULL-UNROLL:   vector.insert %{{.*}}, %[[VEC1]] [1] : vector<15xf32> into vector<3x15xf32>
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: }
  // FULL-UNROLL: affine.apply #[[$MAP2]]()[%[[base]]]
  // FULL-UNROLL: cmpi slt, %{{.*}}, %[[DIM]] : index
  // FULL-UNROLL: %[[VEC3:.*]] = scf.if %{{.*}} -> (vector<3x15xf32>) {
  // FULL-UNROLL:   vector.transfer_read %[[A]][%{{.*}}, %[[base]]], %[[pad]] : memref<?x?xf32>, vector<15xf32>
  // FULL-UNROLL:   vector.insert %{{.*}}, %[[VEC2]] [2] : vector<15xf32> into vector<3x15xf32>
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: } else {
  // FULL-UNROLL:   vector.insert %{{.*}}, %[[VEC2]] [2] : vector<15xf32> into vector<3x15xf32>
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: }

  %f = vector.transfer_read %A[%base, %base], %f7 :
    memref<?x?xf32>, vector<3x15xf32>

  return %f: vector<3x15xf32>
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// FULL-UNROLL-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 + 1)>
// FULL-UNROLL-DAG: #[[$MAP2:.*]] = affine_map<()[s0] -> (s0 + 2)>

// CHECK-LABEL: transfer_write_progressive(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[base:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:   %[[vec:[a-zA-Z0-9]+]]: vector<3x15xf32>
// FULL-UNROLL-LABEL: transfer_write_progressive(
//  FULL-UNROLL-SAME:   %[[A:[a-zA-Z0-9]+]]: memref<?x?xf32>,
//  FULL-UNROLL-SAME:   %[[base:[a-zA-Z0-9]+]]: index,
//  FULL-UNROLL-SAME:   %[[vec:[a-zA-Z0-9]+]]: vector<3x15xf32>
func @transfer_write_progressive(%A : memref<?x?xf32>, %base: index, %vec: vector<3x15xf32>) {
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[alloc:.*]] = alloca() : memref<3xvector<15xf32>>
  // CHECK: %[[vmemref:.*]] = vector.type_cast %[[alloc]] : memref<3xvector<15xf32>> to memref<vector<3x15xf32>>
  // CHECK: store %[[vec]], %[[vmemref]][] : memref<vector<3x15xf32>>
  // CHECK: %[[dim:.*]] = dim %[[A]], %[[C0]] : memref<?x?xf32>
  // CHECK: affine.for %[[I:.*]] = 0 to 3 {
  // CHECK:   %[[add:.*]] = affine.apply #[[$MAP0]](%[[I]])[%[[base]]]
  // CHECK:   %[[cmp:.*]] = cmpi slt, %[[add]], %[[dim]] : index
  // CHECK:   scf.if %[[cmp]] {
  // CHECK:     %[[vec_1d:.*]] = load %0[%[[I]]] : memref<3xvector<15xf32>>
  // CHECK:     vector.transfer_write %[[vec_1d]], %[[A]][%[[add]], %[[base]]] : vector<15xf32>, memref<?x?xf32>
  // CHECK:   }

  // FULL-UNROLL: %[[C0:.*]] = constant 0 : index
  // FULL-UNROLL: %[[DIM:.*]] = dim %[[A]], %[[C0]] : memref<?x?xf32>
  // FULL-UNROLL: %[[CMP0:.*]] = cmpi slt, %[[base]], %[[DIM]] : index
  // FULL-UNROLL: scf.if %[[CMP0]] {
  // FULL-UNROLL:   %[[V0:.*]] = vector.extract %[[vec]][0] : vector<3x15xf32>
  // FULL-UNROLL:   vector.transfer_write %[[V0]], %[[A]][%[[base]], %[[base]]] : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: }
  // FULL-UNROLL: %[[I1:.*]] = affine.apply #[[$MAP1]]()[%[[base]]]
  // FULL-UNROLL: %[[CMP1:.*]] = cmpi slt, %[[I1]], %[[DIM]] : index
  // FULL-UNROLL: scf.if %[[CMP1]] {
  // FULL-UNROLL:   %[[V1:.*]] = vector.extract %[[vec]][1] : vector<3x15xf32>
  // FULL-UNROLL:   vector.transfer_write %[[V1]], %[[A]][%[[I1]], %[[base]]] : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: }
  // FULL-UNROLL: %[[I2:.*]] = affine.apply #[[$MAP2]]()[%[[base]]]
  // FULL-UNROLL: %[[CMP2:.*]] = cmpi slt, %[[I2]], %[[DIM]] : index
  // FULL-UNROLL: scf.if %[[CMP2]] {
  // FULL-UNROLL:   %[[V2:.*]] = vector.extract %[[vec]][2] : vector<3x15xf32>
  // FULL-UNROLL:   vector.transfer_write %[[V2]], %[[A]][%[[I2]], %[[base]]] : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: }

  vector.transfer_write %vec, %A[%base, %base] :
    vector<3x15xf32>, memref<?x?xf32>
  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// FULL-UNROLL-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 + 1)>
// FULL-UNROLL-DAG: #[[$MAP2:.*]] = affine_map<()[s0] -> (s0 + 2)>

// CHECK-LABEL: transfer_write_progressive_unmasked(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[base:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:   %[[vec:[a-zA-Z0-9]+]]: vector<3x15xf32>
// FULL-UNROLL-LABEL: transfer_write_progressive_unmasked(
//  FULL-UNROLL-SAME:   %[[A:[a-zA-Z0-9]+]]: memref<?x?xf32>,
//  FULL-UNROLL-SAME:   %[[base:[a-zA-Z0-9]+]]: index,
//  FULL-UNROLL-SAME:   %[[vec:[a-zA-Z0-9]+]]: vector<3x15xf32>
func @transfer_write_progressive_unmasked(%A : memref<?x?xf32>, %base: index, %vec: vector<3x15xf32>) {
  // CHECK-NOT:    scf.if
  // CHECK-NEXT: %[[alloc:.*]] = alloca() : memref<3xvector<15xf32>>
  // CHECK-NEXT: %[[vmemref:.*]] = vector.type_cast %[[alloc]] : memref<3xvector<15xf32>> to memref<vector<3x15xf32>>
  // CHECK-NEXT: store %[[vec]], %[[vmemref]][] : memref<vector<3x15xf32>>
  // CHECK-NEXT: affine.for %[[I:.*]] = 0 to 3 {
  // CHECK-NEXT:   %[[add:.*]] = affine.apply #[[$MAP0]](%[[I]])[%[[base]]]
  // CHECK-NEXT:   %[[vec_1d:.*]] = load %0[%[[I]]] : memref<3xvector<15xf32>>
  // CHECK-NEXT:   vector.transfer_write %[[vec_1d]], %[[A]][%[[add]], %[[base]]] {masked = [false]} : vector<15xf32>, memref<?x?xf32>

  // FULL-UNROLL: %[[VEC0:.*]] = vector.extract %[[vec]][0] : vector<3x15xf32>
  // FULL-UNROLL: vector.transfer_write %[[VEC0]], %[[A]][%[[base]], %[[base]]] {masked = [false]} : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: %[[I1:.*]] = affine.apply #[[$MAP1]]()[%[[base]]]
  // FULL-UNROLL: %[[VEC1:.*]] = vector.extract %[[vec]][1] : vector<3x15xf32>
  // FULL-UNROLL: vector.transfer_write %2, %[[A]][%[[I1]], %[[base]]] {masked = [false]} : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: %[[I2:.*]] = affine.apply #[[$MAP2]]()[%[[base]]]
  // FULL-UNROLL: %[[VEC2:.*]] = vector.extract %[[vec]][2] : vector<3x15xf32>
  // FULL-UNROLL: vector.transfer_write %[[VEC2:.*]], %[[A]][%[[I2]], %[[base]]] {masked = [false]} : vector<15xf32>, memref<?x?xf32>
  vector.transfer_write %vec, %A[%base, %base] {masked = [false, false]} :
    vector<3x15xf32>, memref<?x?xf32>
  return
}

// -----

// FULL-UNROLL-LABEL: transfer_read_simple
func @transfer_read_simple(%A : memref<2x2xf32>) -> vector<2x2xf32> {
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32
  // FULL-UNROLL-DAG: %[[VC0:.*]] = constant dense<0.000000e+00> : vector<2x2xf32>
  // FULL-UNROLL-DAG: %[[C0:.*]] = constant 0 : index
  // FULL-UNROLL-DAG: %[[C1:.*]] = constant 1 : index
  // FULL-UNROLL: %[[V0:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]]
  // FULL-UNROLL: %[[RES0:.*]] = vector.insert %[[V0]], %[[VC0]] [0] : vector<2xf32> into vector<2x2xf32>
  // FULL-UNROLL: %[[V1:.*]] = vector.transfer_read %{{.*}}[%[[C1]], %[[C0]]]
  // FULL-UNROLL: %[[RES1:.*]] = vector.insert %[[V1]], %[[RES0]] [1] : vector<2xf32> into vector<2x2xf32>
  %0 = vector.transfer_read %A[%c0, %c0], %f0 : memref<2x2xf32>, vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

func @transfer_read_minor_identity(%A : memref<?x?x?x?xf32>) -> vector<3x3xf32> {
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32
  %0 = vector.transfer_read %A[%c0, %c0, %c0, %c0], %f0
    { permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)> }
      : memref<?x?x?x?xf32>, vector<3x3xf32>
  return %0 : vector<3x3xf32>
}

// CHECK-LABEL: transfer_read_minor_identity(
//  CHECK-SAME:   %[[A:.*]]: memref<?x?x?x?xf32>) -> vector<3x3xf32>
//       CHECK:   %[[c2:.*]] = constant 2 : index
//       CHECK:   %[[cst0:.*]] = constant dense<0.000000e+00> : vector<3xf32>
//       CHECK:   %[[m:.*]] = alloca() : memref<3xvector<3xf32>>
//       CHECK:   %[[c0:.*]] = constant 0 : index
//       CHECK:   %[[cst:.*]] = constant 0.000000e+00 : f32
//       CHECK:   %[[d:.*]] = dim %[[A]], %[[c2]] : memref<?x?x?x?xf32>
//       CHECK:   affine.for %[[arg1:.*]] = 0 to 3 {
//       CHECK:      %[[cmp:.*]] = cmpi slt, %[[arg1]], %[[d]] : index
//       CHECK:      scf.if %[[cmp]] {
//       CHECK:        %[[tr:.*]] = vector.transfer_read %[[A]][%[[c0]], %[[c0]], %[[arg1]], %[[c0]]], %[[cst]] : memref<?x?x?x?xf32>, vector<3xf32>
//       CHECK:        store %[[tr]], %[[m]][%[[arg1]]] : memref<3xvector<3xf32>>
//       CHECK:      } else {
//       CHECK:        store %[[cst0]], %[[m]][%[[arg1]]] : memref<3xvector<3xf32>>
//       CHECK:      }
//       CHECK:    }
//       CHECK:    %[[cast:.*]] = vector.type_cast %[[m]] : memref<3xvector<3xf32>> to memref<vector<3x3xf32>>
//       CHECK:    %[[ret:.*]]  = load %[[cast]][] : memref<vector<3x3xf32>>
//       CHECK:    return %[[ret]] : vector<3x3xf32>

func @transfer_write_minor_identity(%A : vector<3x3xf32>, %B : memref<?x?x?x?xf32>) {
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32
  vector.transfer_write %A, %B[%c0, %c0, %c0, %c0]
    { permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)> }
      : vector<3x3xf32>, memref<?x?x?x?xf32>
  return
}

// CHECK-LABEL: transfer_write_minor_identity(
//  CHECK-SAME:   %[[A:.*]]: vector<3x3xf32>,
//  CHECK-SAME:   %[[B:.*]]: memref<?x?x?x?xf32>)
//       CHECK:   %[[c2:.*]] = constant 2 : index
//       CHECK:   %[[m:.*]] = alloca() : memref<3xvector<3xf32>>
//       CHECK:   %[[c0:.*]] = constant 0 : index
//       CHECK:   %[[cast:.*]] = vector.type_cast %[[m]] : memref<3xvector<3xf32>> to memref<vector<3x3xf32>>
//       CHECK:   store %[[A]], %[[cast]][] : memref<vector<3x3xf32>>
//       CHECK:   %[[d:.*]] = dim %[[B]], %[[c2]] : memref<?x?x?x?xf32>
//       CHECK:   affine.for %[[arg2:.*]] = 0 to 3 {
//       CHECK:      %[[cmp:.*]] = cmpi slt, %[[arg2]], %[[d]] : index
//       CHECK:      scf.if %[[cmp]] {
//       CHECK:        %[[tmp:.*]] = load %[[m]][%[[arg2]]] : memref<3xvector<3xf32>>
//       CHECK:        vector.transfer_write %[[tmp]], %[[B]][%[[c0]], %[[c0]], %[[arg2]], %[[c0]]] : vector<3xf32>, memref<?x?x?x?xf32>
//       CHECK:      }
//       CHECK:    }
//       CHECK:    return

// -----

func @transfer_read_strided(%A : memref<8x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>) -> vector<4xf32> {
  %c0 = constant 0 : index
  %f0 = constant 0.0 : f32
  %0 = vector.transfer_read %A[%c0, %c0], %f0
      : memref<8x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>, vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: transfer_read_strided(
// CHECK: scf.for
// CHECK: load

func @transfer_write_strided(%A : vector<4xf32>, %B : memref<8x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>) {
  %c0 = constant 0 : index
  vector.transfer_write %A, %B[%c0, %c0] :
    vector<4xf32>, memref<8x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>
  return
}

// CHECK-LABEL: transfer_write_strided(
// CHECK: scf.for
// CHECK: store
