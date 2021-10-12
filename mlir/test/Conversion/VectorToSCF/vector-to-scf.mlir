// RUN: mlir-opt %s -convert-vector-to-scf -split-input-file -allow-unregistered-dialect | FileCheck %s
// RUN: mlir-opt %s -convert-vector-to-scf=full-unroll=true -split-input-file -allow-unregistered-dialect | FileCheck %s --check-prefix=FULL-UNROLL

// CHECK-LABEL: func @vector_transfer_ops_0d(
//  CHECK-SAME:   %[[MEM:.*]]: memref<f32>) {
func @vector_transfer_ops_0d(%M: memref<f32>) {
    %f0 = arith.constant 0.0 : f32

//  CHECK: %[[V0:.*]] = arith.constant dense<0{{.*}}> : vector<1xf32>
//  CHECK: %[[R0:.*]] = scf.for %[[I:.*]] = {{.*}} iter_args(%[[V0_ITER:.*]] = %[[V0]]) -> (vector<1xf32>) {
//  CHECK:   %[[IDX:.*]] = arith.index_cast %[[I]] : index to i32
//  CHECK:   %[[S:.*]] = memref.load %[[MEM]][] : memref<f32>
//  CHECK:   %[[R_ITER:.*]] = vector.insertelement %[[S]], %[[V0_ITER]][%[[IDX]] : i32] : vector<1xf32>
//  CHECK:   scf.yield %[[R_ITER]] : vector<1xf32>
    %0 = vector.transfer_read %M[], %f0 {permutation_map = affine_map<()->(0)>} :
      memref<f32>, vector<1xf32>

//  CHECK: scf.for %[[J:.*]] = %{{.*}}
//  CHECK:   %[[JDX:.*]] = arith.index_cast %[[J]] : index to i32
//  CHECK:   %[[SS:.*]] = vector.extractelement %[[R0]][%[[JDX]] : i32] : vector<1xf32>
//  CHECK:   memref.store %[[SS]], %[[MEM]][] : memref<f32>
    vector.transfer_write %0, %M[] {permutation_map = affine_map<()->(0)>} :
      vector<1xf32>, memref<f32>

    return
}

// -----

// CHECK-LABEL: func @materialize_read_1d() {
func @materialize_read_1d() {
  %f0 = arith.constant 0.0: f32
  %A = memref.alloc () : memref<7x42xf32>
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
      // CHECK-NEXT: memref.load
      // CHECK-NEXT: vector.insertelement
      // CHECK-NEXT: scf.yield
      // CHECK-NEXT: else
      // CHECK-NEXT: scf.yield
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
  %f0 = arith.constant 0.0: f32
  %A = memref.alloc (%dyn1, %dyn2, %dyn4) : memref<7x?x?x42x?xf32>
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
  // CHECK: %[[tensor:[0-9]+]] = memref.alloc
  // CHECK-NOT: {{.*}} memref.dim %[[tensor]], %c0
  // CHECK-NOT: {{.*}} memref.dim %[[tensor]], %c3
  return
}

// -----

// CHECK: #[[$ADD:map.*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: func @materialize_read(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
func @materialize_read(%M: index, %N: index, %O: index, %P: index) {
  %f0 = arith.constant 0.0: f32
  // CHECK-DAG:  %[[ALLOC:.*]] = memref.alloca() : memref<vector<5x4x3xf32>>
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG:  %[[C5:.*]] = arith.constant 5 : index
  // CHECK:      %{{.*}} = memref.alloc(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?x?x?xf32>
  // CHECK-NEXT:  affine.for %[[I0:.*]] = 0 to %{{.*}} step 3 {
  // CHECK-NEXT:    affine.for %[[I1:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:      affine.for %[[I2:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:        affine.for %[[I3:.*]] = 0 to %{{.*}} step 5 {
  // CHECK:               scf.for %[[I4:.*]] = %[[C0]] to %[[C5]] step %[[C1]] {
  // CHECK:                 scf.if
  // CHECK:                   %[[L3:.*]] = affine.apply #[[$ADD]](%[[I3]], %[[I4]])
  // CHECK:                   scf.for %[[I5:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
  // CHECK:                     %[[VEC:.*]] = scf.for %[[I6:.*]] = %[[C0]] to %[[C3]] step %[[C1]] {{.*}} -> (vector<3xf32>) {
  // CHECK:                       %[[L0:.*]] = affine.apply #[[$ADD]](%[[I0]], %[[I6]])
  // CHECK:                       %[[VIDX:.*]] = arith.index_cast %[[I6]]
  // CHECK:                       scf.if {{.*}} -> (vector<3xf32>) {
  // CHECK-NEXT:                    %[[SCAL:.*]] = memref.load %{{.*}}[%[[L0]], %[[I1]], %[[I2]], %[[L3]]] : memref<?x?x?x?xf32>
  // CHECK-NEXT:                    %[[RVEC:.*]] = vector.insertelement %[[SCAL]], %{{.*}}[%[[VIDX]] : i32] : vector<3xf32>
  // CHECK-NEXT:                    scf.yield
  // CHECK-NEXT:                  } else {
  // CHECK-NEXT:                    scf.yield
  // CHECK-NEXT:                  }
  // CHECK-NEXT:                  scf.yield
  // CHECK-NEXT:                }
  // CHECK-NEXT:                memref.store %[[VEC]], {{.*}} : memref<5x4xvector<3xf32>>
  // CHECK-NEXT:              }
  // CHECK-NEXT:            } else {
  // CHECK-NEXT:              memref.store {{.*}} : memref<5xvector<4x3xf32>>
  // CHECK-NEXT:            }
  // CHECK-NEXT:          }
  // CHECK-NEXT:          %[[LD:.*]] = memref.load %[[ALLOC]][] : memref<vector<5x4x3xf32>>
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
  %A = memref.alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
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
  // CHECK-DAG:  %[[ALLOC:.*]] = memref.alloca() : memref<vector<5x4x3xf32>>
  // CHECK-DAG:  %{{.*}} = arith.constant dense<1.000000e+00> : vector<5x4x3xf32>
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG:  %[[C5:.*]] = arith.constant 5 : index
  // CHECK:      %{{.*}} = memref.alloc(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?x?x?xf32>
  // CHECK-NEXT: affine.for %[[I0:.*]] = 0 to %{{.*}} step 3 {
  // CHECK-NEXT:   affine.for %[[I1:.*]] = 0 to %{{.*}} step 4 {
  // CHECK-NEXT:     affine.for %[[I2:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:       affine.for %[[I3:.*]] = 0 to %{{.*}} step 5 {
  // CHECK:              memref.store %{{.*}}, %[[ALLOC]][] : memref<vector<5x4x3xf32>>
  // CHECK:              %[[VECTOR_VIEW1:.*]] = vector.type_cast %[[ALLOC]] : memref<vector<5x4x3xf32>> to memref<5xvector<4x3xf32>>
  // CHECK:              scf.for %[[I4:.*]] = %[[C0]] to %[[C5]] step %[[C1]] {
  // CHECK:                scf.if
  // CHECK:                  %[[S3:.*]] = affine.apply #[[$ADD]](%[[I3]], %[[I4]])
  // CHECK:                  %[[VECTOR_VIEW2:.*]] = vector.type_cast %[[VECTOR_VIEW1]] : memref<5xvector<4x3xf32>> to memref<5x4xvector<3xf32>>
  // CHECK:                  scf.for %[[I5:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
  // CHECK:                    scf.if
  // CHECK:                      %[[S1:.*]] = affine.apply #[[$ADD]](%[[I1]], %[[I5]])
  // CHECK:                      %[[VEC:.*]] = memref.load %[[VECTOR_VIEW2]][%[[I4]], %[[I5]]] : memref<5x4xvector<3xf32>>
  // CHECK:                      scf.for %[[I6:.*]] = %[[C0]] to %[[C3]] step %[[C1]] {
  // CHECK:                        %[[S0:.*]] = affine.apply #[[$ADD]](%[[I0]], %[[I6]])
  // CHECK:                        %[[VIDX:.*]] = arith.index_cast %[[I6]]
  // CHECK:                        scf.if
  // CHECK:                          %[[SCAL:.*]] = vector.extractelement %[[VEC]][%[[VIDX]] : i32] : vector<3xf32>
  // CHECK:                          memref.store %[[SCAL]], {{.*}}[%[[S0]], %[[S1]], %[[I2]], %[[S3]]] : memref<?x?x?x?xf32>
  // CHECK:                        }
  // CHECK:                      }
  // CHECK:                    }
  // CHECK:                  }
  // CHECK:                }
  // CHECK:              }
  // CHECK:            }
  // CHECK:          }
  // CHECK:        }
  // CHECK:      }
  // CHECK:      return

  // Check that I0 + I4 (of size 3) read from last index load(..., I4) and write into first index store(S0, ...)
  // Check that I1 + I5 (of size 4) read from second index load(..., I5, ...) and write into second index store(..., S1, ...)
  // Check that I3 + I6 (of size 5) read from first index load(I6, ...) and write into last index store(..., S3)
  // Other dimension is just accessed with I2.
  %A = memref.alloc (%M, %N, %O, %P) : memref<?x?x?x?xf32, 0>
  %f1 = arith.constant dense<1.000000e+00> : vector<5x4x3xf32>
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
  %f7 = arith.constant 7.0: f32
  // CHECK-DAG: %[[C7:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[splat:.*]] = arith.constant dense<7.000000e+00> : vector<15xf32>
  // CHECK-DAG: %[[alloc:.*]] = memref.alloca() : memref<vector<3x15xf32>>
  // CHECK:     %[[alloc_casted:.*]] = vector.type_cast %[[alloc]] : memref<vector<3x15xf32>> to memref<3xvector<15xf32>>
  // CHECK:     scf.for %[[I:.*]] = %[[C0]] to %[[C3]]
  // CHECK:       %[[dim:.*]] = memref.dim %[[A]], %[[C0]] : memref<?x?xf32>
  // CHECK:       %[[add:.*]] = affine.apply #[[$MAP0]](%[[I]])[%[[base]]]
  // CHECK:       %[[cond1:.*]] = arith.cmpi sgt, %[[dim]], %[[add]] : index
  // CHECK:       scf.if %[[cond1]] {
  // CHECK:         %[[vec_1d:.*]] = vector.transfer_read %[[A]][%{{.*}}, %[[base]]], %[[C7]] : memref<?x?xf32>, vector<15xf32>
  // CHECK:         memref.store %[[vec_1d]], %[[alloc_casted]][%[[I]]] : memref<3xvector<15xf32>>
  // CHECK:       } else {
  // CHECK:         store %[[splat]], %[[alloc_casted]][%[[I]]] : memref<3xvector<15xf32>>
  // CHECK:       }
  // CHECK:     }
  // CHECK:     %[[cst:.*]] = memref.load %[[alloc]][] : memref<vector<3x15xf32>>

  // FULL-UNROLL: %[[C7:.*]] = arith.constant 7.000000e+00 : f32
  // FULL-UNROLL: %[[VEC0:.*]] = arith.constant dense<7.000000e+00> : vector<3x15xf32>
  // FULL-UNROLL: %[[C0:.*]] = arith.constant 0 : index
  // FULL-UNROLL: %[[DIM:.*]] = memref.dim %[[A]], %[[C0]] : memref<?x?xf32>
  // FULL-UNROLL: cmpi sgt, %[[DIM]], %[[base]] : index
  // FULL-UNROLL: %[[VEC1:.*]] = scf.if %{{.*}} -> (vector<3x15xf32>) {
  // FULL-UNROLL:   vector.transfer_read %[[A]][%[[base]], %[[base]]], %[[C7]] : memref<?x?xf32>, vector<15xf32>
  // FULL-UNROLL:   vector.insert %{{.*}}, %[[VEC0]] [0] : vector<15xf32> into vector<3x15xf32>
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: } else {
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: }
  // FULL-UNROLL: affine.apply #[[$MAP1]]()[%[[base]]]
  // FULL-UNROLL: cmpi sgt, %{{.*}}, %{{.*}} : index
  // FULL-UNROLL: %[[VEC2:.*]] = scf.if %{{.*}} -> (vector<3x15xf32>) {
  // FULL-UNROLL:   vector.transfer_read %[[A]][%{{.*}}, %[[base]]], %[[C7]] : memref<?x?xf32>, vector<15xf32>
  // FULL-UNROLL:   vector.insert %{{.*}}, %[[VEC1]] [1] : vector<15xf32> into vector<3x15xf32>
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: } else {
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: }
  // FULL-UNROLL: affine.apply #[[$MAP2]]()[%[[base]]]
  // FULL-UNROLL: cmpi sgt, %{{.*}}, %{{.*}} : index
  // FULL-UNROLL: %[[VEC3:.*]] = scf.if %{{.*}} -> (vector<3x15xf32>) {
  // FULL-UNROLL:   vector.transfer_read %[[A]][%{{.*}}, %[[base]]], %[[C7]] : memref<?x?xf32>, vector<15xf32>
  // FULL-UNROLL:   vector.insert %{{.*}}, %[[VEC2]] [2] : vector<15xf32> into vector<3x15xf32>
  // FULL-UNROLL:   scf.yield %{{.*}} : vector<3x15xf32>
  // FULL-UNROLL: } else {
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
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK:     %[[alloc:.*]] = memref.alloca() : memref<vector<3x15xf32>>
  // CHECK:     memref.store %[[vec]], %[[alloc]][] : memref<vector<3x15xf32>>
  // CHECK:     %[[vmemref:.*]] = vector.type_cast %[[alloc]] : memref<vector<3x15xf32>> to memref<3xvector<15xf32>>
  // CHECK:     scf.for %[[I:.*]] = %[[C0]] to %[[C3]]
  // CHECK:       %[[dim:.*]] = memref.dim %[[A]], %[[C0]] : memref<?x?xf32>
  // CHECK:       %[[add:.*]] = affine.apply #[[$MAP0]](%[[I]])[%[[base]]]
  // CHECK:       %[[cmp:.*]] = arith.cmpi sgt, %[[dim]], %[[add]] : index
  // CHECK:       scf.if %[[cmp]] {
  // CHECK:         %[[vec_1d:.*]] = memref.load %[[vmemref]][%[[I]]] : memref<3xvector<15xf32>>
  // CHECK:         vector.transfer_write %[[vec_1d]], %[[A]][{{.*}}, %[[base]]] : vector<15xf32>, memref<?x?xf32>
  // CHECK:       }
  // CHECK:     }

  // FULL-UNROLL: %[[C0:.*]] = arith.constant 0 : index
  // FULL-UNROLL: %[[DIM:.*]] = memref.dim %[[A]], %[[C0]] : memref<?x?xf32>
  // FULL-UNROLL: %[[CMP0:.*]] = arith.cmpi sgt, %[[DIM]], %[[base]] : index
  // FULL-UNROLL: scf.if %[[CMP0]] {
  // FULL-UNROLL:   %[[V0:.*]] = vector.extract %[[vec]][0] : vector<3x15xf32>
  // FULL-UNROLL:   vector.transfer_write %[[V0]], %[[A]][%[[base]], %[[base]]] : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: }
  // FULL-UNROLL: %[[I1:.*]] = affine.apply #[[$MAP1]]()[%[[base]]]
  // FULL-UNROLL: %[[CMP1:.*]] = arith.cmpi sgt, %{{.*}}, %[[I1]] : index
  // FULL-UNROLL: scf.if %[[CMP1]] {
  // FULL-UNROLL:   %[[V1:.*]] = vector.extract %[[vec]][1] : vector<3x15xf32>
  // FULL-UNROLL:   vector.transfer_write %[[V1]], %[[A]][%{{.*}}, %[[base]]] : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: }
  // FULL-UNROLL: %[[I2:.*]] = affine.apply #[[$MAP2]]()[%[[base]]]
  // FULL-UNROLL: %[[CMP2:.*]] = arith.cmpi sgt, %{{.*}}, %[[I2]] : index
  // FULL-UNROLL: scf.if %[[CMP2]] {
  // FULL-UNROLL:   %[[V2:.*]] = vector.extract %[[vec]][2] : vector<3x15xf32>
  // FULL-UNROLL:   vector.transfer_write %[[V2]], %[[A]][%{{.*}}, %[[base]]] : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: }

  vector.transfer_write %vec, %A[%base, %base] :
    vector<3x15xf32>, memref<?x?xf32>
  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>

// FULL-UNROLL-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 + 1)>
// FULL-UNROLL-DAG: #[[$MAP2:.*]] = affine_map<()[s0] -> (s0 + 2)>

// CHECK-LABEL: transfer_write_progressive_inbounds(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]+]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[base:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:   %[[vec:[a-zA-Z0-9]+]]: vector<3x15xf32>
// FULL-UNROLL-LABEL: transfer_write_progressive_inbounds(
//  FULL-UNROLL-SAME:   %[[A:[a-zA-Z0-9]+]]: memref<?x?xf32>,
//  FULL-UNROLL-SAME:   %[[base:[a-zA-Z0-9]+]]: index,
//  FULL-UNROLL-SAME:   %[[vec:[a-zA-Z0-9]+]]: vector<3x15xf32>
func @transfer_write_progressive_inbounds(%A : memref<?x?xf32>, %base: index, %vec: vector<3x15xf32>) {
  // CHECK-NOT:    scf.if
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[C3:.*]] = arith.constant 3 : index
  // CHECK:      %[[alloc:.*]] = memref.alloca() : memref<vector<3x15xf32>>
  // CHECK-NEXT: memref.store %[[vec]], %[[alloc]][] : memref<vector<3x15xf32>>
  // CHECK-NEXT: %[[vmemref:.*]] = vector.type_cast %[[alloc]] : memref<vector<3x15xf32>> to memref<3xvector<15xf32>>
  // CHECK-NEXT: scf.for %[[I:.*]] = %[[C0]] to %[[C3]]
  // CHECK-NEXT:   %[[add:.*]] = affine.apply #[[$MAP0]](%[[I]])[%[[base]]]
  // CHECK-NEXT:   %[[vec_1d:.*]] = memref.load %[[vmemref]][%[[I]]] : memref<3xvector<15xf32>>
  // CHECK-NEXT:   vector.transfer_write %[[vec_1d]], %[[A]][%[[add]], %[[base]]] {in_bounds = [true]} : vector<15xf32>, memref<?x?xf32>

  // FULL-UNROLL: %[[VEC0:.*]] = vector.extract %[[vec]][0] : vector<3x15xf32>
  // FULL-UNROLL: vector.transfer_write %[[VEC0]], %[[A]][%[[base]], %[[base]]] {in_bounds = [true]} : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: %[[I1:.*]] = affine.apply #[[$MAP1]]()[%[[base]]]
  // FULL-UNROLL: %[[VEC1:.*]] = vector.extract %[[vec]][1] : vector<3x15xf32>
  // FULL-UNROLL: vector.transfer_write %2, %[[A]][%[[I1]], %[[base]]] {in_bounds = [true]} : vector<15xf32>, memref<?x?xf32>
  // FULL-UNROLL: %[[I2:.*]] = affine.apply #[[$MAP2]]()[%[[base]]]
  // FULL-UNROLL: %[[VEC2:.*]] = vector.extract %[[vec]][2] : vector<3x15xf32>
  // FULL-UNROLL: vector.transfer_write %[[VEC2:.*]], %[[A]][%[[I2]], %[[base]]] {in_bounds = [true]} : vector<15xf32>, memref<?x?xf32>
  vector.transfer_write %vec, %A[%base, %base] {in_bounds = [true, true]} :
    vector<3x15xf32>, memref<?x?xf32>
  return
}

// -----

// FULL-UNROLL-LABEL: transfer_read_simple
func @transfer_read_simple(%A : memref<2x2xf32>) -> vector<2x2xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  // FULL-UNROLL-DAG: %[[VC0:.*]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
  // FULL-UNROLL-DAG: %[[C0:.*]] = arith.constant 0 : index
  // FULL-UNROLL-DAG: %[[C1:.*]] = arith.constant 1 : index
  // FULL-UNROLL: %[[V0:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]]]
  // FULL-UNROLL: %[[RES0:.*]] = vector.insert %[[V0]], %[[VC0]] [0] : vector<2xf32> into vector<2x2xf32>
  // FULL-UNROLL: %[[V1:.*]] = vector.transfer_read %{{.*}}[%[[C1]], %[[C0]]]
  // FULL-UNROLL: %[[RES1:.*]] = vector.insert %[[V1]], %[[RES0]] [1] : vector<2xf32> into vector<2x2xf32>
  %0 = vector.transfer_read %A[%c0, %c0], %f0 : memref<2x2xf32>, vector<2x2xf32>
  return %0 : vector<2x2xf32>
}

func @transfer_read_minor_identity(%A : memref<?x?x?x?xf32>) -> vector<3x3xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %A[%c0, %c0, %c0, %c0], %f0
    { permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)> }
      : memref<?x?x?x?xf32>, vector<3x3xf32>
  return %0 : vector<3x3xf32>
}

// CHECK-LABEL: transfer_read_minor_identity(
//  CHECK-SAME: %[[A:.*]]: memref<?x?x?x?xf32>) -> vector<3x3xf32>
//  CHECK-DAG:    %[[c0:.*]] = arith.constant 0 : index
//  CHECK-DAG:    %[[c1:.*]] = arith.constant 1 : index
//  CHECK-DAG:    %[[c2:.*]] = arith.constant 2 : index
//  CHECK-DAG:    %[[c3:.*]] = arith.constant 3 : index
//  CHECK-DAG:    %[[f0:.*]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG:    %[[cst0:.*]] = arith.constant dense<0.000000e+00> : vector<3xf32>
//  CHECK:        %[[m:.*]] = memref.alloca() : memref<vector<3x3xf32>>
//  CHECK:        %[[cast:.*]] = vector.type_cast %[[m]] : memref<vector<3x3xf32>> to memref<3xvector<3xf32>>
//  CHECK:        scf.for %[[arg1:.*]] = %[[c0]] to %[[c3]]
//  CHECK:          %[[d:.*]] = memref.dim %[[A]], %[[c2]] : memref<?x?x?x?xf32>
//  CHECK:          %[[cmp:.*]] = arith.cmpi sgt, %[[d]], %[[arg1]] : index
//  CHECK:          scf.if %[[cmp]] {
//  CHECK:            %[[tr:.*]] = vector.transfer_read %[[A]][%c0, %c0, %[[arg1]], %c0], %[[f0]] : memref<?x?x?x?xf32>, vector<3xf32>
//  CHECK:            memref.store %[[tr]], %[[cast]][%[[arg1]]] : memref<3xvector<3xf32>>
//  CHECK:          } else {
//  CHECK:            memref.store %[[cst0]], %[[cast]][%[[arg1]]] : memref<3xvector<3xf32>>
//  CHECK:          }
//  CHECK:        }
//  CHECK:        %[[ret:.*]]  = memref.load %[[m]][] : memref<vector<3x3xf32>>
//  CHECK:        return %[[ret]] : vector<3x3xf32>

func @transfer_write_minor_identity(%A : vector<3x3xf32>, %B : memref<?x?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  vector.transfer_write %A, %B[%c0, %c0, %c0, %c0]
    { permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)> }
      : vector<3x3xf32>, memref<?x?x?x?xf32>
  return
}

// CHECK-LABEL: transfer_write_minor_identity(
// CHECK-SAME:      %[[A:.*]]: vector<3x3xf32>,
// CHECK-SAME:      %[[B:.*]]: memref<?x?x?x?xf32>)
// CHECK-DAG:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[c2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[c3:.*]] = arith.constant 3 : index
// CHECK:         %[[m:.*]] = memref.alloca() : memref<vector<3x3xf32>>
// CHECK:         memref.store %[[A]], %[[m]][] : memref<vector<3x3xf32>>
// CHECK:         %[[cast:.*]] = vector.type_cast %[[m]] : memref<vector<3x3xf32>> to memref<3xvector<3xf32>>
// CHECK:         scf.for %[[arg2:.*]] = %[[c0]] to %[[c3]]
// CHECK:           %[[d:.*]] = memref.dim %[[B]], %[[c2]] : memref<?x?x?x?xf32>
// CHECK:           %[[cmp:.*]] = arith.cmpi sgt, %[[d]], %[[arg2]] : index
// CHECK:           scf.if %[[cmp]] {
// CHECK:             %[[tmp:.*]] = memref.load %[[cast]][%[[arg2]]] : memref<3xvector<3xf32>>
// CHECK:             vector.transfer_write %[[tmp]], %[[B]][%[[c0]], %[[c0]], %[[arg2]], %[[c0]]] : vector<3xf32>, memref<?x?x?x?xf32>
// CHECK:           }
// CHECK:         }
// CHECK:         return


// -----

func @transfer_read_strided(%A : memref<8x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %A[%c0, %c0], %f0
      : memref<8x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>, vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: transfer_read_strided(
// CHECK: scf.for
// CHECK: memref.load

func @transfer_write_strided(%A : vector<4xf32>, %B : memref<8x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %A, %B[%c0, %c0] :
    vector<4xf32>, memref<8x4xf32, affine_map<(d0, d1) -> (d0 + d1 * 8)>>
  return
}

// CHECK-LABEL: transfer_write_strided(
// CHECK: scf.for
// CHECK: store
