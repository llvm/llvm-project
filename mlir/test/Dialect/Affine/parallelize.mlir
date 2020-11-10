// RUN: mlir-opt %s -allow-unregistered-dialect -affine-parallelize| FileCheck %s

// CHECK-LABEL:    func @reduce_window_max() {
func @reduce_window_max() {
  %cst = constant 0.000000e+00 : f32
  %0 = alloc() : memref<1x8x8x64xf32>
  %1 = alloc() : memref<1x18x18x64xf32>
  affine.for %arg0 = 0 to 1 {
    affine.for %arg1 = 0 to 8 {
      affine.for %arg2 = 0 to 8 {
        affine.for %arg3 = 0 to 64 {
          affine.store %cst, %0[%arg0, %arg1, %arg2, %arg3] : memref<1x8x8x64xf32>
        }
      }
    }
  }
  affine.for %arg0 = 0 to 1 {
    affine.for %arg1 = 0 to 8 {
      affine.for %arg2 = 0 to 8 {
        affine.for %arg3 = 0 to 64 {
          affine.for %arg4 = 0 to 1 {
            affine.for %arg5 = 0 to 3 {
              affine.for %arg6 = 0 to 3 {
                affine.for %arg7 = 0 to 1 {
                  %2 = affine.load %0[%arg0, %arg1, %arg2, %arg3] : memref<1x8x8x64xf32>
                  %3 = affine.load %1[%arg0 + %arg4, %arg1 * 2 + %arg5, %arg2 * 2 + %arg6, %arg3 + %arg7] : memref<1x18x18x64xf32>
                  %4 = cmpf "ogt", %2, %3 : f32
                  %5 = select %4, %2, %3 : f32
                  affine.store %5, %0[%arg0, %arg1, %arg2, %arg3] : memref<1x8x8x64xf32>
                }
              }
            }
          }
        }
      }
    }
  }
  return
}

// CHECK:        %[[cst:.*]] = constant 0.000000e+00 : f32
// CHECK:        %[[v0:.*]] = alloc() : memref<1x8x8x64xf32>
// CHECK:        %[[v1:.*]] = alloc() : memref<1x18x18x64xf32>
// CHECK:        affine.parallel (%[[arg0:.*]]) = (0) to (1) {
// CHECK:          affine.parallel (%[[arg1:.*]]) = (0) to (8) {
// CHECK:            affine.parallel (%[[arg2:.*]]) = (0) to (8) {
// CHECK:              affine.parallel (%[[arg3:.*]]) = (0) to (64) {
// CHECK:                affine.store %[[cst]], %[[v0]][%[[arg0]], %[[arg1]], %[[arg2]], %[[arg3]]] : memref<1x8x8x64xf32>
// CHECK:              }
// CHECK:            }
// CHECK:          }
// CHECK:        }
// CHECK:        affine.parallel (%[[a0:.*]]) = (0) to (1) {
// CHECK:          affine.parallel (%[[a1:.*]]) = (0) to (8) {
// CHECK:            affine.parallel (%[[a2:.*]]) = (0) to (8) {
// CHECK:              affine.parallel (%[[a3:.*]]) = (0) to (64) {
// CHECK:                affine.parallel (%[[a4:.*]]) = (0) to (1) {
// CHECK:                  affine.for %[[a5:.*]] = 0 to 3 {
// CHECK:                    affine.for %[[a6:.*]] = 0 to 3 {
// CHECK:                      affine.parallel (%[[a7:.*]]) = (0) to (1) {
// CHECK:                        %[[lhs:.*]] = affine.load %[[v0]][%[[a0]], %[[a1]], %[[a2]], %[[a3]]] : memref<1x8x8x64xf32>
// CHECK:                        %[[rhs:.*]] = affine.load %[[v1]][%[[a0]] + %[[a4]], %[[a1]] * 2 + %[[a5]], %[[a2]] * 2 + %[[a6]], %[[a3]] + %[[a7]]] : memref<1x18x18x64xf32>
// CHECK:                        %[[res:.*]] = cmpf "ogt", %[[lhs]], %[[rhs]] : f32
// CHECK:                        %[[sel:.*]] = select %[[res]], %[[lhs]], %[[rhs]] : f32
// CHECK:                        affine.store %[[sel]], %[[v0]][%[[a0]], %[[a1]], %[[a2]], %[[a3]]] : memref<1x8x8x64xf32>
// CHECK:                      }
// CHECK:                    }
// CHECK:                  }
// CHECK:                }
// CHECK:              }
// CHECK:            }
// CHECK:          }
// CHECK:        }
// CHECK:      }

func @loop_nest_3d_outer_two_parallel(%N : index) {
  %0 = alloc() : memref<1024 x 1024 x vector<64xf32>>
  %1 = alloc() : memref<1024 x 1024 x vector<64xf32>>
  %2 = alloc() : memref<1024 x 1024 x vector<64xf32>>
  affine.for %i = 0 to %N {
    affine.for %j = 0 to %N {
      %7 = affine.load %2[%i, %j] : memref<1024x1024xvector<64xf32>>
      affine.for %k = 0 to %N {
        %5 = affine.load %0[%i, %k] : memref<1024x1024xvector<64xf32>>
        %6 = affine.load %1[%k, %j] : memref<1024x1024xvector<64xf32>>
        %8 = mulf %5, %6 : vector<64xf32>
        %9 = addf %7, %8 : vector<64xf32>
        affine.store %9, %2[%i, %j] : memref<1024x1024xvector<64xf32>>
      }
    }
  }
  return
}

// CHECK:      affine.parallel (%[[arg1:.*]]) = (0) to (symbol(%arg0)) {
// CHECK-NEXT:        affine.parallel (%[[arg2:.*]]) = (0) to (symbol(%arg0)) {
// CHECK:          affine.for %[[arg3:.*]] = 0 to %arg0 {

// CHECK-LABEL: unknown_op_conservative
func @unknown_op_conservative() {
  affine.for %i = 0 to 10 {
// CHECK:  affine.for %[[arg1:.*]] = 0 to 10 {
    "unknown"() : () -> ()
  }
  return
}

// CHECK-LABEL: non_affine_load
func @non_affine_load() {
  %0 = alloc() : memref<100 x f32>
  affine.for %i = 0 to 100 {
// CHECK:  affine.for %{{.*}} = 0 to 100 {
    load %0[%i] : memref<100 x f32>
  }
  return
}
