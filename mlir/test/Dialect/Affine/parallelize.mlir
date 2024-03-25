// RUN: mlir-opt %s -allow-unregistered-dialect -affine-parallelize | FileCheck %s
// RUN: mlir-opt %s -allow-unregistered-dialect -affine-parallelize='max-nested=1' | FileCheck --check-prefix=MAX-NESTED %s
// RUN: mlir-opt %s -allow-unregistered-dialect -affine-parallelize='parallel-reductions=1' | FileCheck --check-prefix=REDUCE %s

// CHECK-LABEL:    func @reduce_window_max() {
func.func @reduce_window_max() {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.alloc() : memref<1x8x8x64xf32>
  %1 = memref.alloc() : memref<1x18x18x64xf32>
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
                  %4 = arith.cmpf ogt, %2, %3 : f32
                  %5 = arith.select %4, %2, %3 : f32
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

// CHECK:        %[[cst:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[v0:.*]] = memref.alloc() : memref<1x8x8x64xf32>
// CHECK:        %[[v1:.*]] = memref.alloc() : memref<1x18x18x64xf32>
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
// CHECK:                        %[[res:.*]] = arith.cmpf ogt, %[[lhs]], %[[rhs]] : f32
// CHECK:                        %[[sel:.*]] = arith.select %[[res]], %[[lhs]], %[[rhs]] : f32
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

func.func @loop_nest_3d_outer_two_parallel(%N : index) {
  %0 = memref.alloc() : memref<1024 x 1024 x vector<64xf32>>
  %1 = memref.alloc() : memref<1024 x 1024 x vector<64xf32>>
  %2 = memref.alloc() : memref<1024 x 1024 x vector<64xf32>>
  affine.for %i = 0 to %N {
    affine.for %j = 0 to %N {
      %7 = affine.load %2[%i, %j] : memref<1024x1024xvector<64xf32>>
      affine.for %k = 0 to %N {
        %5 = affine.load %0[%i, %k] : memref<1024x1024xvector<64xf32>>
        %6 = affine.load %1[%k, %j] : memref<1024x1024xvector<64xf32>>
        %8 = arith.mulf %5, %6 : vector<64xf32>
        %9 = arith.addf %7, %8 : vector<64xf32>
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
func.func @unknown_op_conservative() {
  affine.for %i = 0 to 10 {
// CHECK:  affine.for %[[arg1:.*]] = 0 to 10 {
    "unknown"() : () -> ()
  }
  return
}

// CHECK-LABEL: non_affine_load
func.func @non_affine_load() {
  %0 = memref.alloc() : memref<100 x f32>
  affine.for %i = 0 to 100 {
// CHECK:  affine.for %{{.*}} = 0 to 100 {
    memref.load %0[%i] : memref<100 x f32>
  }
  return
}

// CHECK-LABEL: for_with_minmax
func.func @for_with_minmax(%m: memref<?xf32>, %lb0: index, %lb1: index,
                      %ub0: index, %ub1: index) {
  // CHECK: affine.parallel (%{{.*}}) = (max(%{{.*}}, %{{.*}})) to (min(%{{.*}}, %{{.*}}))
  affine.for %i = max affine_map<(d0, d1) -> (d0, d1)>(%lb0, %lb1)
          to min affine_map<(d0, d1) -> (d0, d1)>(%ub0, %ub1) {
    affine.load %m[%i] : memref<?xf32>
  }
  return
}

// CHECK-LABEL: nested_for_with_minmax
func.func @nested_for_with_minmax(%m: memref<?xf32>, %lb0: index,
                             %ub0: index, %ub1: index) {
  // CHECK: affine.parallel (%[[I:.*]]) =
  affine.for %j = 0 to 10 {
    // CHECK: affine.parallel (%{{.*}}) = (max(%{{.*}}, %[[I]])) to (min(%{{.*}}, %{{.*}}))
    affine.for %i = max affine_map<(d0, d1) -> (d0, d1)>(%lb0, %j)
            to min affine_map<(d0, d1) -> (d0, d1)>(%ub0, %ub1) {
      affine.load %m[%i] : memref<?xf32>
    }
  }
  return
}

// MAX-NESTED-LABEL: @max_nested
func.func @max_nested(%m: memref<?x?xf32>, %lb0: index, %lb1: index,
                 %ub0: index, %ub1: index) {
  // MAX-NESTED: affine.parallel
  affine.for %i = affine_map<(d0) -> (d0)>(%lb0) to affine_map<(d0) -> (d0)>(%ub0) {
    // MAX-NESTED: affine.for
    affine.for %j = affine_map<(d0) -> (d0)>(%lb1) to affine_map<(d0) -> (d0)>(%ub1) {
      affine.load %m[%i, %j] : memref<?x?xf32>
    }
  }
  return
}

// MAX-NESTED-LABEL: @max_nested_1
func.func @max_nested_1(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) {
  %0 = memref.alloc() : memref<4096x4096xf32>
  // MAX-NESTED: affine.parallel
  affine.for %arg3 = 0 to 4096 {
    // MAX-NESTED-NEXT: affine.for
    affine.for %arg4 = 0 to 4096 {
      // MAX-NESTED-NEXT: affine.for
      affine.for %arg5 = 0 to 4096 {
        %1 = affine.load %arg0[%arg3, %arg5] : memref<4096x4096xf32>
        %2 = affine.load %arg1[%arg5, %arg4] : memref<4096x4096xf32>
        %3 = affine.load %0[%arg3, %arg4] : memref<4096x4096xf32>
        %4 = arith.mulf %1, %2 : f32
        %5 = arith.addf %3, %4 : f32
        affine.store %5, %0[%arg3, %arg4] : memref<4096x4096xf32>
      }
    }
  }
  return
}

// CHECK-LABEL: @iter_args
// REDUCE-LABEL: @iter_args
func.func @iter_args(%in: memref<10xf32>) {
  // REDUCE: %[[init:.*]] = arith.constant
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK-NOT: affine.parallel
  // REDUCE: %[[reduced:.*]] = affine.parallel (%{{.*}}) = (0) to (10) reduce ("addf")
  %final_red = affine.for %i = 0 to 10 iter_args(%red_iter = %cst) -> (f32) {
    // REDUCE: %[[red_value:.*]] = affine.load
    %ld = affine.load %in[%i] : memref<10xf32>
    // REDUCE-NOT: arith.addf
    %add = arith.addf %red_iter, %ld : f32
    // REDUCE: affine.yield %[[red_value]]
    affine.yield %add : f32
  }
  // REDUCE: arith.addf %[[init]], %[[reduced]]
  return
}

// CHECK-LABEL: @nested_iter_args
// REDUCE-LABEL: @nested_iter_args
func.func @nested_iter_args(%in: memref<20x10xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: affine.parallel
  affine.for %i = 0 to 20 {
    // CHECK-NOT: affine.parallel
    // REDUCE: affine.parallel
    // REDUCE: reduce ("addf")
    %final_red = affine.for %j = 0 to 10 iter_args(%red_iter = %cst) -> (f32) {
      %ld = affine.load %in[%i, %j] : memref<20x10xf32>
      %add = arith.addf %red_iter, %ld : f32
      affine.yield %add : f32
    }
  }
  return
}

// REDUCE-LABEL: @strange_butterfly
func.func @strange_butterfly() {
  %cst1 = arith.constant 0.0 : f32
  %cst2 = arith.constant 1.0 : f32
  // REDUCE-NOT: affine.parallel
  affine.for %i = 0 to 10 iter_args(%it1 = %cst1, %it2 = %cst2) -> (f32, f32) {
    %0 = arith.addf %it1, %it2 : f32
    affine.yield %0, %0 : f32, f32
  }
  return
}

// An iter arg is used more than once. This is not a simple reduction and
// should not be parallelized.
// REDUCE-LABEL: @repeated_use
func.func @repeated_use() {
  %cst1 = arith.constant 0.0 : f32
  // REDUCE-NOT: affine.parallel
  affine.for %i = 0 to 10 iter_args(%it1 = %cst1) -> (f32) {
    %0 = arith.addf %it1, %it1 : f32
    affine.yield %0 : f32
  }
  return
}

// An iter arg is used in the chain of operations defining the value being
// reduced, this is not a simple reduction and should not be parallelized.
// REDUCE-LABEL: @use_in_backward_slice
func.func @use_in_backward_slice() {
  %cst1 = arith.constant 0.0 : f32
  %cst2 = arith.constant 1.0 : f32
  // REDUCE-NOT: affine.parallel
  affine.for %i = 0 to 10 iter_args(%it1 = %cst1, %it2 = %cst2) -> (f32, f32) {
    %0 = "test.some_modification"(%it2) : (f32) -> f32
    %1 = arith.addf %it1, %0 : f32
    affine.yield %1, %1 : f32, f32
  }
  return
}

// REDUCE-LABEL: @nested_min_max
// CHECK-LABEL: @nested_min_max
// CHECK: (%{{.*}}, %[[LB0:.*]]: index, %[[UB0:.*]]: index, %[[UB1:.*]]: index)
func.func @nested_min_max(%m: memref<?xf32>, %lb0: index,
                     %ub0: index, %ub1: index) {
  // CHECK: affine.parallel (%[[J:.*]]) =
  affine.for %j = 0 to 10 {
    // CHECK: affine.parallel (%{{.*}}) = (max(%[[LB0]], %[[J]]))
    // CHECK:                          to (min(%[[UB0]], %[[UB1]]))
    affine.for %i = max affine_map<(d0, d1) -> (d0, d1)>(%lb0, %j)
            to min affine_map<(d0, d1) -> (d0, d1)>(%ub0, %ub1) {
      affine.load %m[%i] : memref<?xf32>
    }
  }
  return
}

// Test in the presence of locally allocated memrefs.

// CHECK: func @local_alloc
func.func @local_alloc() {
  %cst = arith.constant 0.0 : f32
  affine.for %i = 0 to 100 {
    %m = memref.alloc() : memref<1xf32>
    %ma = memref.alloca() : memref<1xf32>
    affine.store %cst, %m[0] : memref<1xf32>
  }
  // CHECK: affine.parallel
  return
}

// CHECK: func @local_alloc_cast
func.func @local_alloc_cast() {
  %cst = arith.constant 0.0 : f32
  affine.for %i = 0 to 100 {
    %m = memref.alloc() : memref<128xf32>
    affine.for %j = 0 to 128 {
      affine.store %cst, %m[%j] : memref<128xf32>
    }
    affine.for %j = 0 to 128 {
      affine.store %cst, %m[0] : memref<128xf32>
    }
    %r = memref.reinterpret_cast %m to offset: [0], sizes: [8, 16],
           strides: [16, 1] : memref<128xf32> to memref<8x16xf32>
    affine.for %j = 0 to 8 {
      affine.store %cst, %r[%j, %j] : memref<8x16xf32>
    }
  }
  // CHECK: affine.parallel
  // CHECK:   affine.parallel
  // CHECK:   }
  // CHECK:   affine.for
  // CHECK:   }
  // CHECK:   affine.parallel
  // CHECK:   }
  // CHECK: }

  return
}

// CHECK-LABEL: @iter_arg_memrefs
func.func @iter_arg_memrefs(%in: memref<10xf32>) {
  %mi = memref.alloc() : memref<f32>
  // Loop-carried memrefs are treated as serializing the loop.
  // CHECK: affine.for
  %mo = affine.for %i = 0 to 10 iter_args(%m_arg = %mi) -> (memref<f32>) {
    affine.yield %m_arg : memref<f32>
  }
  return
}

// CHECK-LABEL: @scf_affine_scope
func.func @scf_affine_scope() {
  %c0 = arith.constant 0 : index
  %0 = tensor.empty(%c0) : tensor<?xi1>
  %1 = bufferization.to_memref %0 : memref<?xi1>
  %alloc = memref.alloc(%c0) : memref<?xi1>
  %2 = scf.index_switch %c0 -> tensor<?x31x6xf16>
  default {
    %dim = memref.dim %1, %c0 : memref<?xi1>
    affine.for %arg0 = 0 to %dim {
      %3 = affine.load %1[%arg0] : memref<?xi1>
      affine.store %3, %alloc[%arg0] : memref<?xi1>
    }
    %3 = tensor.empty(%c0) : tensor<?x31x6xf16>
    scf.yield %3 : tensor<?x31x6xf16>
  } 
  return
}
