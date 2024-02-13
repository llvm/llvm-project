// RUN: mlir-opt %s -split-input-file -affine-loop-tile="tile-size=32" | FileCheck %s
// RUN: mlir-opt %s -split-input-file -affine-loop-tile="cache-size=512" | FileCheck %s --check-prefix=MODEL
// RUN: mlir-opt %s -split-input-file -affine-loop-tile="tile-size=32 separate" | FileCheck %s --check-prefix=SEPARATE

// -----

// CHECK-DAG: [[$UB:#map[0-9]*]] = affine_map<(d0) -> (d0 + 32)>
// CHECK-DAG: [[$UB_MIN:#map[0-9]*]] = affine_map<(d0) -> (d0 + 32, 50)>
// CHECK-DAG: [[$ID:#map[0-9]*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[$ID_PLUS_21:#map[0-9]*]] = affine_map<(d0) -> (d0 + 21)>

// CHECK-LABEL: func @loop_tiling()
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 256 step 32 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 512 step 32 {
// CHECK-NEXT:       affine.for %{{.*}} = 0 to 1024 step 32 {
// CHECK-NEXT:         affine.for %[[I:.*]] = [[$ID]](%{{.*}}) to [[$UB]](%{{.*}}) {
// CHECK-NEXT:           affine.for %[[J:.*]] = [[$ID]](%{{.*}}) to [[$UB]](%{{.*}}) {
// CHECK-NEXT:             affine.for %[[K:.*]] = [[$ID]](%{{.*}}) to [[$UB]](%{{.*}}) {
// CHECK-NEXT:               "test.foo"(%[[I]], %[[J]], %[[K]])
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 50 step 32 {
// CHECK-NEXT:     affine.for %[[X:.*]] = [[$ID]](%{{.*}}) to min [[$UB_MIN]](%{{.*}}) {
// CHECK-NEXT:       "test.bar"(%[[X]], %[[X]])
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: affine.for %[[I:.*]] = 0 to 21 step 32 {
// CHECK-NEXT:   affine.for %[[Y:.*]] = [[$ID]](%[[I]]) to [[$ID_PLUS_21]](%[[I]])  {
// CHECK-NEXT:     "test.foobar"(%[[Y]])
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT:  return
func.func @loop_tiling() {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 512 {
      affine.for %k = 0 to 1024 {
        "test.foo"(%i, %j, %k) : (index, index, index) -> ()
      }
    }
  }

  affine.for %x = 0 to 50 {
    "test.bar"(%x, %x) : (index, index) -> ()
  }

  // Intra-tile loop won't need a min expression.
  affine.for %y = 0 to 21 {
    "test.foobar"(%y) : (index) -> ()
  }

  return
}

// -----

// CHECK-DAG: [[$IDENTITY:#map[0-9]*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[$LB:#map[0-9]*]] = affine_map<()[s0] -> (0, s0)>
// CHECK-DAG: [[$UB:#map[0-9]*]] = affine_map<()[s0, s1] -> (s0, 4096 floordiv s1)>
// CHECK-DAG: [[$UB_INTRA_TILE:#map[0-9]*]] = affine_map<(d0)[s0, s1] -> (d0 + 32, s0, 4096 floordiv s1)>

#lb = affine_map<()[s0] -> (0, s0)>
#ub = affine_map<()[s0, s1] -> (s0, 4096 floordiv s1)>
// CHECK-LABEL: func @loop_max_min_bound(%{{.*}}: memref<?xi32>, %{{.*}}: index, %{{.*}}: index) {
func.func @loop_max_min_bound(%A : memref<? x i32>, %L : index, %U : index) {
  %c0 = arith.constant 0 : index
  %M = memref.dim %A, %c0 : memref<? x i32>
  affine.for %i = max #lb()[%L] to min #ub()[%M, %U] {
    arith.addi %i, %i : index
  }
  return
// CHECK:       affine.for %{{.*}} = max [[$LB]]()[%{{.*}}] to min [[$UB]]()[%{{.*}}, %{{.*}}] step 32 {
// CHECK-NEXT:    affine.for %[[I:.*]] = [[$IDENTITY]](%{{.*}}) to min [[$UB_INTRA_TILE]](%{{.*}})[%{{.*}}, %{{.*}}] {
// CHECK-NEXT:      arith.addi %[[I]], %[[I]]
// CHECK-NEXT:    }
// CHECK-NEXT:  }
}

// -----

// Cache size is set to 512 KiB. This loop nest accesses about 49 MiB, and the
// tile sizes chosen would be 6 x 6 x 6. However, to avoid min/max, which is
// possible here, they are adjusted to 4 x 4 x 5.

// MODEL-LABEL: func @simple_matmul
func.func @simple_matmul(%arg0: memref<256x256xvector<64xf32>>, %arg1: memref<256x256xvector<64xf32>>, %arg2: memref<256x256xvector<64xf32>>) -> memref<256x256xvector<64xf32>> {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      affine.for %k = 0 to 250 {
        %l = affine.load %arg0[%i, %k] : memref<256x256xvector<64xf32>>
        %r = affine.load %arg1[%k, %j] : memref<256x256xvector<64xf32>>
        %o = affine.load %arg2[%i, %j] : memref<256x256xvector<64xf32>>
        %m = arith.mulf %l, %r : vector<64xf32>
        %a = arith.addf %o, %m : vector<64xf32>
        affine.store %a, %arg2[%i, %j] : memref<256x256xvector<64xf32>>
      }
    }
  }
  return %arg2 : memref<256x256xvector<64xf32>>
}
// MODEL:       affine.for %{{.*}} = 0 to 256 step 4 {
// MODEL-NEXT:    affine.for %{{.*}} = 0 to 256 step 4 {
// MODEL-NEXT:      affine.for %{{.*}} = 0 to 250 step 5 {


// -----

// CHECK-DAG: [[$UBMAP:#map[0-9]*]] = affine_map<(d0)[s0] -> (d0 + 32, s0)>

func.func @tile_using_symbolic_loop_upper_bounds(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  affine.for %i0 = 0 to %0 {
    affine.for %i1 = 0 to %0 {
      affine.store %cst, %arg2[%i0, %i1] : memref<?x?xf32>
      affine.for %i2 = 0 to %0 {
        %1 = affine.load %arg0[%i0, %i2] : memref<?x?xf32>
        %2 = affine.load %arg1[%i2, %i1] : memref<?x?xf32>
        %3 = arith.mulf %1, %2 : f32
        %4 = affine.load %arg2[%i0, %i1] : memref<?x?xf32>
        %5 = arith.addf %4, %3 : f32
        affine.store %5, %arg2[%i0, %i1] : memref<?x?xf32>
      }
    }
  }
  return
}

// CHECK:       memref.dim %{{.*}}, %c0 : memref<?x?xf32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to %{{.*}} step 32 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to %{{.*}} step 32 {
// CHECK-NEXT:      affine.for %{{.*}} = #[[$MAP:.*]](%{{.*}}) to min [[$UBMAP]](%{{.*}})[%{{.*}}] {
// CHECK-NEXT:        affine.for %{{.*}} = #[[$MAP]](%{{.*}}) to min [[$UBMAP]](%{{.*}})[%{{.*}}] {
// CHECK-NEXT:          affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
// CHECK-NEXT:          affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK-NEXT:            affine.load
// CHECK-NEXT:            affine.load
// CHECK-NEXT:            arith.mulf
// CHECK-NEXT:            affine.load
// CHECK-NEXT:            arith.addf
// CHECK-NEXT:            affine.store
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return

// -----

// CHECK-DAG: [[MAP0:#map[0-9]*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[MAP1:#map[0-9]*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-DAG: [[$UBMAP:#map[0-9]*]] = affine_map<(d0)[s0, s1] -> (d0 + 32, s0 + s1)>

func.func @tile_using_loop_upper_bounds_in_two_symbols(%arg0: memref<?xf32>, %limit: index) {
  %c0 = arith.constant 0 : index
  %dim0 = memref.dim %arg0, %c0 : memref<?xf32>
  affine.for %i0 = 0 to affine_map<()[s0, s1] -> (s0 + s1)> ()[%dim0, %limit] {
    %v0 = affine.load %arg0[%i0] : memref<?xf32>
  }
  return
}

// CHECK:       memref.dim %{{.*}}, %c0 : memref<?xf32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to [[MAP1]]()[%{{.*}}, %{{.*}}] step 32 {
// CHECK-NEXT:    affine.for %{{.*}} = [[MAP0]](%{{.*}}) to min [[$UBMAP]](%{{.*}})[%{{.*}}, %{{.*}}] {
// CHECK-NEXT:      affine.load
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

// CHECK-DAG:  #[[$ID:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:  [[$UBMAP:#map[0-9]*]] = affine_map<(d0)[s0] -> (d0 + 160, s0)>

func.func @tile_loop_with_non_unit_step(%arg0 : memref<50xf32>, %arg1 : index) {
  affine.for %i = 0 to %arg1 step 5 {
    affine.load %arg0[%i] : memref<50xf32>
  }
  return
}

// CHECK-LABEL: func @tile_loop_with_non_unit_step(%arg{{.*}}: memref<50xf32>, %arg{{.*}}: index)
// CHECK:     affine.for %[[I:.*]] = 0 to %[[N:.*]] step 160 {
// CHECK-NEXT:       affine.for %[[II:.*]] = [[$ID:.*]](%[[I]]) to min
// [[$UBMAP]](%[[I]])[%[[N]]] step 5 {
// CHECK-NEXT:         affine.load %arg{{.*}}[%arg{{.*}}] : memref<50xf32>

// -----

func.func @tile_size_larger_than_trip_count_symbolic_bound(%M: index, %N :  index) {
  affine.for %i = affine_map<(d0) -> (d0)>(%M) to affine_map<(d0) -> (d0 + 2)>(%M) {
    affine.for %j = affine_map<(d0) -> (d0)>(%N) to affine_map<(d0) -> (d0 + 4)>(%N) {
      "test.foo" () : () -> ()
    }
  }
  return
}

// CHECK-DAG: #[[$ID:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$ID_PLUS_2:.*]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG: #[[$ID_PLUS_4:.*]] = affine_map<(d0) -> (d0 + 4)>
// CHECK: %[[M:.*]]: index, %[[N:.*]]: index
// CHECK:      affine.for %[[I:.*]] = #[[$ID]](%[[M]]) to #[[$ID_PLUS_2]](%[[M]]) step 32
// CHECK-NEXT:   affine.for %[[J:.*]] = #[[$ID]](%[[N]]) to #[[$ID_PLUS_4]](%[[N]]) step 32
// CHECK-NEXT:     affine.for %arg4 = #[[$ID]](%[[I]]) to #[[$ID_PLUS_2]](%[[I]])
// CHECK-NEXT:       affine.for %arg5 = #[[$ID]](%[[J]]) to #[[$ID_PLUS_4]](%[[J]])
// CHECK-NEXT:         "test.foo"

// -----

// CHECK-LABEL: func @trip_count_one
// SEPARATE-LABEL: func @trip_count_one
func.func @trip_count_one(%arg0: memref<196608x1xf32>, %arg1: memref<196608x1xf32>)
    -> memref<196608x1xf32> {
  affine.for %i1 = 0 to 196608 {
    affine.for %i3 = 0 to 1 {
      %4 = affine.load %arg0[%i1, %i3] : memref<196608x1xf32>
      affine.store %4, %arg1[%i1, %i3] : memref<196608x1xf32>
    }
  }
  // CHECK: affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<196608x1xf32>
  return %arg1 : memref<196608x1xf32>
}
// To make sure SEPARATE-DAGs further below do not match with something above.
// SEPARATE: return

// -----

func.func @separate_full_tile_2d(%M : index, %N : index) {
  affine.for %i = 0 to %M {
    affine.for %j = 0 to %N {
      "test.foo"() : () -> ()
    }
  }
  return
}

// -----

#ub = affine_map<(d0)[s0] -> (d0, s0)>
// CHECK-LABEL: func @non_hyperrectangular_loop
func.func @non_hyperrectangular_loop() {
  %N = arith.constant 128 : index
  affine.for %i = 0 to %N {
    affine.for %j = 0 to min #ub(%i)[%N] {
      "test.foo"() : () -> ()
    }
 }
  // No tiling is performed here.
  // CHECK:      arith.constant
  // CHECK-NEXT: affine.for
  // CHECK-NEXT:   affine.for
  // CHECK-NEXT:     test.foo
  return
}

// -----

// No tiling supported on loops with yield values.

// CHECK-LABEL: func @yield_values
func.func @yield_values(%init : index) {
  %r = affine.for %i = 0 to 10 iter_args(%s = %init) -> index {
    "test.foo"() : () -> ()
    affine.yield %s : index
  }
  // No tiling here.
  // CHECK-NEXT: affine.for {{.*}} {
  // CHECK-NEXT:   test.foo
  return
}

// -----

// SEPARATE-DAG: #[[$SEP_COND:.*]] = affine_set<(d0, d1)[s0, s1] : (-d0 + s0 - 32 >= 0, -d1 + s1 - 32 >= 0)>
// SEPARATE-DAG: #[[$LB:.*]] = affine_map<(d0) -> (d0)>
// SEPARATE-DAG: #[[$FULL_TILE_UB:.*]] = affine_map<(d0) -> (d0 + 32)>
// SEPARATE-DAG: #[[$PART_TILE_UB:.*]] = affine_map<(d0)[s0] -> (d0 + 32, s0)>

// SEPARATE-LABEL: func @separate_full_tile_2d(
// SEPARATE: %[[M:.*]]: index, %[[N:.*]]: index

// SEPARATE:       affine.for %[[I:.*]] =
// SEPARATE-NEXT:    affine.for %[[J:.*]] =
// SEPARATE-NEXT:      affine.if #[[$SEP_COND]](%arg2, %arg3)[%arg0, %arg1] {
// SEPARATE-NEXT:        affine.for %{{.*}} = #[[$LB]](%[[I]]) to #[[$FULL_TILE_UB]](%[[I]]) {
// SEPARATE-NEXT:          affine.for %{{.*}} = #[[$LB]](%[[J]]) to #[[$FULL_TILE_UB]](%[[J]]) {
// SEPARATE-NEXT:           "test.foo"
// SEPARATE-NEXT:          }
// SEPARATE-NEXT:        }
// SEPARATE-NEXT:      } else {
// SEPARATE-NEXT:        affine.for %{{.*}} = #[[$LB]](%[[I]]) to min #[[$PART_TILE_UB]](%[[I]])[%[[M]]] {
// SEPARATE-NEXT:          affine.for %{{.*}} = #[[$LB]](%[[J]]) to min #[[$PART_TILE_UB]](%[[J]])[%[[N]]] {
// SEPARATE-NEXT:           "test.foo"
// SEPARATE-NEXT:          }
// SEPARATE-NEXT:        }
// SEPARATE-NEXT:      }
// SEPARATE-NEXT:    }
// SEPARATE-NEXT:  }
// SEPARATE-NEXT:  return

// -----

func.func @separate_full_tile_1d_max_min(%M : index, %N : index, %P : index, %Q : index) {
  affine.for %i0 = max affine_map<(d0, d1) -> (d0, d1)>  (%M, %N) to min affine_map< (d0, d1) -> (d0, d1)> (%P, %Q) {
  }
  return
}

// SEPARATE-DAG: #[[$SEP_COND:.*]] = affine_set<(d0)[s0, s1] : (-d0 + s0 - 32 >= 0, -d0 + s1 - 32 >= 0)>
// SEPARATE-DAG: #[[TILE_LB:.*]] = affine_map<(d0) -> (d0)>
// SEPARATE-DAG: #[[$FULL_TILE_UB:.*]] = affine_map<(d0) -> (d0 + 32)>
// SEPARATE-DAG: #[[PARTIAL_TILE_UB:.*]] = affine_map<(d0, d1, d2) -> (d2 + 32, d0, d1)>

// SEPARATE:         affine.for %arg4
// SEPARATE-NEXT:      affine.if #[[$SEP_COND]](%arg4)[%arg2, %arg3] {
// SEPARATE-NEXT:        affine.for %arg5 = #[[TILE_LB]](%arg4) to #[[$FULL_TILE_UB]](%arg4) {
// SEPARATE-NEXT:        }
// SEPARATE-NEXT:      } else {
// SEPARATE-NEXT:        affine.for %arg5 = #[[TILE_LB]](%arg4) to min #[[PARTIAL_TILE_UB]](%arg2, %arg3, %arg4) {
// SEPARATE-NEXT:        }
// SEPARATE-NEXT:      }
// SEPARATE-NEXT:    }
