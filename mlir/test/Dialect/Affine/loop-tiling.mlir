// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -affine-loop-tile="tile-size=32" | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -affine-loop-tile="cache-size=512" | FileCheck %s --check-prefix=MODEL
// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -affine-loop-tile="tile-size=32 separate" | FileCheck %s --check-prefix=SEPARATE

// -----

// CHECK-DAG: [[MAP0:#map[0-9]+]] = affine_map<(d0) -> (d0 + 32)>
// CHECK-DAG: [[MAP1:#map[0-9]+]] = affine_map<(d0) -> (d0 + 32, 50)>
// CHECK-DAG: [[ID:#map[0-9]+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[ID_PLUS_21:#map[0-9]+]] = affine_map<(d0) -> (d0 + 21)>

// CHECK-LABEL: func @loop_tiling()
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 256 step 32 {
// CHECK-NEXT:     affine.for %{{.*}} = 0 to 512 step 32 {
// CHECK-NEXT:       affine.for %{{.*}} = 0 to 1024 step 32 {
// CHECK-NEXT:         affine.for %{{.*}} = [[ID]](%{{.*}}) to [[MAP0]](%{{.*}}) {
// CHECK-NEXT:           affine.for %{{.*}} = [[ID]](%{{.*}}) to [[MAP0]](%{{.*}}) {
// CHECK-NEXT:             affine.for %{{.*}} = [[ID]](%{{.*}}) to [[MAP0]](%{{.*}}) {
// CHECK-NEXT:               "foo"(%{{.*}}, %{{.*}}, %{{.*}}) : (index, index, index) -> ()
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 50 step 32 {
// CHECK-NEXT:     affine.for %{{.*}} = [[ID]](%{{.*}}) to min [[MAP1]](%{{.*}}) {
// CHECK-NEXT:       "bar"(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: affine.for %[[I:.*]] = 0 to 21 step 32 {
// CHECK-NEXT:    affine.for %{{.*}} = [[ID]](%[[I]]) to [[ID_PLUS_21]](%[[I]]) {
// CHECK-NEXT:      "foobar"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
func @loop_tiling() {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 512 {
      affine.for %k = 0 to 1024 {
        "foo"(%i, %j, %k) : (index, index, index) -> ()
      }
    }
  }

  affine.for %x = 0 to 50 {
    "bar"(%x, %x) : (index, index) -> ()
  }

  // Intra-tile loop won't need a min expression.
  affine.for %y = 0 to 21 {
    "foobar"(%y) : (index) -> ()
  }

  return
}

// -----

// CHECK-DAG: [[IDENTITY:#map[0-9]+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[LB:#map[0-9]+]] = affine_map<()[s0] -> (0, s0)>
// CHECK-DAG: [[UB:#map[0-9]+]] = affine_map<()[s0, s1] -> (s0, 4096 floordiv s1)>
// CHECK-DAG: [[UB_INTRA_TILE:#map[0-9]+]] = affine_map<(d0)[s0, s1] -> (d0 + 32, s0, 4096 floordiv s1)>

#lb = affine_map<()[s0] -> (0, s0)>
#ub = affine_map<()[s0, s1] -> (s0, 4096 floordiv s1)>
// CHECK-LABEL: func @loop_max_min_bound(%{{.*}}: memref<?xi32>, %{{.*}}: index, %{{.*}}: index) {
func @loop_max_min_bound(%A : memref<? x i32>, %L : index, %U : index) {
  %M = dim %A, 0 : memref<? x i32>
  affine.for %iTT = max #lb()[%L] to min #ub()[%M, %U] {
      %out = affine.apply affine_map<(d0) -> (d0)> (%iTT)
  }
  return
// CHECK:       affine.for %{{.*}} = max [[LB]]()[%{{.*}}] to min [[UB]]()[%{{.*}}, %{{.*}}] step 32 {
// CHECK-NEXT:    affine.for %{{.*}} = [[IDENTITY]](%{{.*}}) to min [[UB_INTRA_TILE]](%{{.*}})[%{{.*}}, %{{.*}}] {
// CHECK-NEXT:      %{{.*}} = affine.apply [[IDENTITY]](%{{.*}})
// CHECK-NEXT:    }
// CHECK-NEXT:  }
}

// -----

// Cache size is set to 512 KiB. This loop nest accesses about 49 MiB, and the
// tile sizes chosen would be 6 x 6 x 6. However, to avoid min/max, which is
// possible here, they are adjusted to 4 x 4 x 5.

// MODEL-LABEL: func @simple_matmul
func @simple_matmul(%arg0: memref<256x256xvector<64xf32>>, %arg1: memref<256x256xvector<64xf32>>, %arg2: memref<256x256xvector<64xf32>>) -> memref<256x256xvector<64xf32>> {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      affine.for %k = 0 to 250 {
        %l = affine.load %arg0[%i, %k] : memref<256x256xvector<64xf32>>
        %r = affine.load %arg1[%k, %j] : memref<256x256xvector<64xf32>>
        %o = affine.load %arg2[%i, %j] : memref<256x256xvector<64xf32>>
        %m = mulf %l, %r : vector<64xf32>
        %a = addf %o, %m : vector<64xf32>
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

// CHECK-DAG: [[UBMAP:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 + 32, s0)>

func @tile_with_symbolic_loop_upper_bounds(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  %cst = constant 0.000000e+00 : f32
  %0 = dim %arg0, 0 : memref<?x?xf32>
  affine.for %i0 = 0 to %0 {
    affine.for %i1 = 0 to %0 {
      affine.store %cst, %arg2[%i0, %i1] : memref<?x?xf32>
      affine.for %i2 = 0 to %0 {
        %1 = affine.load %arg0[%i0, %i2] : memref<?x?xf32>
        %2 = affine.load %arg1[%i2, %i1] : memref<?x?xf32>
        %3 = mulf %1, %2 : f32
        %4 = affine.load %arg2[%i0, %i1] : memref<?x?xf32>
        %5 = addf %4, %3 : f32
        affine.store %5, %arg2[%i0, %i1] : memref<?x?xf32>
      }
    }
  }
  return
}

// CHECK:       %{{.*}} = dim %{{.*}}, 0 : memref<?x?xf32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to %{{.*}} step 32 {
// CHECK-NEXT:    affine.for %{{.*}} = 0 to %{{.*}} step 32 {
// CHECK-NEXT:      affine.for %{{.*}} = #map3(%{{.*}}) to min [[UBMAP]](%{{.*}})[%{{.*}}] {
// CHECK-NEXT:        affine.for %{{.*}} = #map3(%{{.*}}) to min [[UBMAP]](%{{.*}})[%{{.*}}] {
// CHECK-NEXT:          affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
// CHECK-NEXT:          affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK-NEXT:            %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
// CHECK-NEXT:            %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
// CHECK-NEXT:            %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:            %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
// CHECK-NEXT:            %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:            affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return

// -----

// CHECK-DAG: [[MAP0:#map[0-9]+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[MAP1:#map[0-9]+]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-DAG: [[UBMAP:#map[0-9]+]] = affine_map<(d0)[s0, s1] -> (d0 + 32, s0 + s1)>

func @tile_with_loop_upper_bounds_in_two_symbols(%arg0: memref<?xf32>, %limit: index) {
  %dim0 = dim %arg0, 0 : memref<?xf32>
  affine.for %i0 = 0 to affine_map<()[s0, s1] -> (s0 + s1)> ()[%dim0, %limit] {
    %v0 = affine.load %arg0[%i0] : memref<?xf32>
  }
  return
}

// CHECK:       %{{.*}} = dim %{{.*}}, 0 : memref<?xf32>
// CHECK-NEXT:  affine.for %{{.*}} = 0 to [[MAP1]]()[%{{.*}}, %{{.*}}] step 32 {
// CHECK-NEXT:    affine.for %{{.*}} = [[MAP0]](%{{.*}}) to min [[UBMAP]](%{{.*}})[%{{.*}}, %{{.*}}] {
// CHECK-NEXT:      %{{.*}} = affine.load %{{.*}}[%{{.*}}] : memref<?xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

func @tile_size_larger_than_trip_count_symbolic_bound(%M: index, %N :  index) {
  affine.for %i = affine_map<(d0) -> (d0)>(%M) to affine_map<(d0) -> (d0 + 2)>(%M) {
    affine.for %j = affine_map<(d0) -> (d0)>(%N) to affine_map<(d0) -> (d0 + 4)>(%N) {
      "test.foo" () : () -> ()
    }
  }
  return
}

// CHECK-DAG: #[[ID:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[ID_PLUS_2:.*]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG: #[[ID_PLUS_4:.*]] = affine_map<(d0) -> (d0 + 4)>
// CHECK: %[[M:.*]]: index, %[[N:.*]]: index
// CHECK:      affine.for %[[I:.*]] = #[[ID]](%[[M]]) to #[[ID_PLUS_2]](%[[M]]) step 32
// CHECK-NEXT:   affine.for %[[J:.*]] = #[[ID]](%[[N]]) to #[[ID_PLUS_4]](%[[N]]) step 32
// CHECK-NEXT:     affine.for %arg4 = #[[ID]](%[[I]]) to #[[ID_PLUS_2]](%[[I]])
// CHECK-NEXT:       affine.for %arg5 = #[[ID]](%[[J]]) to #[[ID_PLUS_4]](%[[J]])
// CHECK-NEXT:         "test.foo"

// -----

// CHECK-LABEL: func @trip_count_1
// SEPARATE-LABEL: func @trip_count_1
func @trip_count_1(%arg0: memref<196608x1xf32>, %arg1: memref<196608x1xf32>)
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
// SEPARATE: return

// -----

func @separate_full_tile_2d(%M : index, %N : index) {
  affine.for %i0 = 0 to %M {
    affine.for %i1 = 0 to %N {
      "foo"() : () -> ()
    }
  }
  return
}

// SEPARATE-DAG: #[[SEP_COND:.*]] = affine_set<(d0, d1)[s0, s1] : (-d0 + s0 - 32 >= 0, -d1 + s1 - 32 >= 0)>
// SEPARATE-DAG: #[[LB:.*]] = affine_map<(d0) -> (d0)>
// SEPARATE-DAG: #[[FULL_TILE_UB:.*]] = affine_map<(d0) -> (d0 + 32)>
// SEPARATE-DAG: #[[PART_TILE_UB:.*]] = affine_map<(d0)[s0] -> (d0 + 32, s0)>

// SEPARATE:       affine.for %arg2
// SEPARATE-NEXT:    affine.for %arg3
// SEPARATE-NEXT:      affine.if #[[SEP_COND]](%arg2, %arg3)[%arg0, %arg1] {
// SEPARATE-NEXT:        affine.for %arg4 = #[[LB]](%arg2) to #[[FULL_TILE_UB]](%arg2) {
// SEPARATE-NEXT:          affine.for %arg5 = #[[LB]](%arg3) to #[[FULL_TILE_UB]](%arg3) {
// SEPARATE-NEXT:           "foo"
// SEPARATE-NEXT:          }
// SEPARATE-NEXT:        }
// SEPARATE-NEXT:      } else {
// SEPARATE-NEXT:        affine.for %arg4 = #[[LB]](%arg2) to min #[[PART_TILE_UB]](%arg2)[%arg0] {
// SEPARATE-NEXT:          affine.for %arg5 = #[[LB]](%arg3) to min #[[PART_TILE_UB]](%arg3)[%arg1] {
// SEPARATE-NEXT:           "foo"
// SEPARATE-NEXT:          }
// SEPARATE-NEXT:        }
// SEPARATE-NEXT:      }
// SEPARATE-NEXT:    }
// SEPARATE-NEXT:  }
// SEPARATE-NEXT:  return

// -----

func @separate_full_tile_1d_max_min(%M : index, %N : index, %P : index, %Q : index) {
  affine.for %i0 = max affine_map<(d0, d1) -> (d0, d1)>  (%M, %N) to min affine_map< (d0, d1) -> (d0, d1)> (%P, %Q) {
  }
  return
}

// SEPARATE-DAG: #[[SEP_COND:.*]] = affine_set<(d0)[s0, s1] : (-d0 + s0 - 32 >= 0, -d0 + s1 - 32 >= 0)>
// SEPARATE-DAG: #[[TILE_LB:.*]] = affine_map<(d0) -> (d0)>
// SEPARATE-DAG: #[[FULL_TILE_UB:.*]] = affine_map<(d0) -> (d0 + 32)>
// SEPARATE-DAG: #[[PARTIAL_TILE_UB:.*]] = affine_map<(d0, d1, d2) -> (d2 + 32, d0, d1)>

// SEPARATE:         affine.for %arg4
// SEPARATE-NEXT:      affine.if #[[SEP_COND]](%arg4)[%arg2, %arg3] {
// SEPARATE-NEXT:        affine.for %arg5 = #[[TILE_LB]](%arg4) to #[[FULL_TILE_UB]](%arg4) {
// SEPARATE-NEXT:        }
// SEPARATE-NEXT:      } else {
// SEPARATE-NEXT:        affine.for %arg5 = #[[TILE_LB]](%arg4) to min #[[PARTIAL_TILE_UB]](%arg2, %arg3, %arg4) {
// SEPARATE-NEXT:        }
// SEPARATE-NEXT:      }
// SEPARATE-NEXT:    }
