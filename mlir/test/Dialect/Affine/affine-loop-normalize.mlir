// RUN: mlir-opt %s -affine-loop-normalize -split-input-file | FileCheck %s

// Normalize steps to 1 and lower bounds to 0.

// CHECK-DAG: [[$MAP0:#map[0-9]+]] = affine_map<(d0) -> (d0 * 3)>
// CHECK-DAG: [[$MAP1:#map[0-9]+]] = affine_map<(d0) -> (d0 * 2 + 1)>
// CHECK-DAG: [[$MAP2:#map[0-9]+]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: func @normalize_parallel()
func.func @normalize_parallel() {
  %cst = arith.constant 1.0 : f32
  %0 = memref.alloc() : memref<2x4xf32>
  // CHECK: affine.parallel (%[[i0:.*]], %[[j0:.*]]) = (0, 0) to (4, 2)
  affine.parallel (%i, %j) = (0, 1) to (10, 5) step (3, 2) {
    // CHECK: %[[i1:.*]] = affine.apply [[$MAP0]](%[[i0]])
    // CHECK: %[[j1:.*]] = affine.apply [[$MAP1]](%[[j0]])
    // CHECK: affine.parallel (%[[k0:.*]]) = (0) to (%[[j1]] - %[[i1]])
    affine.parallel (%k) = (%i) to (%j) {
      // CHECK: %[[k1:.*]] = affine.apply [[$MAP2]](%[[i1]], %[[k0]])
      // CHECK: affine.store %{{.*}}, %{{.*}}[%[[i1]], %[[k1]]] : memref<2x4xf32>
      affine.store %cst, %0[%i, %k] : memref<2x4xf32>
    }
  }
  return
}

// -----

// CHECK-LABEL: func @relative_bounds
func.func @relative_bounds(%arg: index) {
  // CHECK: affine.for %{{.*}} = 0 to 4
  affine.for %i = affine_map<(d0) -> (d0)>(%arg) to affine_map<(d0) -> (d0 + 4)>(%arg) {
  }
  return
}

// -----

// Check that single iteration loop is removed and its body is promoted to the
// parent block.

// CHECK-LABEL: func @single_iteration_loop
func.func @single_iteration_loop(%in: memref<1xf32>, %out: memref<1xf32>) {
  affine.for %i = 0 to 1 {
    %1 = affine.load %in[%i] : memref<1xf32>
    affine.store %1, %out[%i] : memref<1xf32>
  }
  return
}

// CHECK-NOT:  affine.for
// CHECK:      affine.load
// CHECK-NEXT: affine.store
// CHECK-NEXT: return

// -----

// CHECK-DAG: [[$IV0:#map[0-9]+]] = affine_map<(d0) -> (d0 * 2 + 2)>
// CHECK-DAG: [[$IV1:#map[0-9]+]] = affine_map<(d0) -> (d0 * 3)>

// CHECK-LABEL: func @simple_loop_nest()
// CHECK-NEXT:   affine.for %[[I:.*]] = 0 to 15 {
// CHECK-NEXT:     %[[IIV:.*]] = affine.apply [[$IV0]](%[[I]])
// CHECK-NEXT:     affine.for %[[II:.*]] = 0 to 11 {
// CHECK-NEXT:       %[[IIIV:.*]] = affine.apply [[$IV1]](%[[II]])
// CHECK-NEXT:       "test.foo"(%[[IIV]], %[[IIIV]])
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @simple_loop_nest(){
  affine.for %i0 = 2 to 32 step 2 {
    affine.for %i1 =  0 to 32 step 3 {
      "test.foo"(%i0, %i1) : (index, index) -> ()
    }
  }
  return
}

// -----

// CHECK-DAG: [[$IV00:#map[0-9]+]] = affine_map<(d0) -> (d0 * 32 + 2)>
// CHECK-DAG: [[$IV11:#map[0-9]+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-DAG: [[$UB00:#map[0-9]+]] = affine_map<()[s0] -> ((s0 - 2) ceildiv 32)>
// CHECK-DAG: [[$UB11:#map[0-9]+]] = affine_map<()[s0] -> (s0 ceildiv 2)>

// CHECK-LABEL: func @loop_with_unknown_upper_bound
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: index)
// CHECK-NEXT:  %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:  %[[DIM:.*]] = memref.dim %arg0, %c0 : memref<?x?xf32>
// CHECK-NEXT:   affine.for %[[I:.*]] = 0 to [[$UB00]]()[%[[DIM]]] {
// CHECK-NEXT:     %[[IIV:.*]] = affine.apply [[$IV00]](%[[I]])
// CHECK-NEXT:     affine.for %[[II:.*]] = 0 to [[$UB11]]()[%[[ARG1]]] {
// CHECK-NEXT:       %[[IIIV:.*]] = affine.apply [[$IV11]](%[[II]])
// CHECK-NEXT:       "test.foo"(%[[IIV]], %[[IIIV]])
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @loop_with_unknown_upper_bound(%arg0: memref<?x?xf32>, %arg1: index) {
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  affine.for %i0 = 2 to %0 step 32 {
    affine.for %i1 = 0 to %arg1 step 2 {
      "test.foo"(%i0, %i1) : (index, index) -> ()
    }
  }
  return
}

// -----

// CHECK-DAG: [[$OUTERIV:#map[0-9]+]] = affine_map<(d0) -> (d0 * 32 + 2)>
// CHECK-DAG: [[$INNERIV:#map[0-9]+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG: [[$OUTERUB:#map[0-9]+]] = affine_map<()[s0] -> ((s0 - 2) ceildiv 32)>
// CHECK-DAG: [[$INNERUB:#map[0-9]+]] = affine_map<()[s0] -> (s0 - 2, 510)>

// CHECK-LABEL: func @loop_with_multiple_upper_bounds
// CHECK-SAME: (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: index)
// CHECK-NEXT:  %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:  %[[DIM:.*]] = memref.dim %arg0, %c0 : memref<?x?xf32>
// CHECK-NEXT:   affine.for %[[I:.*]] = 0 to [[$OUTERUB]]()[%[[DIM]]] {
// CHECK-NEXT:     %[[IIV:.*]] = affine.apply [[$OUTERIV]](%[[I]])
// CHECK-NEXT:     affine.for %[[II:.*]] = 0 to min [[$INNERUB]]()[%[[ARG1]]] {
// CHECK-NEXT:       %[[IIIV:.*]] = affine.apply [[$INNERIV]](%[[II]])
// CHECK-NEXT:       "test.foo"(%[[IIV]], %[[IIIV]])
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
func.func @loop_with_multiple_upper_bounds(%arg0: memref<?x?xf32>, %arg1 : index) {
  %c0 = arith.constant 0 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  affine.for %i0 = 2 to %0 step 32{
    affine.for %i1 = 2 to min affine_map<(d0)[] -> (d0, 512)>(%arg1) {
      "test.foo"(%i0, %i1) : (index, index) -> ()
    }
  }
  return
}

// -----

// CHECK-DAG: [[$INTERUB:#map[0-9]+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
// CHECK-DAG: [[$INTERIV:#map[0-9]+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG: [[$INTRAUB:#map[0-9]+]] = affine_map<(d0)[s0] -> (32, -d0 + s0)>
// CHECK-DAG: [[$INTRAIV:#map[0-9]+]] = affine_map<(d0, d1) -> (d1 + d0)>

// CHECK-LABEL: func @tiled_matmul
// CHECK-SAME: (%[[ARG0:.*]]: memref<1024x1024xf32>, %[[ARG1:.*]]: memref<1024x1024xf32>, %[[ARG2:.*]]: memref<1024x1024xf32>)
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:    %[[DIM0:.*]] = memref.dim %[[ARG0]], %{{.*}}
// CHECK-NEXT:    %[[DIM1:.*]] = memref.dim %[[ARG1]], %{{.*}}
// CHECK-NEXT:    %[[DIM2:.*]] = memref.dim %[[ARG0]], %{{.*}}
// CHECK-NEXT:    affine.for %[[I:.*]] = 0 to [[$INTERUB]]()[%[[DIM0]]] {
// CHECK-NEXT:      %[[IIV:.*]] = affine.apply [[$INTERIV]](%[[I]])
// CHECK-NEXT:      affine.for %[[J:.*]] = 0 to [[$INTERUB]]()[%[[DIM1]]] {
// CHECK-NEXT:        %[[JIV:.*]] = affine.apply [[$INTERIV]](%[[J]])
// CHECK-NEXT:        affine.for %[[K:.*]] = 0 to [[$INTERUB]]()[%[[DIM2]]] {
// CHECK-NEXT:          %[[KIV:.*]] = affine.apply [[$INTERIV]](%[[K]])
// CHECK-NEXT:          affine.for %[[II:.*]] = 0 to min [[$INTRAUB]](%[[IIV]])[%[[DIM0]]] {
// CHECK-NEXT:            %[[IIIV:.*]] = affine.apply [[$INTRAIV]](%[[IIV]], %[[II]])
// CHECK-NEXT:            affine.for %[[JJ:.*]] = 0 to min [[$INTRAUB]](%[[JIV]])[%[[DIM1]]] {
// CHECK-NEXT:              %[[JJIV:.*]] = affine.apply [[$INTRAIV]](%[[JIV]], %[[JJ]])
// CHECK-NEXT:              affine.for %[[KK:.*]] = 0 to min [[$INTRAUB]](%[[KIV]])[%[[DIM2]]] {
// CHECK-NEXT:                %[[KKIV:.*]] = affine.apply [[$INTRAIV]](%[[KIV]], %[[KK]])
// CHECK-NEXT:                %{{.*}} = affine.load %[[ARG0]][%[[IIIV]], %[[KKIV]]] : memref<1024x1024xf32>
// CHECK-NEXT:                %{{.*}} = affine.load %[[ARG1]][%[[KKIV]], %[[JJIV]]] : memref<1024x1024xf32>
// CHECK-NEXT:                %{{.*}} = affine.load %[[ARG2]][%[[IIIV]], %[[JJIV]]] : memref<1024x1024xf32>
// CHECK-NEXT:                %{{.*}} = arith.mulf 
// CHECK-NEXT:                %{{.*}} = arith.addf 
// CHECK-NEXT:                affine.store %{{.*}}, %[[ARG2]]{{.*}} : memref<1024x1024xf32>
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0)[s0] -> (d0 + 32, s0)>
#map3 = affine_map<() -> (0)>
#map4 = affine_map<()[s0] -> (s0)>

func.func @tiled_matmul(%0: memref<1024x1024xf32>, %1: memref<1024x1024xf32>, %2: memref<1024x1024xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %3 = memref.dim %0, %c0 : memref<1024x1024xf32>
  %4 = memref.dim %1, %c1 : memref<1024x1024xf32>
  %5 = memref.dim %0, %c1 : memref<1024x1024xf32>
  affine.for %arg0 = 0 to %3 step 32 {
    affine.for %arg1 = 0 to %4 step 32 {
      affine.for %arg2 = 0 to %5 step 32 {
        affine.for %arg3 = #map1(%arg0) to min #map2(%arg0)[%3] {
          affine.for %arg4 = #map1(%arg1) to min #map2(%arg1)[%4] {
            affine.for %arg5 = #map1(%arg2) to min #map2(%arg2)[%5] {
              %6 = affine.load %0[%arg3, %arg5] : memref<1024x1024xf32>
              %7 = affine.load %1[%arg5, %arg4] : memref<1024x1024xf32>
              %8 = affine.load %2[%arg3, %arg4] : memref<1024x1024xf32>
              %9 = arith.mulf %6, %7 : f32
              %10 = arith.addf %8, %9 : f32
              affine.store %10, %2[%arg3, %arg4] : memref<1024x1024xf32>
            }
          }
        }
      }
    }
  }
  return
}
