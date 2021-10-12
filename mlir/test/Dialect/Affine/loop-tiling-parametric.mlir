// RUN: mlir-opt %s -split-input-file -test-affine-parametric-tile | FileCheck %s
// Test cases to test the utility introduced to tile affine for loops using
// SSA values as tiling parameters(tile sizes). The tile sizes are expected
// to be passed as input arguments(before any other argument) to the function
// enclosing the loop nest. Currently hyper-rectangular loop nests with constant
// lower bounds are supported.

// -----

// CHECK-DAG: [[LBI:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 256)>
// CHECK-DAG: [[UBI1:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 512)>
// CHECK-DAG: [[UBI2:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 1024)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<()[s0] -> (256 ceildiv s0)>
// CHECK-DAG: [[UBO1:#map[0-9]+]] = affine_map<()[s0] -> (512 ceildiv s0)>
// CHECK-DAG: [[UBO2:#map[0-9]+]] = affine_map<()[s0] -> (1024 ceildiv s0)>

// CHECK: func @loop_tiling_3d([[ARG0:%arg[0-9]+]]: index, [[ARG1:%arg[0-9]+]]: index, [[ARG2:%arg[0-9]+]]: index)
// CHECK-NEXT:   affine.for [[ARG3:%arg[0-9]+]] = 0 to [[UBO0]](){{.*}}[[ARG0]]
// CHECK-NEXT:     affine.for [[ARG4:%arg[0-9]+]] = 0 to [[UBO1]](){{.*}}[[ARG1]]
// CHECK-NEXT:       affine.for [[ARG5:%arg[0-9]+]] = 0 to [[UBO2]](){{.*}}[[ARG2]]
// CHECK-NEXT:         affine.for %[[I:.*]] = [[LBI]]{{.*}}[[ARG3]]{{.*}}[[ARG0]]{{.*}} to min [[UBI0]]{{.*}}[[ARG3]]{{.*}}[[ARG0]]
// CHECK-NEXT:          affine.for %[[J:.*]] = [[LBI]]{{.*}}[[ARG4]]{{.*}}[[ARG1]]{{.*}} to min [[UBI1]]{{.*}}[[ARG4]]{{.*}}[[ARG1]]
// CHECK-NEXT:            affine.for %[[K:.*]] = [[LBI]]{{.*}}[[ARG5]]{{.*}}[[ARG2]]{{.*}} to min [[UBI2]]{{.*}}[[ARG5]]{{.*}}[[ARG2]]
// CHECK-NEXT:              "test.foo"(%[[I]], %[[J]], %[[K]])
func @loop_tiling_3d(%t0 : index, %t1 : index, %t2 : index) {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 512 {
      affine.for %k = 0 to 1024 {
        "test.foo"(%i, %j, %k) : (index, index, index) -> ()
      }
    }
  }
  return
}

// -----

// CHECK-DAG: [[LBI:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0 + s0 * 4, 256)>
// CHECK-DAG: [[UBI1:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0 + s0 * 3, 512)>
// CHECK-DAG: [[UBI2:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0 + s0 * 2, 1024)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<()[s0] -> (256 ceildiv s0)>
// CHECK-DAG: [[UBO1:#map[0-9]+]] = affine_map<()[s0] -> (512 ceildiv s0)>
// CHECK-DAG: [[UBO2:#map[0-9]+]] = affine_map<()[s0] -> (1024 ceildiv s0)>

// CHECK: func @loop_tiling_non_unit_step([[ARG0:%arg[0-9]+]]: index, [[ARG1:%arg[0-9]+]]: index, [[ARG2:%arg[0-9]+]]: index)
// CHECK-NEXT:  affine.for [[ARG3:%arg[0-9]+]] = 0 to [[UBO0]](){{.*}}[[ARG0]]{{.*}}step 4
// CHECK-NEXT:    affine.for [[ARG4:%arg[0-9]+]] = 0 to [[UBO1]](){{.*}}[[ARG1]]{{.*}} step 3
// CHECK-NEXT:       affine.for [[ARG5:%arg[0-9]+]] = 0 to [[UBO2]](){{.*}}[[ARG2]]{{.*}} step 2
// CHECK-NEXT:         affine.for %[[I:.*]] = [[LBI]]{{.*}}[[ARG3]]{{.*}}[[ARG0]]{{.*}} to min [[UBI0]]{{.*}}[[ARG3]]{{.*}}[[ARG0]]{{.*}} step 4
// CHECK-NEXT:          affine.for %[[J:.*]] = [[LBI]]{{.*}}[[ARG4]]{{.*}}[[ARG1]]{{.*}} to min [[UBI1]]{{.*}}[[ARG4]]{{.*}}[[ARG1]]{{.*}} step 3
// CHECK-NEXT:            affine.for %[[K:.*]] = [[LBI]]{{.*}}[[ARG5]]{{.*}}[[ARG2]]{{.*}} to min [[UBI2]]{{.*}}[[ARG5]]{{.*}}[[ARG2]]{{.*}} step 2
// CHECK-NEXT:              "test.foo"(%[[I]], %[[J]], %[[K]])
func @loop_tiling_non_unit_step(%t0: index, %t1: index, %t2: index){
  affine.for %i = 0 to 256 step 4 {
    affine.for %j = 0 to 512  step 3 {
      affine.for %k = 0 to 1024 step 2 {
        "test.foo"(%i, %j, %k) : (index, index, index) -> ()
      }
    }
  }
  return
}

// -----

// CHECK-DAG: [[LBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0)[s0, s1, s2] -> (d0 * s2 + s2, s0, 4096 floordiv s1)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<()[s0, s1, s2] -> (s0 ceildiv s2, (4096 floordiv s1) ceildiv s2)>

// CHECK: func @tile_loop_with_div_in_upper_bound([[ARG0:%arg[0-9]+]]: index, %{{.*}}: memref<?xi32>, %{{.*}}: index, %{{.*}}: index)
#ub = affine_map<()[s0, s1] -> (s0, 4096 floordiv s1)>
func @tile_loop_with_div_in_upper_bound(%t5 : index, %A : memref<? x i32>, %L : index, %U : index) {
  %c0 = arith.constant 0 : index
  %M = memref.dim %A, %c0 : memref<? x i32>
  affine.for %i = 0 to min #ub()[%M, %U] {
    arith.addi %i, %i : index
  }
  // CHECK:  affine.for [[ARG1:%arg[0-9]+]] = 0 to min [[UBO0]]()[%{{.*}}, %{{.*}}, [[ARG0]]]
  // CHECK-NEXT:    affine.for %[[I:.*]] = [[LBI0]]([[ARG1]]){{.*}}[[ARG0]]{{.*}} to min [[UBI0]]({{.*}})[{{.*}}, {{.*}}, [[ARG0]]]
  // CHECK-NEXT:      arith.addi %[[I]], %[[I]]
  return
}

// -----

// CHECK-DAG: [[LBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0)[s0, s1, s2] -> (d0 * s2 + s2 * 4, s0, 4096 floordiv s1)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<()[s0, s1, s2] -> (s0 ceildiv s2, (4096 floordiv s1) ceildiv s2)>

// CHECK: func @tile_loop_with_div_in_upper_bound_non_unit_step([[ARG0:%arg[0-9]+]]: index, %{{.*}}: memref<?xi32>, %{{.*}}: index, %{{.*}}: index)
#ub = affine_map<()[s0, s1] -> (s0, 4096 floordiv s1)>
func @tile_loop_with_div_in_upper_bound_non_unit_step(%t5 : index, %A : memref<? x i32>, %L : index, %U : index) {
  %c0 = arith.constant 0 : index
  %M = memref.dim %A, %c0 : memref<? x i32>
  affine.for %i = 0 to min #ub()[%M, %U] step 4 {
    arith.addi %i, %i : index
  }
  // CHECK: affine.for [[ARG1:%arg[0-9]+]] = 0 to min [[UBO0]]()[%{{.*}}, %{{.*}}, [[ARG0]]]{{.*}} step 4{{.*}}
  // CHECK-NEXT:    affine.for %[[I:.*]] = [[LBI0]]([[ARG1]]){{.*}}[[ARG0]]{{.*}} to min [[UBI0]]({{.*}})[{{.*}}, {{.*}}, [[ARG0]]]{{.*}} step 4{{.*}}
  // CHECK-NEXT:      arith.addi %[[I]], %[[I]]
  return
}

// -----

// CHECK-DAG: [[LBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> ((d0 - 8) * s0 + 8)>
// CHECK-DAG: [[UBI2:#map[0-9]+]] = affine_map<(d0)[s0, s1] -> ((d0 - 8) * s1 + s1 * 4 + 8, s0 + 16)>
// CHECK-DAG: [[UBI1:#map[0-9]+]] = affine_map<(d0)[s0, s1] -> ((d0 - 8) * s1 + s1 + 8, s0 + 16)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> ((d0 - 8) * s0 + s0 + 8, 256)>
// CHECK-DAG: [[UBO1:#map[0-9]+]] = affine_map<()[s0, s1] -> ((s0 + 8) ceildiv s1 + 8)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<()[s0] -> (248 ceildiv s0 + 8)>

// CHECK: func @tile_loop_with_non_zero_lb([[ARG0:%arg[0-9]+]]: index, [[ARG1:%arg[0-9]+]]: index, [[ARG2:%arg[0-9]+]]: index, %{{.*}}: index)
// CHECK-NEXT:  affine.for [[ARG3:%arg[0-9+]]] = 8 to [[UBO0]]{{.*}}[[ARG0]]{{.*}}
// CHECK-NEXT:    affine.for [[ARG4:%arg[0-9+]]] = 8 to [[UBO1]]{{.*}}[[ARG1]]{{.*}}
// CHECK-NEXT:      affine.for [[ARG5:%arg[0-9+]]] = 8 to [[UBO1]]{{.*}}[[ARG2]]{{.*}} step 4
// CHECK-NEXT:        affine.for %[[I:.*]] = [[LBI0]]([[ARG3]]){{.*}}[[ARG0]]{{.*}} to min [[UBI0]]([[ARG3]]){{.*}}[[ARG0]]{{.*}}
// CHECK-NEXT:          affine.for %[[J:.*]] = [[LBI0]]([[ARG4]]){{.*}}[[ARG1]]{{.*}} to min [[UBI1]]([[ARG4]]){{.*}}[[ARG1]]{{.*}}
// CHECK-NEXT:            affine.for %[[K:.*]] = [[LBI0]]([[ARG5]]){{.*}}[[ARG2]]{{.*}} to min [[UBI2]]([[ARG5]]){{.*}}[[ARG2]]{{.*}}step 4{{.*}}
// CHECK-NEXT:              "test.foo"(%[[I]], %[[J]], %[[K]]) : (index, index, index) -> ()
#ubi = affine_map<()[s0] -> (s0 + 16)>
func @tile_loop_with_non_zero_lb(%t0: index, %t1: index, %t2: index, %U: index){
  affine.for %i = 8 to 256 {
    affine.for %j = 8 to #ubi()[%U] {
      affine.for %k = 8 to #ubi()[%U] step 4 {
        "test.foo"(%i, %j, %k) : (index, index, index) -> ()
      }
    }
  }
  return
}

// -----

// CHECK-DAG: [[LBI:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 256)>
// CHECK-DAG: [[UBI1:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0 + s0, 250)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<()[s0] -> (256 ceildiv s0)>
// CHECK-DAG: [[UBO1:#map[0-9]+]] = affine_map<()[s0] -> (250 ceildiv s0)>

// CHECK: func @simple_matmul([[ARG0:%arg[0-9]+]]: index, [[ARG1:%arg[0-9]+]]: index, [[ARG2:%arg[0-9]+]]: index{{.*}})
// CHECK-NEXT:   affine.for [[ARG3:%arg[0-9]+]] = 0 to [[UBO0]](){{.*}}[[ARG0]]{{.*}}
// CHECK-NEXT:     affine.for [[ARG4:%arg[0-9]+]] = 0 to [[UBO0]](){{.*}}[[ARG1]]{{.*}}
// CHECK-NEXT:       affine.for [[ARG5:%arg[0-9]+]] = 0 to [[UBO1]](){{.*}}[[ARG2]]{{.*}}
// CHECK-NEXT:         affine.for %[[I:.*]] = [[LBI]]{{.*}}[[ARG3]]{{.*}}[[ARG0]]{{.*}} to min [[UBI0]]{{.*}}[[ARG3]]{{.*}}[[ARG0]]{{.*}}
// CHECK-NEXT:          affine.for %[[J:.*]] = [[LBI]]{{.*}}[[ARG4]]{{.*}}[[ARG1]]{{.*}} to min [[UBI0]]{{.*}}[[ARG4]]{{.*}}[[ARG1]]{{.*}}
// CHECK-NEXT:            affine.for %[[K:.*]] = [[LBI]]{{.*}}[[ARG5]]{{.*}}[[ARG2]]{{.*}} to min [[UBI1]]{{.*}}[[ARG5]]{{.*}}[[ARG2]]{{.*}}
// CHECK-NEXT:                 affine.load %{{.*}}[%[[I]], %[[K]]]
// CHECK-NEXT:                 affine.load %{{.*}}[%[[K]], %[[J]]]
// CHECK-NEXT:                 affine.load %{{.*}}[%[[I]], %[[J]]]
// CHECK-NEXT:                 arith.mulf %{{.*}}
// CHECK-NEXT:                 arith.addf %{{.*}}
// CHECK-NEXT:                 affine.store %{{.*}}[%[[I]], %[[J]]]
func @simple_matmul(%t6 : index, %t7 : index, %t8 : index, %arg0: memref<256x256xvector<64xf32>>, %arg1: memref<256x256xvector<64xf32>>, %arg2: memref<256x256xvector<64xf32>>) -> memref<256x256xvector<64xf32>> {
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

// -----

// CHECK-DAG: [[LBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s1, s0)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>

// CHECK: func @tile_with_symbolic_loop_upper_bounds([[ARG0:%arg[0-9]+]]: index, [[ARG1:%arg[0-9]+]]: index{{.*}}){{.*}}
// CHECK:        affine.for [[ARG2:%arg[0-9]+]] = 0 to [[UBO0]](){{.*}}[[ARG0]]{{.*}}
// CHECK-NEXT:     affine.for [[ARG3:%arg[0-9]+]] = 0 to [[UBO0]](){{.*}}[[ARG1]]{{.*}}
// CHECK-NEXT:       affine.for %[[I0:.*]] = [[LBI0]]{{.*}}[[ARG2]]{{.*}}[[ARG0]]{{.*}} to min [[UBI0]]{{.*}}[[ARG2]]{{.*}}[[ARG0]]{{.*}}
// CHECK-NEXT:         affine.for %[[I1:.*]] = [[LBI0]]{{.*}}[[ARG3]]{{.*}}[[ARG1]]{{.*}} to min [[UBI0]]{{.*}}[[ARG3]]{{.*}}[[ARG1]]{{.*}}
// CHECK-NEXT:           affine.store %{{.*}}, %{{.*}}[%[[I0]], %[[I1]]] : memref<?x?xf32>
// CHECK-NEXT:           affine.for %[[I2:.*]] = 0 to %{{.*}} {
// CHECK-NEXT:             affine.load %{{.*}}%[[I0]], %[[I2]]
// CHECK-NEXT:             affine.load %{{.*}}%[[I2]], %[[I1]]
// CHECK-NEXT:             arith.mulf
// CHECK-NEXT:             affine.load %{{.*}}%[[I0]], %[[I1]]
// CHECK-NEXT:             arith.addf
// CHECK-NEXT:             affine.store %{{.*}}%[[I0]], %[[I1]]
func @tile_with_symbolic_loop_upper_bounds(%t9 : index, %t10: index, %arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
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

// -----

// CHECK-DAG: [[LBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0)[s0, s1, s2] -> (d0 * s2 + s2, s0 + s1)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<()[s0, s1, s2] -> ((s0 + s1) ceildiv s2)>

// CHECK: func @tile_with_loop_upper_bounds_in_two_symbols([[ARG0:%arg[0-9]+]]: index{{.*}}){{.*}}
func @tile_with_loop_upper_bounds_in_two_symbols(%t11 : index, %arg0: memref<?xf32>, %limit: index) {
  %c0 = arith.constant 0 : index
  %dim0 = memref.dim %arg0, %c0 : memref<?xf32>
  affine.for %i0 = 0 to affine_map<()[s0, s1] -> (s0 + s1)> ()[%dim0, %limit] {
    %v0 = affine.load %arg0[%i0] : memref<?xf32>
  }
  // CHECK:  affine.for [[ARG1:%arg[0-9]+]] = 0 to [[UBO0]]()[%{{.*}}, %{{.*}}, [[ARG0]]]
  // CHECK-NEXT:    affine.for %[[I:.*]] = [[LBI0]]([[ARG1]]){{.*}}[[ARG0]]{{.*}} to min [[UBI0]]([[ARG1]])[{{.*}}, {{.*}}, [[ARG0]]]
  // CHECK-NEXT:      affine.load %{{.*}}[%[[I]]]
  return
}

// -----

// CHECK-DAG: [[LBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: [[UBI1:#map[0-9]+]] = affine_map<(d0, d1)[s0, s1] -> (d1 * s1 + s1, d0 + s0 + 4)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0, d1)[s0, s1] -> (d1 * s1 + s1, d0 + s0 + 2)>
// CHECK-DAG: [[UBO1:#map[0-9]+]] = affine_map<(d0)[s0, s1] -> ((d0 + s0 + 4) ceildiv s1)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<(d0)[s0, s1] -> ((d0 + s0 + 2) ceildiv s1)>

// CHECK: func @tile_with_upper_bounds_in_dimensions_and_symbols([[ARG0:%arg[0-9]+]]: index, [[ARG1:%arg[0-9]+]]: index, [[ARG2:%arg[0-9]+]]: index, [[ARG3:%arg[0-9]+]]: index{{.*}}){{.*}}
// CHECK-NEXT: affine.for [[ARG4:%arg[0-9]+]] = 0 to [[UBO0]]({{.*}}){{.*}}[[ARG0]]
// CHECK-NEXT:   affine.for [[ARG5:%arg[0-9]+]] = 0 to [[UBO1]]({{.*}}){{.*}}[[ARG1]]
// CHECK-NEXT:     affine.for {{.*}} = [[LBI0]]([[ARG4]]){{.*}}[[ARG0]]{{.*}} to min [[UBI0]]({{.*}}, [[ARG4]]){{.*}}[[ARG0]]{{.*}}
// CHECK-NEXT:       affine.for {{.*}} = [[LBI0]]([[ARG5]]){{.*}}[[ARG1]]{{.*}} to min [[UBI1]]({{.*}}, [[ARG5]]){{.*}}[[ARG1]]{{.*}}
func @tile_with_upper_bounds_in_dimensions_and_symbols(%t12 : index, %t13 :index, %M: index, %N:  index, %K: index) {
  affine.for %i = 0 to affine_map<(d0)[s0] -> (d0 + s0 + 2)>(%M)[%K] {
    affine.for %j = 0 to affine_map<(d0)[s0] -> (d0 + s0 + 4)>(%N)[%K] {
      "test.foo" () : () -> ()
    }
  }
  return
}

// -----

// CHECK-DAG: [[LBI0:#map[0-9]+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: [[UBI1:#map[0-9]+]] = affine_map<(d0, d1)[s0, s1] -> (d1 * s1 + s1 * 4, d0 + s0 + 4)>
// CHECK-DAG: [[UBI0:#map[0-9]+]] = affine_map<(d0, d1)[s0, s1] -> (d1 * s1 + s1 * 2, d0 + s0 + 2)>
// CHECK-DAG: [[UBO1:#map[0-9]+]] = affine_map<(d0)[s0, s1] -> ((d0 + s0 + 4) ceildiv s1)>
// CHECK-DAG: [[UBO0:#map[0-9]+]] = affine_map<(d0)[s0, s1] -> ((d0 + s0 + 2) ceildiv s1)>

// CHECK: func @tile_with_upper_bounds_in_dimensions_and_symbols_non_unit_steps
// CHECK-SAME: ([[ARG0:%arg[0-9]+]]: index, [[ARG1:%arg[0-9]+]]: index, [[ARG2:%arg[0-9]+]]: index, [[ARG3:%arg[0-9]+]]: index{{.*}}){{.*}}
// CHECK-NEXT: affine.for [[ARG4:%arg[0-9]+]] = 0 to [[UBO0]]({{.*}}){{.*}}[[ARG0]]{{.*}} step 2{{.*}}
// CHECK-NEXT:   affine.for [[ARG5:%arg[0-9]+]] = 0 to [[UBO1]]({{.*}}){{.*}}[[ARG1]]{{.*}} step 4{{.*}}
// CHECK-NEXT:     affine.for {{.*}} = [[LBI0]]([[ARG4]]){{.*}}[[ARG0]]{{.*}} to min [[UBI0]]({{.*}}, [[ARG4]]){{.*}}[[ARG0]]{{.*}} step 2{{.*}}
// CHECK-NEXT:       affine.for {{.*}} = [[LBI0]]([[ARG5]]){{.*}}[[ARG1]]{{.*}} to min [[UBI1]]({{.*}}, [[ARG5]]){{.*}}[[ARG1]]{{.*}} step 4{{.*}}
func @tile_with_upper_bounds_in_dimensions_and_symbols_non_unit_steps(%t12 : index, %t13 :index, %M: index, %N :  index, %K: index) {
  affine.for %i = 0 to affine_map<(d0)[s0] -> (d0 + s0 + 2)>(%M)[%K] step 2 {
    affine.for %j = 0 to affine_map<(d0)[s0] -> (d0 + s0 + 4)>(%N)[%K] step 4 {
      "test.foo" () : () -> ()
    }
  }
  return
}
