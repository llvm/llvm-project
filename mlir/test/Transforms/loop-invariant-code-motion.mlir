// RUN: mlir-opt %s  -split-input-file -loop-invariant-code-motion | FileCheck %s

func.func @nested_loops_both_having_invariant_code() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = arith.addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = arith.addf %v0, %cf8 : f32
      affine.store %v0, %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[CST0:.*]] = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[CST1:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: %[[ADD0:.*]] = arith.addf %[[CST0]], %[[CST1]] : f32
  // CHECK-NEXT: arith.addf %[[ADD0]], %[[CST1]] : f32
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.store

  return
}

// -----

func.func @nested_loops_code_invariant_to_both() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      %v0 = arith.addf %cf7, %cf8 : f32
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = arith.addf %cst, %cst_0 : f32

  return
}

// -----

func.func @single_loop_nothing_invariant() {
  %m1 = memref.alloc() : memref<10xf32>
  %m2 = memref.alloc() : memref<10xf32>
  affine.for %arg0 = 0 to 10 {
    %v0 = affine.load %m1[%arg0] : memref<10xf32>
    %v1 = affine.load %m2[%arg0] : memref<10xf32>
    %v2 = arith.addf %v0, %v1 : f32
    affine.store %v2, %m1[%arg0] : memref<10xf32>
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %1 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: %2 = affine.load %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: %3 = affine.load %1[%arg0] : memref<10xf32>
  // CHECK-NEXT: %4 = arith.addf %2, %3 : f32
  // CHECK-NEXT: affine.store %4, %0[%arg0] : memref<10xf32>

  return
}

// -----

func.func @invariant_code_inside_affine_if() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %t0 = affine.apply affine_map<(d1) -> (d1 + 1)>(%arg0)
    affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %t0) {
        %cf9 = arith.addf %cf8, %cf8 : f32
        affine.store %cf9, %m[%arg0] : memref<10xf32>

    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: %1 = affine.apply #map(%arg0)
  // CHECK-NEXT: affine.if #set(%arg0, %1) {
  // CHECK-NEXT: %2 = arith.addf %cst, %cst : f32
  // CHECK-NEXT: affine.store %2, %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: }


  return
}

// -----

func.func @invariant_affine_if() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = arith.addf %cf8, %cf8 : f32
      }
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[CST:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %[[ARG:.*]] = 0 to 10 {
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %[[ARG:.*]] = 0 to 10 {
  // CHECK-NEXT: affine.if #set(%[[ARG]], %[[ARG]]) {
  // CHECK-NEXT: arith.addf %[[CST]], %[[CST]] : f32
  // CHECK-NEXT: }

  return
}

// -----

func.func @invariant_affine_if2() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = arith.addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg1] : memref<10xf32>
      }
    }
  }

  // CHECK: memref.alloc
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: affine.store
  // CHECK-NEXT: }
  // CHECK-NEXT: }

  return
}

// -----

func.func @invariant_affine_nested_if() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
        %cf9 = arith.addf %cf8, %cf8 : f32
        affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf10 = arith.addf %cf9, %cf9 : f32
        }
      }
    }
  }

  // CHECK: memref.alloc
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

// -----

func.func @invariant_affine_nested_if_else() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = arith.addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg0] : memref<10xf32>
          affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
            %cf10 = arith.addf %cf9, %cf9 : f32
          } else {
            affine.store %cf9, %m[%arg1] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: memref.alloc
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: affine.store
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: } else {
  // CHECK-NEXT: affine.store
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

// -----

func.func @invariant_loop_dialect() {
  %ci0 = arith.constant 0 : index
  %ci10 = arith.constant 10 : index
  %ci1 = arith.constant 1 : index
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32
  scf.for %arg0 = %ci0 to %ci10 step %ci1 {
    scf.for %arg1 = %ci0 to %ci10 step %ci1 {
      %v0 = arith.addf %cf7, %cf8 : f32
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = arith.addf %cst, %cst_0 : f32

  return
}

// -----

func.func @variant_loop_dialect() {
  %ci0 = arith.constant 0 : index
  %ci10 = arith.constant 10 : index
  %ci1 = arith.constant 1 : index
  %m = memref.alloc() : memref<10xf32>
  scf.for %arg0 = %ci0 to %ci10 step %ci1 {
    scf.for %arg1 = %ci0 to %ci10 step %ci1 {
      %v0 = arith.addi %arg0, %arg1 : index
    }
  }

  // CHECK: %0 = memref.alloc() : memref<10xf32>
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: arith.addi

  return
}

// -----

func.func @parallel_loop_with_invariant() {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : i32
  %c8 = arith.constant 8 : i32
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
      %v0 = arith.addi %c7, %c8 : i32
      %v3 = arith.addi %arg0, %arg1 : index
  }

  // CHECK-LABEL: func @parallel_loop_with_invariant
  // CHECK: %c0 = arith.constant 0 : index
  // CHECK-NEXT: %c10 = arith.constant 10 : index
  // CHECK-NEXT: %c1 = arith.constant 1 : index
  // CHECK-NEXT: %c7_i32 = arith.constant 7 : i32
  // CHECK-NEXT: %c8_i32 = arith.constant 8 : i32
  // CHECK-NEXT: arith.addi %c7_i32, %c8_i32 : i32
  // CHECK-NEXT: scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c10, %c10) step (%c1, %c1)
  // CHECK-NEXT:   arith.addi %arg0, %arg1 : index
  // CHECK-NEXT:   yield
  // CHECK-NEXT: }
  // CHECK-NEXT: return

  return
}

// -----

func.func private @make_val() -> (index)

// CHECK-LABEL: func @nested_uses_inside
func.func @nested_uses_inside(%lb: index, %ub: index, %step: index) {
  %true = arith.constant true

  // Check that ops that contain nested uses to values not defiend outside 
  // remain in the loop.
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: scf.for
  // CHECK-NEXT:   call @
  // CHECK-NEXT:   call @
  // CHECK-NEXT:   scf.if
  // CHECK-NEXT:     scf.yield
  // CHECK-NEXT:   else
  // CHECK-NEXT:     scf.yield
  scf.for %i = %lb to %ub step %step {
    %val = func.call @make_val() : () -> (index)
    %val2 = func.call @make_val() : () -> (index)
    %r = scf.if %true -> (index) {
      scf.yield %val: index
    } else {
      scf.yield %val2: index
    }
  }
  return
}

// -----

// Test that two ops that feed into each other are moved without violating
// dominance in non-graph regions.
// CHECK-LABEL: func @invariant_subgraph
// CHECK-SAME: %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %[[ARG:.*]]: i32
func.func @invariant_subgraph(%lb: index, %ub: index, %step: index, %arg: i32) {
  // CHECK:      %[[V0:.*]] = arith.addi %[[ARG]], %[[ARG]]
  // CHECK-NEXT: %[[V1:.*]] = arith.addi %[[ARG]], %[[V0]]
  // CHECK-NEXT: scf.for
  scf.for %i = %lb to %ub step %step {
    // CHECK-NEXT: "test.sink"(%[[V1]])
    %v0 = arith.addi %arg, %arg : i32
    %v1 = arith.addi %arg, %v0 : i32
    "test.sink"(%v1) : (i32) -> ()
  }
  return
}

// -----

// Test invariant nested loop is hoisted.
// CHECK-LABEL: func @test_invariant_nested_loop
func.func @test_invariant_nested_loop() {
  // CHECK: %[[C:.*]] = arith.constant
  %0 = arith.constant 5 : i32
  // CHECK: %[[V0:.*]] = arith.addi %[[C]], %[[C]]
  // CHECK-NEXT: %[[V1:.*]] = arith.addi %[[V0]], %[[C]]
  // CHECK-NEXT: test.graph_loop
  // CHECK-NEXT: ^bb0(%[[ARG0:.*]]: i32)
  // CHECK-NEXT: %[[V2:.*]] = arith.subi %[[ARG0]], %[[ARG0]]
  // CHECK-NEXT: test.region_yield %[[V2]]
  // CHECK: test.graph_loop
  // CHECK-NEXT: test.region_yield %[[V1]]
  test.graph_loop {
    %1 = arith.addi %0, %0 : i32
    %2 = arith.addi %1, %0 : i32
    test.graph_loop {
    ^bb0(%arg0: i32):
      %3 = arith.subi %arg0, %arg0 : i32
      test.region_yield %3 : i32
    } : () -> ()
    test.region_yield %2 : i32
  } : () -> ()
  return
}


// -----

// Test ops in a graph region are hoisted.
// CHECK-LABEL: func @test_invariants_in_graph_region
func.func @test_invariants_in_graph_region() {
  // CHECK: test.single_no_terminator_op
  test.single_no_terminator_op : {
    // CHECK-NEXT: %[[C:.*]] = arith.constant
    // CHECK-NEXT: %[[V1:.*]] = arith.addi %[[C]], %[[C]]
    // CHECK-NEXT: %[[V0:.*]] = arith.addi %[[C]], %[[V1]]
    test.graph_loop {
      %v0 = arith.addi %c0, %v1 : i32
      %v1 = arith.addi %c0, %c0 : i32
      %c0 = arith.constant 5 : i32
      test.region_yield %v0 : i32
    } : () -> ()
  }
  return
}

// -----

// Test ops in a graph region are hoisted in topological order into non-graph
// regions and that dominance is preserved.
// CHECK-LABEL: func @test_invariant_backedge
func.func @test_invariant_backedge() {
  // CHECK-NEXT: %[[C:.*]] = arith.constant
  // CHECK-NEXT: %[[V1:.*]] = arith.addi %[[C]], %[[C]]
  // CHECK-NEXT: %[[V0:.*]] = arith.addi %[[C]], %[[V1]]
  // CHECK-NEXT: test.graph_loop
  test.graph_loop {
    // CHECK-NEXT: test.region_yield %[[V0]]
    %v0 = arith.addi %c0, %v1 : i32
    %v1 = arith.addi %c0, %c0 : i32
    %c0 = arith.constant 5 : i32
    test.region_yield %v0 : i32
  } : () -> ()
  return
}

// -----

// Test that cycles aren't hoisted from graph regions to non-graph regions.
// CHECK-LABEL: func @test_invariant_cycle_not_hoisted
func.func @test_invariant_cycle_not_hoisted() {
  // CHECK: test.graph_loop
  test.graph_loop {
    // CHECK-NEXT: %[[A:.*]] = "test.a"(%[[B:.*]]) :
    // CHECK-NEXT: %[[B]] = "test.b"(%[[A]]) :
    // CHECK-NEXT: test.region_yield %[[A]]
    %a = "test.a"(%b) : (i32) -> i32
    %b = "test.b"(%a) : (i32) -> i32
    test.region_yield %a : i32
  } : () -> ()
  return
}
