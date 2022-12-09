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

  // CHECK: memref.alloc() : memref<10xf32>
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

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.addf

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

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: affine.for 
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: affine.load
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: affine.store

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

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.apply
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: affine.store
  // CHECK-NEXT: }


  return
}

// -----

func.func @invariant_affine_if() {
  %m = memref.alloc() : memref<10xf32>
  %cf8 = arith.constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 20 {
      affine.if affine_set<(d0, d1) : (d1 - d0 >= 0)> (%arg0, %arg0) {
          %cf9 = arith.addf %cf8, %cf8 : f32
      }
    }
  }

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: %[[CST:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %[[ARG:.*]] = 0 to 20 {
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %[[ARG:.*]] = 0 to 10 {
  // CHECK-NEXT: affine.if #set(%[[ARG]], %[[ARG]]) {
  // CHECK-NEXT: arith.addf %[[CST]], %[[CST]] : f32
  // CHECK-NEXT: }

  return
}

// -----

func.func @hoist_affine_for_with_unknown_trip_count(%lb: index, %ub: index) {
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = %lb to %ub {
    }
  }

  // CHECK: @hoist_affine_for_with_unknown_trip_count(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) {
  // CHECK-NEXT: affine.for %[[ARG2:.*]] = %[[ARG0]] to %[[ARG1]] {
  // CHECK-NEXT: }
  // CHECK-NEXT: affine.for %[[ARG3:.*]] = 0 to 10 {
  // CHECK-NEXT: }

  return
}

// -----

func.func @hoist_affine_for_with_unknown_trip_count_non_unit_step(%lb: index, %ub: index) {
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = %lb to %ub step 2 {
    }
  }

  // CHECK: @hoist_affine_for_with_unknown_trip_count_non_unit_step(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) {
  // CHECK-NEXT: affine.for %[[ARG2:.*]] = 0 to 10 {
  // CHECK-NEXT: affine.for %[[ARG3:.*]] = %[[ARG0]] to %[[ARG1]] step 2 {
  // CHECK-NEXT: }
  // CHECK-NEXT: }

  return
}

// -----

func.func @hoist_scf_for_with_unknown_trip_count_unit_step(%lb: index, %ub: index) {
  %c1 = arith.constant 1 : index
  scf.for %arg0 = %lb to %ub step %c1 {
    scf.for %arg1 = %lb to %ub step %c1 {
    }
  }

  // CHECK: @hoist_scf_for_with_unknown_trip_count_unit_step(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) {
  // CHECK: scf.for %[[ARG2:.*]] = %[[ARG0]] to %[[ARG1]]
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.for %[[ARG3:.*]] = %[[ARG0]] to %[[ARG1]]
  // CHECK-NEXT: }

  return
}

// -----

func.func @hoist_scf_for_with_unknown_trip_count_non_unit_constant_step(%lb: index, %ub: index) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %arg0 = %lb to %ub step %c1 {
    scf.for %arg1 = %lb to %ub step %c2 {
    }
  }

  // CHECK: @hoist_scf_for_with_unknown_trip_count_non_unit_constant_step(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) {
  // CHECK: scf.for %[[ARG2:.*]] = %[[ARG0]] to %[[ARG1]]
  // CHECK-NEXT: scf.for %[[ARG3:.*]] = %[[ARG0]] to %[[ARG1]]
  // CHECK-NEXT: }
  // CHECK-NEXT: }

  return
}

// -----

func.func @hoist_scf_for_with_unknown_trip_count_unknown_step(%lb: index, %ub: index, %step: index) {
  %c1 = arith.constant 1 : index
  scf.for %arg0 = %lb to %ub step %c1 {
    scf.for %arg1 = %lb to %ub step %step {
    }
  }

  // CHECK: @hoist_scf_for_with_unknown_trip_count_unknown_step(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[STEP:.*]]: index) {
  // CHECK: scf.for %[[ARG2:.*]] = %[[ARG0]] to %[[ARG1]]
  // CHECK-NEXT: scf.for %[[ARG3:.*]] = %[[ARG0]] to %[[ARG1]] step %[[STEP]]
  // CHECK-NEXT: }
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

  // CHECK: memref.alloc() : memref<10xf32>
  // CHECK-NEXT: arith.constant 7.000000e+00 : f32
  // CHECK-NEXT: arith.constant 8.000000e+00 : f32
  // CHECK-NEXT: arith.addf

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

  // CHECK: memref.alloc() : memref<10xf32>
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
  // CHECK: arith.constant 0 : index
  // CHECK-NEXT: arith.constant 10 : index
  // CHECK-NEXT: arith.constant 1 : index
  // CHECK-NEXT: arith.constant 7 : i32
  // CHECK-NEXT: arith.constant 8 : i32
  // CHECK-NEXT: arith.addi
  // CHECK-NEXT: scf.parallel (%[[A:.*]],{{.*}}) =
  // CHECK-NEXT:   arith.addi %[[A]]
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

// -----

// CHECK-LABEL: test_always_speculatable_op
func.func @test_always_speculatable_op(%lb: index, %ub: index, %step: index) {
  // CHECK: test.always_speculatable_op
  // CHECK-NEXT: scf.for
  scf.for %i = %lb to %ub step %step {
    %val = "test.always_speculatable_op"() : () -> i32
  }

  return
}

// CHECK-LABEL: test_never_speculatable_op
func.func @test_never_speculatable_op(%lb: index, %ub: index, %step: index) {
  // CHECK: scf.for
  // CHECK-NEXT: test.never_speculatable_op
  scf.for %i = %lb to %ub step %step {
    %val = "test.never_speculatable_op"() : () -> i32
  }

  return
}

// CHECK-LABEL: test_conditionally_speculatable_op_success
func.func @test_conditionally_speculatable_op_success(%lb: index, %ub: index, %step: index) {
  // CHECK: test.conditionally_speculatable_op
  // CHECK-NEXT: scf.for
  scf.for %i = %lb to %ub step %step {
    %const_val = arith.constant 5 : i32
    %val = "test.conditionally_speculatable_op"(%const_val) : (i32) -> i32
  }

  return
}

// CHECK-LABEL: test_conditionally_speculatable_op_failure
func.func @test_conditionally_speculatable_op_failure(%lb: index, %ub: index, %step: index, %arg: i32) {
  // CHECK: scf.for
  // CHECK-NEXT: test.conditionally_speculatable_op
  %const_5 = arith.constant 5 : i32
  %non_const = arith.addi %arg, %const_5 : i32
  scf.for %i = %lb to %ub step %step {
    %val = "test.conditionally_speculatable_op"(%non_const) : (i32) -> i32
  }

  return
}

// CHECK-LABEL: test_recursively_speculatable_op_success
func.func @test_recursively_speculatable_op_success(%lb: index, %ub: index, %step: index, %arg: i32) {
  // CHECK: test.recursively_speculatable_op
  // CHECK: scf.for
  scf.for %i = %lb to %ub step %step {
    %val = "test.recursively_speculatable_op"()({
      %result = arith.addi %arg, %arg : i32
      test.region_yield %result : i32
    }) : () -> i32
  }

  return
}

// CHECK-LABEL: test_recursively_speculatable_op_failure
func.func @test_recursively_speculatable_op_failure(%lb: index, %ub: index, %step: index, %arg: i32) {
  // CHECK: scf.for
  // CHECK-NEXT: test.recursively_speculatable_op
  scf.for %i = %lb to %ub step %step {
    %val = "test.recursively_speculatable_op"()({
      %result = "test.never_speculatable_op"() : () -> i32
      test.region_yield %result : i32
    }) : () -> i32
  }

  return
}

// -----

func.func @speculate_tensor_dim_unknown_rank_unknown_dim(
// CHECK-LABEL: @speculate_tensor_dim_unknown_rank_unknown_dim
    %t: tensor<*xf32>, %dim_idx: index, %lb: index, %ub: index, %step: index) {
  // CHECK: scf.for
  // CHECK-NEXT: tensor.dim
  scf.for %i = %lb to %ub step %step {
    %val = tensor.dim %t, %dim_idx : tensor<*xf32>
  }

  return
}

func.func @speculate_tensor_dim_known_rank_unknown_dim(
// CHECK-LABEL: @speculate_tensor_dim_known_rank_unknown_dim
    %t: tensor<?x?x?x?xf32>, %dim_idx: index, %lb: index, %ub: index, %step: index) {
  // CHECK: scf.for
  // CHECK-NEXT: tensor.dim
  scf.for %i = %lb to %ub step %step {
    %val = tensor.dim %t, %dim_idx : tensor<?x?x?x?xf32>
  }

  return
}

func.func @speculate_tensor_dim_unknown_rank_known_dim(
// CHECK-LABEL: @speculate_tensor_dim_unknown_rank_known_dim
    %t: tensor<*xf32>, %dim_idx: index, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  // CHECK: scf.for
  // CHECK-NEXT: tensor.dim
  scf.for %i = %lb to %ub step %step {
    %val = tensor.dim %t, %c0 : tensor<*xf32>
  }

  return
}

func.func @speculate_tensor_dim_known_rank_known_dim_inbounds(
// CHECK-LABEL: @speculate_tensor_dim_known_rank_known_dim_inbounds
    %t: tensor<?x?x?x?xf32>, %dim_idx: index, %lb: index, %ub: index, %step: index) {
  %c1 = arith.constant 1 : index
  // CHECK: tensor.dim
  // CHECK-NEXT: scf.for
  scf.for %i = %lb to %ub step %step {
    %val = tensor.dim %t, %c1 : tensor<?x?x?x?xf32>
  }

  return
}

// -----

func.func @speculate_memref_dim_unknown_rank_unknown_dim(
// CHECK-LABEL: @speculate_memref_dim_unknown_rank_unknown_dim
    %t: memref<*xf32>, %dim_idx: index, %lb: index, %ub: index, %step: index) {
  // CHECK: scf.for
  // CHECK-NEXT: memref.dim
  scf.for %i = %lb to %ub step %step {
    %val = memref.dim %t, %dim_idx : memref<*xf32>
  }

  return
}

func.func @speculate_memref_dim_known_rank_unknown_dim(
// CHECK-LABEL: @speculate_memref_dim_known_rank_unknown_dim
    %t: memref<?x?x?x?xf32>, %dim_idx: index, %lb: index, %ub: index, %step: index) {
  // CHECK: scf.for
  // CHECK-NEXT: memref.dim
  scf.for %i = %lb to %ub step %step {
    %val = memref.dim %t, %dim_idx : memref<?x?x?x?xf32>
  }

  return
}

func.func @speculate_memref_dim_unknown_rank_known_dim(
// CHECK-LABEL: @speculate_memref_dim_unknown_rank_known_dim
    %t: memref<*xf32>, %dim_idx: index, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : index
  // CHECK: scf.for
  // CHECK-NEXT: memref.dim
  scf.for %i = %lb to %ub step %step {
    %val = memref.dim %t, %c0 : memref<*xf32>
  }

  return
}

func.func @speculate_memref_dim_known_rank_known_dim_inbounds(
// CHECK-LABEL: @speculate_memref_dim_known_rank_known_dim_inbounds
    %t: memref<?x?x?x?xf32>, %dim_idx: index, %lb: index, %ub: index, %step: index) {
  %c1 = arith.constant 1 : index
  // CHECK: memref.dim
  // CHECK-NEXT: scf.for
  scf.for %i = %lb to %ub step %step {
    %val = memref.dim %t, %c1 : memref<?x?x?x?xf32>
  }

  return
}

// -----

func.func @no_speculate_divui(
// CHECK-LABEL: @no_speculate_divui(
    %num: i32, %denom: i32, %lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
// CHECK: scf.for
// CHECK: arith.divui
    %val = arith.divui %num, %denom : i32
  }

  return
}

func.func @no_speculate_divsi(
// CHECK-LABEL: @no_speculate_divsi(
    %num: i32, %denom: i32, %lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
// CHECK: scf.for
// CHECK: arith.divsi
    %val = arith.divsi %num, %denom : i32
  }

  return
}

func.func @no_speculate_ceildivui(
// CHECK-LABEL: @no_speculate_ceildivui(
    %num: i32, %denom: i32, %lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
// CHECK: scf.for
// CHECK: arith.ceildivui
    %val = arith.ceildivui %num, %denom : i32
  }

  return
}

func.func @no_speculate_ceildivsi(
// CHECK-LABEL: @no_speculate_ceildivsi(
    %num: i32, %denom: i32, %lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
// CHECK: scf.for
// CHECK: arith.ceildivsi
    %val = arith.ceildivsi %num, %denom : i32
  }

  return
}

func.func @no_speculate_divui_const(%num: i32, %lb: index, %ub: index, %step: index) {
// CHECK-LABEL: @no_speculate_divui_const(
  %c0 = arith.constant 0 : i32
  scf.for %i = %lb to %ub step %step {
// CHECK: scf.for
// CHECK: arith.divui
    %val = arith.divui %num, %c0 : i32
  }

  return
}

func.func @speculate_divui_const(
// CHECK-LABEL: @speculate_divui_const(
    %num: i32, %lb: index, %ub: index, %step: index) {
  %c5 = arith.constant 5 : i32
// CHECK: arith.divui
// CHECK: scf.for
  scf.for %i = %lb to %ub step %step {
    %val = arith.divui %num, %c5 : i32
  }

  return
}

func.func @no_speculate_ceildivui_const(%num: i32, %lb: index, %ub: index, %step: index) {
// CHECK-LABEL: @no_speculate_ceildivui_const(
  %c0 = arith.constant 0 : i32
  scf.for %i = %lb to %ub step %step {
// CHECK: scf.for
// CHECK: arith.ceildivui
    %val = arith.ceildivui %num, %c0 : i32
  }

  return
}

func.func @speculate_ceildivui_const(
// CHECK-LABEL: @speculate_ceildivui_const(
    %num: i32, %lb: index, %ub: index, %step: index) {
  %c5 = arith.constant 5 : i32
// CHECK: arith.ceildivui
// CHECK: scf.for
  scf.for %i = %lb to %ub step %step {
    %val = arith.ceildivui %num, %c5 : i32
  }

  return
}

func.func @no_speculate_divsi_const0(
// CHECK-LABEL: @no_speculate_divsi_const0(
    %num: i32, %denom: i32, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : i32
  scf.for %i = %lb to %ub step %step {
// CHECK: scf.for
// CHECK: arith.divsi
    %val = arith.divsi %num, %c0 : i32
  }

  return
}

func.func @no_speculate_divsi_const_minus1(
// CHECK-LABEL: @no_speculate_divsi_const_minus1(
    %num: i32, %denom: i32, %lb: index, %ub: index, %step: index) {
  %cm1 = arith.constant -1 : i32
  scf.for %i = %lb to %ub step %step {
// CHECK: scf.for
// CHECK: arith.divsi
    %val = arith.divsi %num, %cm1 : i32
  }

  return
}

func.func @speculate_divsi_const(
// CHECK-LABEL: @speculate_divsi_const(
    %num: i32, %denom: i32, %lb: index, %ub: index, %step: index) {
  %c5 = arith.constant 5 : i32
  scf.for %i = %lb to %ub step %step {
// CHECK: arith.divsi
// CHECK: scf.for
    %val = arith.divsi %num, %c5 : i32
  }

  return
}

func.func @no_speculate_ceildivsi_const0(
// CHECK-LABEL: @no_speculate_ceildivsi_const0(
    %num: i32, %denom: i32, %lb: index, %ub: index, %step: index) {
  %c0 = arith.constant 0 : i32
  scf.for %i = %lb to %ub step %step {
// CHECK: scf.for
// CHECK: arith.ceildivsi
    %val = arith.ceildivsi %num, %c0 : i32
  }

  return
}

func.func @no_speculate_ceildivsi_const_minus1(
// CHECK-LABEL: @no_speculate_ceildivsi_const_minus1(
    %num: i32, %denom: i32, %lb: index, %ub: index, %step: index) {
  %cm1 = arith.constant -1 : i32
  scf.for %i = %lb to %ub step %step {
// CHECK: scf.for
// CHECK: arith.ceildivsi
    %val = arith.ceildivsi %num, %cm1 : i32
  }

  return
}

func.func @speculate_ceildivsi_const(
// CHECK-LABEL: @speculate_ceildivsi_const(
    %num: i32, %denom: i32, %lb: index, %ub: index, %step: index) {
  %c5 = arith.constant 5 : i32
  scf.for %i = %lb to %ub step %step {
// CHECK: arith.ceildivsi
// CHECK: scf.for
    %val = arith.ceildivsi %num, %c5 : i32
  }

  return
}

// -----

func.func @speculate_static_pack_and_unpack(%source: tensor<128x256xf32>, 
  %dest: tensor<4x16x32x16xf32>, %lb: index, %ub: index, %step: index) {

  // CHECK: tensor.pack
  // CHECK-NEXT: scf.for  
  scf.for %i = %lb to %ub step %step {
    %packed = tensor.pack %source 
      inner_dims_pos = [0, 1] 
      inner_tiles = [32, 16] into %dest : tensor<128x256xf32> -> tensor<4x16x32x16xf32>
  }
  
  // CHECK: tensor.unpack
  // CHECK-NEXT: scf.for 
  scf.for %i = %lb to %ub step %step {
    %unpacked = tensor.unpack %dest
      inner_dims_pos = [0, 1] 
      inner_tiles = [32, 16] into %source : tensor<4x16x32x16xf32> -> tensor<128x256xf32>
  }
  return 
}

// -----

func.func @speculate_dynamic_pack_and_unpack(%source: tensor<?x?xf32>,
  %dest: tensor<?x?x?x?xf32>, %lb: index, %ub: index, %step: index,
  %tile_m: index, %tile_n: index, %pad: f32) {

  // CHECK: scf.for
  // CHECK-NEXT: tensor.pack
  scf.for %i = %lb to %ub step %step {
    %packed = tensor.pack %source
      inner_dims_pos = [0, 1]
      inner_tiles = [%tile_n, %tile_m] into %dest : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  }

  // CHECK: scf.for
  // CHECK-NEXT: tensor.unpack
  scf.for %i = %lb to %ub step %step {
    %unpacked = tensor.unpack %dest
      inner_dims_pos = [0, 1] 
      inner_tiles = [%tile_n, %tile_m] into %source : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  }

  // CHECK: tensor.pack
  // CHECK-NEXT: scf.for
  scf.for %i = %lb to %ub step %step {
    %packed = tensor.pack %source padding_value(%pad : f32)
      inner_dims_pos = [0, 1]
      inner_tiles = [%tile_n, %tile_m] into %dest : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  }
  return
}
