// RUN: mlir-opt %s -transform-interpreter --split-input-file --verify-diagnostics

#map0 = affine_map<(d0) -> (d0 * 110)>
#map1 = affine_map<(d0) -> (696, d0 * 110 + 110)>
func.func @test_loops_do_not_get_coalesced() {
  affine.for %i = 0 to 7 {
    affine.for %j = #map0(%i) to min #map1(%i) {
    }
  } {coalesce}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["affine.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.cast %0 : !transform.any_op to !transform.op<"affine.for">
    // expected-error @below {{failed to coalesce}}
    %2 = transform.loop.coalesce %1: (!transform.op<"affine.for">) -> (!transform.op<"affine.for">)
    transform.yield
  }
}

// -----

func.func @test_loops_do_not_get_unrolled() {
  affine.for %i = 0 to 7 {
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "affine.for"} : (!transform.any_op) -> !transform.op<"affine.for">
    // expected-error @below {{failed to unroll}}
    transform.loop.unroll %1 { factor = 8 } : !transform.op<"affine.for">
    transform.yield
  }
}

// -----

func.func @loop_unroll_and_jam_unsupported_trip_count_not_multiple_of_factor() {
  %c0 = arith.constant 0 : index
  %c40 = arith.constant 40 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c40 step %c2 {
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    // expected-error @below {{failed to unroll and jam}}
    transform.loop.unroll_and_jam %1 { factor = 3 } : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

func.func @loop_unroll_and_jam_unsupported_loop_with_results() -> index {
  %c0 = arith.constant 0 : index
  %c40 = arith.constant 40 : index
  %c2 = arith.constant 2 : index
  %sum = scf.for %i = %c0 to %c40 step %c2 iter_args(%does_not_alias_aggregated = %c0) -> (index) {
    %sum = arith.addi %i, %i : index
    scf.yield %sum : index
  }
  return %sum : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    // expected-error @below {{failed to unroll and jam}}
    transform.loop.unroll_and_jam %1 { factor = 4 } : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

func.func private @loop_unroll_and_jam_unsupported_dynamic_trip_count(%arg0: memref<96x128xi8, 3>, %arg1: memref<128xi8, 3>) {
  %c96 = arith.constant 96 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  scf.for %arg4 = %c0 to %c4 step %c1 {
    scf.for %arg2 = %c0 to %c128 step %arg4 {
      %3 = memref.load %arg1[%arg2] : memref<128xi8, 3>
      %sum = scf.for %arg3 = %c0 to %c96 step %c1 iter_args(%does_not_alias_aggregated = %3) -> (i8) {
      %2 = memref.load %arg0[%arg3, %arg2] : memref<96x128xi8, 3>
      %4 = arith.addi %2, %3 : i8
      scf.yield %4 : i8
      }
      memref.store %sum, %arg1[%arg2] : memref<128xi8, 3>
    }
    scf.yield
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["memref.store"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    // expected-error @below {{failed to unroll and jam}}
    transform.loop.unroll_and_jam %1 { factor = 4 } : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

func.func @loop_unroll_and_jam_unsupported_dynamic_trip_count(%upper_bound: index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %upper_bound step %c2 {
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    // expected-error @below {{failed to unroll and jam}}
    transform.loop.unroll_and_jam %1 { factor = 2 } : !transform.op<"scf.for">
    transform.yield
  }
}

// -----

func.func private @cond() -> i1
func.func private @body()

func.func @loop_outline_op_multi_region() {
  // expected-note @below {{target op}}
  scf.while : () -> () {
    %0 = func.call @cond() : () -> i1
    scf.condition(%0)
  } do {
  ^bb0:
    func.call @body() : () -> ()
    scf.yield
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.while"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{failed to outline}}
    transform.loop.outline %0 {func_name = "foo"} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @test_loop_peeling_not_beneficial() {
  // Loop peeling is not beneficial because the step size already divides
  // ub - lb evenly. lb, ub and step are constant in this test case and the
  // "fast path" is exercised.
  %lb = arith.constant 0 : index
  %ub = arith.constant 40 : index
  %step = arith.constant 5 : index
  scf.for %i = %lb to %ub step %step {
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    // expected-error @below {{failed to peel}}
    transform.loop.peel %1 : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @test_loop_peeling_not_beneficial_already_peeled(%lb: index, %ub: index, %step: index) {
  // Loop peeling is not beneficial because the step size already divides
  // ub - lb evenly. This test case exercises the "slow path".
  %new_ub = affine.apply affine_map<()[s0, s1, s2] -> (s1 - (s1 - s0) mod s2)>()[%lb, %ub, %step]
  scf.for %i = %lb to %new_ub step %step {
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    // expected-error @below {{failed to peel}}
    transform.loop.peel %1 : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @test_loop_peeling_not_beneficial_already_peeled_lb_zero(%ub: index, %step: index) {
  // Loop peeling is not beneficial because the step size already divides
  // ub - lb evenly. This test case exercises the "slow path".
  %lb = arith.constant 0 : index
  %new_ub = affine.apply affine_map<()[s1, s2] -> (s1 - s1 mod s2)>()[%ub, %step]
  scf.for %i = %lb to %new_ub step %step {
    arith.addi %i, %i : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    // expected-error @below {{failed to peel}}
    transform.loop.peel %1 : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
