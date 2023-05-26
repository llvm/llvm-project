// RUN: mlir-opt %s -test-transform-dialect-interpreter --split-input-file --verify-diagnostics

#map0 = affine_map<(d0) -> (d0 * 110)>
#map1 = affine_map<(d0) -> (696, d0 * 110 + 110)>
func.func @test_loops_do_not_get_coalesced() {
  affine.for %i = 0 to 7 {
    affine.for %j = #map0(%i) to min #map1(%i) {
    }
  } {coalesce}
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["affine.for"]} attributes {coalesce} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.cast %0 : !transform.any_op to !transform.op<"affine.for">
  // expected-error @below {{failed to coalesce}}
  %2 = transform.loop.coalesce %1: (!transform.op<"affine.for">) -> (!transform.op<"affine.for">)
}

// -----

func.func @test_loops_do_not_get_unrolled() {
  affine.for %i = 0 to 7 {
    arith.addi %i, %i : index
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.loop.get_parent_for %0 { affine = true } : (!transform.any_op) -> !transform.op<"affine.for">
  // expected-error @below {{failed to unroll}}
  transform.loop.unroll %1 { factor = 8 } : !transform.op<"affine.for">
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

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["scf.while"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{failed to outline}}
  transform.loop.outline %0 {func_name = "foo"} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
}
