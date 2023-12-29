// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics --allow-unregistered-dialect | FileCheck %s

func.func @if_no_else(%cond: i1, %a: index, %b: memref<?xf32>, %c: i8) {
  scf.if %cond {
    "some_op"(%cond, %b) : (i1, memref<?xf32>) -> ()
    scf.yield
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %if = transform.structured.match ops{["scf.if"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    // expected-error @+1 {{requires an scf.if op with a single-block `else` region}}
    transform.scf.take_assumed_branch %if take_else_branch
      : (!transform.any_op) -> ()
      transform.yield
  }
}

// -----

// CHECK-LABEL: if_no_else
func.func @if_no_else(%cond: i1, %a: index, %b: memref<?xf32>, %c: i8) {
  scf.if %cond {
    "some_op"(%cond, %b) : (i1, memref<?xf32>) -> ()
    scf.yield
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %if = transform.structured.match ops{["scf.if"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %some_op = transform.structured.match ops{["some_op"]} in %arg1
      : (!transform.any_op) -> !transform.any_op

    transform.scf.take_assumed_branch %if : (!transform.any_op) -> ()

    // Handle to formerly nested `some_op` is still valid after the transform.
    transform.print %some_op: !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: tile_tensor_pad
func.func @tile_tensor_pad(
  %arg0 : tensor<?x?xf32>, %cst : f32, %low: index, %high: index)
    -> tensor<20x40xf32>
{
  //     CHECK: scf.forall
  // CHECK-NOT:   scf.if
  // CHECK-NOT:     tensor.generate
  // CHECK-NOT:   else
  //     CHECK:     tensor.pad {{.*}} nofold
  %0 = tensor.pad %arg0 nofold low[%low, %low] high[%high, %high] {
        ^bb0(%arg9: index, %arg10: index):
          tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<20x40xf32>
  return %0 : tensor<20x40xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.tile_using_forall %0 tile_sizes[1, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %if = transform.structured.match ops{["scf.if"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.scf.take_assumed_branch %if take_else_branch
      : (!transform.any_op) -> ()
      transform.yield
  }
}
