// RUN: mlir-opt %s --transform-interpreter --split-input-file \
// RUN:     --verify-diagnostics | FileCheck %s

// UNSUPPORTED: target=aarch64-pc-windows-msvc

// CHECK-LABEL: func @test_loop_invariant_subset_hoisting(
//  CHECK-SAME:     %[[arg:.*]]: tensor<?xf32>
func.func @test_loop_invariant_subset_hoisting(%arg: tensor<?xf32>) -> tensor<?xf32> {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)
  // CHECK: %[[extract:.*]] = tensor.extract_slice %[[arg]]
  // CHECK: %[[for:.*]]:2 = scf.for {{.*}} iter_args(%[[t:.*]] = %[[arg]], %[[hoisted:.*]] = %[[extract]])
  // expected-remark @below{{new loop op}}
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%t = %arg) -> (tensor<?xf32>) {
    %1 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
    // CHECK: %[[foo:.*]] = "test.foo"(%[[hoisted]])
    %2 = "test.foo"(%1) : (tensor<5xf32>) -> (tensor<5xf32>)
    // Obfuscate the IR by inserting at offset %sub instead of 0; both of them
    // have the same value.
    %3 = tensor.insert_slice %2 into %t[0][5][1] : tensor<5xf32> into tensor<?xf32>
    // CHECK: scf.yield %[[t]], %[[foo]]
    scf.yield %3 : tensor<?xf32>
  }
  // CHECK: %[[insert:.*]] = tensor.insert_slice %[[for]]#1 into %[[for]]#0
  // CHECK: return %[[insert]]
  return %0 : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["tensor.extract_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    transform.loop.hoist_loop_invariant_subsets %0 : !transform.any_op
    // Make sure that the handles are still valid (and were updated in case of
    // the loop).

    %p = transform.num_associations %0 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{1}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>
    transform.debug.emit_remark_at %0, "new loop op" : !transform.any_op
    %p2 = transform.num_associations %1 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{1}}
    transform.debug.emit_param_as_remark %p2 : !transform.param<i64>
    %p3 = transform.num_associations %2 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{1}}
    transform.debug.emit_param_as_remark %p3 : !transform.param<i64>

    transform.yield
  }
}

// -----

// Checks that transform ops from LoopExtensionOps and SCFTransformOps can be
// used together.

// CHECK-LABEL: func @test_mixed_loop_extension_scf_transform(
func.func @test_mixed_loop_extension_scf_transform(%arg: tensor<?xf32>) -> tensor<?xf32> {
  %lb = "test.foo"() : () -> (index)
  %ub = "test.foo"() : () -> (index)
  %step = "test.foo"() : () -> (index)
  // CHECK: scf.for
  // CHECK: scf.for
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%t = %arg) -> (tensor<?xf32>) {
    %1 = "test.foo"(%t) : (tensor<?xf32>) -> (tensor<?xf32>)
    scf.yield %1 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.loop.hoist_loop_invariant_subsets %0 : !transform.any_op
    transform.loop.unroll %0 { factor = 4 } : !transform.any_op
    transform.yield
  }
}
