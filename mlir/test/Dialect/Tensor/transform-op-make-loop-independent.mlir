// RUN: mlir-opt %s -allow-unregistered-dialect \
// RUN:     -test-transform-dialect-interpreter -canonicalize \
// RUN:     -split-input-file -verify-diagnostics | FileCheck %s

// This is a test case where "high" padding depends on the IV.

//       CHECK: #[[$map:.*]] = affine_map<()[s0, s1] -> (s0 - s1)>
//       CHECK: #[[$map1:.*]] = affine_map<(d0)[s0, s1] -> (-d0 + s0 + s1 + 5)>
// CHECK-LABEL: func @make_pad_loop_independent_1(
//  CHECK-SAME:     %[[lb:.*]]: index, %[[ub:.*]]: index, %[[step:.*]]: index,
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
func.func @make_pad_loop_independent_1(%lb: index, %ub: index, %step: index,
                                       %t: tensor<?xf32>, %f: f32) {
  // CHECK: scf.for %[[iv:.*]] = %[[lb]] to %[[ub]]
  scf.for %i = %lb to %ub step %step {
    // CHECK: %[[high:.*]] = affine.apply #[[$map]]()[%[[ub]], %[[lb]]]
    // CHECK: %[[padded:.*]] = tensor.pad %[[t]] low[5] high[%[[high]]]
    // CHECK: %[[dim:.*]] = tensor.dim %[[t]]
    // CHECK: %[[size:.*]] = affine.apply #[[$map1]](%[[iv]])[%[[ub]], %[[dim]]]
    // CHECK: %[[replacement:.*]] = tensor.extract_slice %[[padded]][0] [%[[size]]] [1]
    %high = affine.apply affine_map<(d0)[s0] -> (s0 - d0)> (%i)[%ub]
    %p = tensor.pad %t low[5] high[%high] {
    ^bb0(%arg1: index):
      tensor.yield %f : f32
    } : tensor<?xf32> to tensor<?xf32>
    // CHECK: "dummy.some_use"(%[[replacement]])
    "dummy.some_use"(%p) : (tensor<?xf32>) -> ()
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.tensor.make_loop_independent %0 {num_loops = 1} : (!transform.any_op) -> !transform.any_op
}

// -----

// This is a test case where "low" padding depends on the IV.

//       CHECK: #[[$map:.*]] = affine_map<()[s0, s1] -> (s0 - s1)>
//       CHECK: #[[$map1:.*]] = affine_map<(d0)[s0, s1] -> (-d0 + s0 + s1 + 5)>
//       CHECK: #[[$map2:.*]] = affine_map<(d0)[s0] -> (d0 - s0)>
// CHECK-LABEL: func @make_pad_loop_independent_1(
//  CHECK-SAME:     %[[lb:.*]]: index, %[[ub:.*]]: index, %[[step:.*]]: index,
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
func.func @make_pad_loop_independent_1(%lb: index, %ub: index, %step: index,
                                       %t: tensor<?xf32>, %f: f32) {
  // CHECK: scf.for %[[iv:.*]] = %[[lb]] to %[[ub]]
  scf.for %i = %lb to %ub step %step {
    // CHECK: %[[low:.*]] = affine.apply #[[$map]]()[%[[ub]], %[[lb]]]
    // CHECK: %[[padded:.*]] = tensor.pad %[[t]] low[%[[low]]] high[5]
    // CHECK: %[[dim:.*]] = tensor.dim %[[t]]
    // CHECK: %[[size:.*]] = affine.apply #[[$map1]](%[[iv]])[%[[ub]], %[[dim]]]
    // CHECK: %[[offset:.*]] = affine.apply #[[$map2]](%[[iv]])[%[[lb]]]
    // CHECK: %[[replacement:.*]] = tensor.extract_slice %[[padded]][%[[offset]]] [%[[size]]] [1]
    %low = affine.apply affine_map<(d0)[s0] -> (s0 - d0)> (%i)[%ub]
    %p = tensor.pad %t low[%low] high[5] {
    ^bb0(%arg1: index):
      tensor.yield %f : f32
    } : tensor<?xf32> to tensor<?xf32>
    // CHECK: "dummy.some_use"(%[[replacement]])
    "dummy.some_use"(%p) : (tensor<?xf32>) -> ()
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.tensor.make_loop_independent %0 {num_loops = 1} : (!transform.any_op) -> !transform.any_op
}

// -----

//       CHECK: #[[$map:.*]] = affine_map<()[s0] -> (s0 * 2 - 2)>
// CHECK-LABEL: func @two_loops(
func.func @two_loops(%lb: index, %ub: index, %step: index,
                     %t: tensor<?xf32>, %f: f32) {
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      // CHECK: affine.apply #map()[%{{.*}}]
      %low = affine.apply affine_map<(d0, d1)[] -> (d0 + d1)> (%i, %j)[]
      %p = tensor.pad %t low[%low] high[5] {
      ^bb0(%arg1: index):
        tensor.yield %f : f32
      } : tensor<?xf32> to tensor<?xf32>
      "dummy.some_use"(%p) : (tensor<?xf32>) -> ()
    }
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.tensor.make_loop_independent %0 {num_loops = 2} : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @not_enough_loops(%lb: index, %ub: index, %step: index,
                            %t: tensor<?xf32>, %f: f32) {
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      %low = affine.apply affine_map<(d0, d1)[] -> (d0 + d1)> (%i, %j)[]
      // expected-note@below {{target op}}
      %p = tensor.pad %t low[%low] high[5] {
      ^bb0(%arg1: index):
        tensor.yield %f : f32
      } : tensor<?xf32> to tensor<?xf32>
      "dummy.some_use"(%p) : (tensor<?xf32>) -> ()
    }
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error@below {{could not find 2-th enclosing loop}}
  %1 = transform.tensor.make_loop_independent %0 {num_loops = 3} : (!transform.any_op) -> !transform.any_op
}

// -----

// CHECK: #[[$map:.*]] = affine_map<(d0)[s0] -> (-d0 + s0)>
// CHECK: #[[$map1:.*]] = affine_map<()[s0, s1] -> (s0 - s1)>
// CHECK-LABEL: func @make_empty_loop_independent(
//  CHECK-SAME:     %[[lb:.*]]: index, %[[ub:.*]]: index, %[[step:.*]]: index)
func.func @make_empty_loop_independent(%lb: index, %ub: index, %step: index) {
  // CHECK: scf.for %[[iv:.*]] = %[[lb]] to %[[ub]]
  scf.for %i = %lb to %ub step %step {
    // CHECK: %[[slice_sz:.*]] = affine.apply #[[$map]](%[[iv]])[%[[ub]]]
    // CHECK: %[[empty_sz:.*]] = affine.apply #[[$map1]]()[%[[ub]], %[[lb]]]
    // CHECK: %[[empty:.*]] = tensor.empty(%[[empty_sz]]) : tensor<?xf32>
    // CHECK: %[[replacement:.*]] = tensor.extract_slice %[[empty]][0] [%[[slice_sz]]] [1]
    %sz = affine.apply affine_map<(d0)[s0] -> (s0 - d0)> (%i)[%ub]
    %empty = tensor.empty(%sz) : tensor<?xf32>
    // CHECK: "dummy.some_use"(%[[replacement]])
    "dummy.some_use"(%empty) : (tensor<?xf32>) -> ()
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["tensor.empty"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.tensor.make_loop_independent %0 {num_loops = 1} : (!transform.any_op) -> !transform.any_op
}
