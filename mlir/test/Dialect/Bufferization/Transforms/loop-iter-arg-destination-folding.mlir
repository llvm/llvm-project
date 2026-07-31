// RUN: mlir-opt %s -loop-iter-arg-destination-folding -split-input-file -allow-unregistered-dialect | FileCheck %s

// A read-then-fully-overwritten iter_arg whose yielded value is a whole-tensor
// transfer_write into an outside tensor.empty has its write destination folded
// onto the iter_arg, making the carry in-place.

// CHECK-LABEL: func.func @fold_read_then_write
//       CHECK:   scf.for {{.*}} iter_args(%[[A:.*]] = %{{.*}})
//       CHECK:     vector.transfer_read %[[A]]
//       CHECK:     %[[W:.*]] = vector.transfer_write %{{.*}}, %[[A]]
//       CHECK:     scf.yield %[[W]]
func.func @fold_read_then_write(%init: tensor<128xf32>, %lb: index, %ub: index, %st: index, %pad: f32) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %scratch = tensor.empty() : tensor<128xf32>
  %r = scf.for %i = %lb to %ub step %st iter_args(%a = %init) -> (tensor<128xf32>) {
    %v = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
    %n = arith.addf %v, %v : vector<128xf32>
    %w = vector.transfer_write %n, %scratch[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
    scf.yield %w : tensor<128xf32>
  }
  return %r : tensor<128xf32>
}

// -----

// The iter_arg is read a SECOND time after the write. Folding is still legal:
// the write only feeds the yield, so it is sunk below every read of the iter_arg
// before its destination is folded onto the iter_arg. Both reads therefore still
// observe the incoming value; only the final store defines the next iteration.

// CHECK-LABEL: func.func @fold_read_after_write_via_sink
//       CHECK:   scf.for {{.*}} iter_args(%[[A:.*]] = %{{.*}})
//       CHECK:     %[[V1:.*]] = vector.transfer_read %[[A]]
//       CHECK:     %[[V2:.*]] = vector.transfer_read %[[A]]
//       CHECK:     arith.subf %[[V2]]
//       CHECK:     %[[W:.*]] = vector.transfer_write %{{.*}}, %[[A]]
//       CHECK:     scf.yield %[[W]]
func.func @fold_read_after_write_via_sink(%init: tensor<128xf32>, %lb: index, %ub: index, %st: index, %pad: f32) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %scratch = tensor.empty() : tensor<128xf32>
  %r = scf.for %i = %lb to %ub step %st iter_args(%a = %init) -> (tensor<128xf32>) {
    %v1 = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
    %n = arith.addf %v1, %v1 : vector<128xf32>
    %w = vector.transfer_write %n, %scratch[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
    %v2 = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
    %use = arith.subf %v2, %n : vector<128xf32>
    "test.keep"(%use) : (vector<128xf32>) -> ()
    scf.yield %w : tensor<128xf32>
  }
  return %r : tensor<128xf32>
}

// -----

// Multiple carried values sharing one outside empty: each safe slot is folded
// independently; the empty's other uses are untouched.

// CHECK-LABEL: func.func @fold_two_of_two
//       CHECK:   scf.for {{.*}} iter_args(%[[A:.*]] = %{{.*}}, %[[B:.*]] = %{{.*}})
//       CHECK:     vector.transfer_write %{{.*}}, %[[A]]
//       CHECK:     vector.transfer_write %{{.*}}, %[[B]]
//       CHECK:     scf.yield
func.func @fold_two_of_two(%i0: tensor<128xf32>, %i1: tensor<128xf32>, %lb: index, %ub: index, %st: index, %pad: f32) -> (tensor<128xf32>, tensor<128xf32>) {
  %c0 = arith.constant 0 : index
  %s0 = tensor.empty() : tensor<128xf32>
  %s1 = tensor.empty() : tensor<128xf32>
  %r:2 = scf.for %i = %lb to %ub step %st iter_args(%a = %i0, %b = %i1) -> (tensor<128xf32>, tensor<128xf32>) {
    %va = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
    %na = arith.addf %va, %va : vector<128xf32>
    %wa = vector.transfer_write %na, %s0[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
    %vb = vector.transfer_read %b[%c0], %pad {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
    %nb = arith.addf %vb, %vb : vector<128xf32>
    %wb = vector.transfer_write %nb, %s1[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
    scf.yield %wa, %wb : tensor<128xf32>, tensor<128xf32>
  }
  return %r#0, %r#1 : tensor<128xf32>, tensor<128xf32>
}

// -----

// The yielded write is a partial (masked) write, so it does not fully define the
// destination and reuse could drop live elements. Must NOT fold.

// CHECK-LABEL: func.func @no_fold_partial_write
//       CHECK:   %[[S:.*]] = tensor.empty() : tensor<128xf32>
//       CHECK:   scf.for
//       CHECK:     vector.transfer_write %{{.*}}, %[[S]]
func.func @no_fold_partial_write(%init: tensor<128xf32>, %lb: index, %ub: index, %st: index, %pad: f32, %mask: vector<128xi1>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %scratch = tensor.empty() : tensor<128xf32>
  %r = scf.for %i = %lb to %ub step %st iter_args(%a = %init) -> (tensor<128xf32>) {
    %v = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
    %n = arith.addf %v, %v : vector<128xf32>
    %w = vector.transfer_write %n, %scratch[%c0], %mask {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
    scf.yield %w : tensor<128xf32>
  }
  return %r : tensor<128xf32>
}

// -----

// The write destination is defined inside the loop (not an outside scratch), so
// there is no external buffer to fold away. Must NOT fold.

// CHECK-LABEL: func.func @no_fold_inside_empty
//       CHECK:   scf.for
//       CHECK:     %[[E:.*]] = tensor.empty() : tensor<128xf32>
//       CHECK:     %[[W:.*]] = vector.transfer_write %{{.*}}, %[[E]]
//       CHECK:     scf.yield %[[W]]
func.func @no_fold_inside_empty(%init: tensor<128xf32>, %lb: index, %ub: index, %st: index, %pad: f32) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index
  %r = scf.for %i = %lb to %ub step %st iter_args(%a = %init) -> (tensor<128xf32>) {
    %v = vector.transfer_read %a[%c0], %pad {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
    %n = arith.addf %v, %v : vector<128xf32>
    %scratch = tensor.empty() : tensor<128xf32>
    %w = vector.transfer_write %n, %scratch[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
    scf.yield %w : tensor<128xf32>
  }
  return %r : tensor<128xf32>
}
