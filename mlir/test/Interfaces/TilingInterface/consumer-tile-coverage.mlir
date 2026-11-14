// RUN: mlir-opt --transform-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s

func.func @reject_correlated_projections(%input: tensor<2xi32>, %init: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %empty = tensor.empty() : tensor<2xi32>
  %producer = scf.forall (%i) in (2) shared_outs(%out = %empty) -> tensor<2xi32> {
    %tile = tensor.extract_slice %input[%i] [1] [1] : tensor<2xi32> to tensor<1xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tile into %out[%i] [1] [1] : tensor<1xi32> into tensor<2xi32>
    }
  }
  // expected-error @below {{failed to fuse consumer of slice}}
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%producer, %producer : tensor<2xi32>, tensor<2xi32>) outs(%init : tensor<2x2xi32>) attrs = {consumer} {
  ^bb0(%a: i32, %b: i32, %unused: i32):
    %sum = arith.addi %a, %b : i32
    linalg.yield %sum : i32
  } -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %target = transform.structured.match attributes{consumer} in %root : (!transform.any_op) -> !transform.any_op
    %fused, %new_loop = transform.test.fuse_consumer %target into (%loop) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @fuse_identical_projections(%input: tensor<2xi32>, %init: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %empty = tensor.empty() : tensor<2xi32>
  %producer = scf.forall (%i) in (2) shared_outs(%out = %empty) -> tensor<2xi32> {
    %tile = tensor.extract_slice %input[%i] [1] [1] : tensor<2xi32> to tensor<1xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tile into %out[%i] [1] [1] : tensor<1xi32> into tensor<2xi32>
    }
  }
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%producer, %producer : tensor<2xi32>, tensor<2xi32>) outs(%init : tensor<2x2xi32>) attrs = {consumer} {
  ^bb0(%a: i32, %b: i32, %unused: i32):
    %sum = arith.addi %a, %b : i32
    linalg.yield %sum : i32
  } -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %target = transform.structured.match attributes{consumer} in %root : (!transform.any_op) -> !transform.any_op
    %fused, %new_loop = transform.test.fuse_consumer %target into (%loop) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func.func @fuse_identical_projections(
// CHECK: scf.forall
// CHECK: linalg.generic
// CHECK-SAME: ins({{.*}} : tensor<1xi32>, tensor<1xi32>)
// CHECK-SAME: outs({{.*}} : tensor<2x1xi32>)
// CHECK: tensor.parallel_insert_slice
// CHECK: return

// -----

func.func @fuse_full_slices(%input: tensor<2xi32>, %init: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %empty = tensor.empty() : tensor<2xi32>
  %producer = scf.forall (%i) in (1) shared_outs(%out = %empty) -> tensor<2xi32> {
    %tile = tensor.extract_slice %input[0] [2] [1] : tensor<2xi32> to tensor<2xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tile into %out[0] [2] [1] : tensor<2xi32> into tensor<2xi32>
    }
  }
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%producer, %producer : tensor<2xi32>, tensor<2xi32>) outs(%init : tensor<2x2xi32>) attrs = {consumer} {
  ^bb0(%a: i32, %b: i32, %unused: i32):
    %sum = arith.addi %a, %b : i32
    linalg.yield %sum : i32
  } -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %target = transform.structured.match attributes{consumer} in %root : (!transform.any_op) -> !transform.any_op
    %fused, %new_loop = transform.test.fuse_consumer %target into (%loop) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func.func @fuse_full_slices(
// CHECK: scf.forall
// CHECK: linalg.generic
// CHECK-SAME: ins({{.*}} : tensor<2xi32>, tensor<2xi32>)
// CHECK-SAME: outs({{.*}} : tensor<2x2xi32>)
// CHECK: tensor.parallel_insert_slice
// CHECK: return

// -----

func.func @reject_correlated_projections_for(%input: tensor<2xi32>, %init: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %empty = tensor.empty() : tensor<2xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %producer = scf.for %i = %c0 to %c2 step %c1 iter_args(%out = %empty) -> (tensor<2xi32>) {
    %tile = tensor.extract_slice %input[%i] [1] [1] : tensor<2xi32> to tensor<1xi32>
    %inserted = tensor.insert_slice %tile into %out[%i] [1] [1] : tensor<1xi32> into tensor<2xi32>
    scf.yield %inserted : tensor<2xi32>
  }
  // expected-error @below {{failed to fuse consumer of slice}}
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%producer, %producer : tensor<2xi32>, tensor<2xi32>) outs(%init : tensor<2x2xi32>) attrs = {consumer} {
  ^bb0(%a: i32, %b: i32, %unused: i32):
    %sum = arith.addi %a, %b : i32
    linalg.yield %sum : i32
  } -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    %target = transform.structured.match attributes{consumer} in %root : (!transform.any_op) -> !transform.any_op
    %fused, %new_loop = transform.test.fuse_consumer %target into (%loop) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @query_correlated_projections(%tile: tensor<1xi32>, %dest: tensor<2xi32>, %init: tensor<2x2xi32>, %offset: index) -> tensor<2x2xi32> {
  %slice = tensor.insert_slice %tile into %dest[%offset] [1] [1] : tensor<1xi32> into tensor<2xi32>
  // expected-remark @below {{cannot be fused with producer tensor.insert_slice ops}}
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%slice, %slice : tensor<2xi32>, tensor<2xi32>) outs(%init : tensor<2x2xi32>) {
  ^bb0(%a: i32, %b: i32, %unused: i32):
    %sum = arith.addi %a, %b : i32
    linalg.yield %sum : i32
  } -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %consumer = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.test.query_producer_fusability %consumer : !transform.any_op
    transform.yield
  }
}

// -----

func.func @query_identical_projections(%tile: tensor<1xi32>, %dest: tensor<2xi32>, %init: tensor<2x2xi32>, %offset: index) -> tensor<2x2xi32> {
  %slice = tensor.insert_slice %tile into %dest[%offset] [1] [1] : tensor<1xi32> into tensor<2xi32>
  // expected-remark @below {{can be fused with producer tensor.insert_slice ops}}
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%slice, %slice : tensor<2xi32>, tensor<2xi32>) outs(%init : tensor<2x2xi32>) {
  ^bb0(%a: i32, %b: i32, %unused: i32):
    %sum = arith.addi %a, %b : i32
    linalg.yield %sum : i32
  } -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %consumer = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.test.query_producer_fusability %consumer : !transform.any_op
    transform.yield
  }
}

// -----

func.func @query_full_slices(%tile: tensor<2xi32>, %dest: tensor<2xi32>, %init: tensor<2x2xi32>, %offset: index) -> tensor<2x2xi32> {
  %slice = tensor.insert_slice %tile into %dest[0] [2] [1] : tensor<2xi32> into tensor<2xi32>
  // expected-remark @below {{can be fused with producer tensor.insert_slice ops}}
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%slice, %slice : tensor<2xi32>, tensor<2xi32>) outs(%init : tensor<2x2xi32>) {
  ^bb0(%a: i32, %b: i32, %unused: i32):
    %sum = arith.addi %a, %b : i32
    linalg.yield %sum : i32
  } -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %consumer = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.test.query_producer_fusability %consumer : !transform.any_op
    transform.yield
  }
}
