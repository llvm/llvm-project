// RUN: mlir-opt %s -transform-interpreter -split-input-file -canonicalize -cse -verify-diagnostics | FileCheck %s

// Stable argmax reproducer adapted from IREE #25063, reported by Cerulime.

// CHECK-LABEL: func.func @argmax_signed(
// CHECK: scf.forall
// CHECK: linalg.generic
// CHECK: linalg.reduce
// CHECK: ins(%{{.*}}, %{{.*}} :
// CHECK: arith.cmpi sgt
// CHECK: arith.cmpi eq
// CHECK: arith.cmpi slt
// CHECK: arith.select
// CHECK: arith.select
// CHECK: linalg.yield %{{.*}}, %{{.*}} : i32, i64
// CHECK-NOT: linalg.reduce
// CHECK: return
module attributes {transform.with_named_sequence} {
  func.func @argmax_signed(%input: tensor<1025x8xi32>)
      -> (tensor<8xi32>, tensor<8xi64>) {
    %min = arith.constant -2147483648 : i32
    %last = arith.constant 9223372036854775807 : i64
    %empty_values = tensor.empty() : tensor<8xi32>
    %values = linalg.fill ins(%min : i32)
        outs(%empty_values : tensor<8xi32>) -> tensor<8xi32>
    %empty_indices = tensor.empty() : tensor<8xi64>
    %indices = linalg.fill ins(%last : i64)
        outs(%empty_indices : tensor<8xi64>) -> tensor<8xi64>
    %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(r, c) -> (r, c)>,
        affine_map<(r, c) -> (c)>,
        affine_map<(r, c) -> (c)>
      ],
      iterator_types = ["reduction", "parallel"]
    } ins(%input : tensor<1025x8xi32>)
      outs(%values, %indices : tensor<8xi32>, tensor<8xi64>) {
    ^bb0(%incoming: i32, %best: i32, %best_index: i64):
      %r = linalg.index 0 : index
      %index = arith.index_cast %r : index to i64
      %maximum = arith.maxsi %incoming, %best : i32
      %greater = arith.cmpi sgt, %incoming, %best : i32
      %equal = arith.cmpi eq, %incoming, %best : i32
      %earlier = arith.cmpi slt, %index, %best_index : i64
      %tie = arith.andi %equal, %earlier : i1
      %take = arith.ori %greater, %tie : i1
      %selected = arith.select %take, %index, %best_index : i64
      linalg.yield %maximum, %selected : i32, i64
    } -> (tensor<8xi32>, tensor<8xi64>)
    return %result#0, %result#1 : tensor<8xi32>, tensor<8xi64>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %init_value, %init_index, %split, %merge, %loop = transform.structured.tile_reduction_using_forall %op
      by num_threads = [4, 0] tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @argmin_signed(
// CHECK: scf.forall
// CHECK: linalg.generic
// CHECK: linalg.reduce
// CHECK: ins(%{{.*}}, %{{.*}} :
// CHECK: arith.cmpi slt
// CHECK: arith.cmpi eq
// CHECK: arith.cmpi slt
// CHECK: arith.select
// CHECK: arith.select
// CHECK: linalg.yield %{{.*}}, %{{.*}} : i32, i64
// CHECK-NOT: linalg.reduce
// CHECK: return
module attributes {transform.with_named_sequence} {
  func.func @argmin_signed(%input: tensor<1025x8xi32>)
      -> (tensor<8xi32>, tensor<8xi64>) {
    %min = arith.constant 2147483647 : i32
    %last = arith.constant 9223372036854775807 : i64
    %empty_values = tensor.empty() : tensor<8xi32>
    %values = linalg.fill ins(%min : i32)
        outs(%empty_values : tensor<8xi32>) -> tensor<8xi32>
    %empty_indices = tensor.empty() : tensor<8xi64>
    %indices = linalg.fill ins(%last : i64)
        outs(%empty_indices : tensor<8xi64>) -> tensor<8xi64>
    %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(r, c) -> (r, c)>,
        affine_map<(r, c) -> (c)>,
        affine_map<(r, c) -> (c)>
      ],
      iterator_types = ["reduction", "parallel"]
    } ins(%input : tensor<1025x8xi32>)
      outs(%values, %indices : tensor<8xi32>, tensor<8xi64>) {
    ^bb0(%incoming: i32, %best: i32, %best_index: i64):
      %r = linalg.index 0 : index
      %index = arith.index_cast %r : index to i64
      %maximum = arith.minsi %incoming, %best : i32
      %greater = arith.cmpi slt, %incoming, %best : i32
      %equal = arith.cmpi eq, %incoming, %best : i32
      %earlier = arith.cmpi slt, %index, %best_index : i64
      %tie = arith.andi %equal, %earlier : i1
      %take = arith.ori %greater, %tie : i1
      %selected = arith.select %take, %index, %best_index : i64
      linalg.yield %maximum, %selected : i32, i64
    } -> (tensor<8xi32>, tensor<8xi64>)
    return %result#0, %result#1 : tensor<8xi32>, tensor<8xi64>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %init_value, %init_index, %split, %merge, %loop = transform.structured.tile_reduction_using_forall %op
      by num_threads = [4, 0] tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @argmax_unsigned(
// CHECK: scf.forall
// CHECK: linalg.generic
// CHECK: linalg.reduce
// CHECK: ins(%{{.*}}, %{{.*}} :
// CHECK: arith.cmpi ugt
// CHECK: arith.cmpi eq
// CHECK: arith.cmpi slt
// CHECK: arith.select
// CHECK: arith.select
// CHECK: linalg.yield %{{.*}}, %{{.*}} : i32, i64
// CHECK-NOT: linalg.reduce
// CHECK: return
module attributes {transform.with_named_sequence} {
  func.func @argmax_unsigned(%input: tensor<1025x8xi32>)
      -> (tensor<8xi32>, tensor<8xi64>) {
    %min = arith.constant 0 : i32
    %last = arith.constant 9223372036854775807 : i64
    %empty_values = tensor.empty() : tensor<8xi32>
    %values = linalg.fill ins(%min : i32)
        outs(%empty_values : tensor<8xi32>) -> tensor<8xi32>
    %empty_indices = tensor.empty() : tensor<8xi64>
    %indices = linalg.fill ins(%last : i64)
        outs(%empty_indices : tensor<8xi64>) -> tensor<8xi64>
    %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(r, c) -> (r, c)>,
        affine_map<(r, c) -> (c)>,
        affine_map<(r, c) -> (c)>
      ],
      iterator_types = ["reduction", "parallel"]
    } ins(%input : tensor<1025x8xi32>)
      outs(%values, %indices : tensor<8xi32>, tensor<8xi64>) {
    ^bb0(%incoming: i32, %best: i32, %best_index: i64):
      %r = linalg.index 0 : index
      %index = arith.index_cast %r : index to i64
      %maximum = arith.maxui %incoming, %best : i32
      %greater = arith.cmpi ugt, %incoming, %best : i32
      %equal = arith.cmpi eq, %incoming, %best : i32
      %earlier = arith.cmpi slt, %index, %best_index : i64
      %tie = arith.andi %equal, %earlier : i1
      %take = arith.ori %greater, %tie : i1
      %selected = arith.select %take, %index, %best_index : i64
      linalg.yield %maximum, %selected : i32, i64
    } -> (tensor<8xi32>, tensor<8xi64>)
    return %result#0, %result#1 : tensor<8xi32>, tensor<8xi64>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %init_value, %init_index, %split, %merge, %loop = transform.structured.tile_reduction_using_forall %op
      by num_threads = [4, 0] tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @argmin_unsigned(
// CHECK: scf.forall
// CHECK: linalg.generic
// CHECK: linalg.reduce
// CHECK: ins(%{{.*}}, %{{.*}} :
// CHECK: arith.cmpi ult
// CHECK: arith.cmpi eq
// CHECK: arith.cmpi slt
// CHECK: arith.select
// CHECK: arith.select
// CHECK: linalg.yield %{{.*}}, %{{.*}} : i32, i64
// CHECK-NOT: linalg.reduce
// CHECK: return
module attributes {transform.with_named_sequence} {
  func.func @argmin_unsigned(%input: tensor<1025x8xi32>)
      -> (tensor<8xi32>, tensor<8xi64>) {
    %min = arith.constant -1 : i32
    %last = arith.constant 9223372036854775807 : i64
    %empty_values = tensor.empty() : tensor<8xi32>
    %values = linalg.fill ins(%min : i32)
        outs(%empty_values : tensor<8xi32>) -> tensor<8xi32>
    %empty_indices = tensor.empty() : tensor<8xi64>
    %indices = linalg.fill ins(%last : i64)
        outs(%empty_indices : tensor<8xi64>) -> tensor<8xi64>
    %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(r, c) -> (r, c)>,
        affine_map<(r, c) -> (c)>,
        affine_map<(r, c) -> (c)>
      ],
      iterator_types = ["reduction", "parallel"]
    } ins(%input : tensor<1025x8xi32>)
      outs(%values, %indices : tensor<8xi32>, tensor<8xi64>) {
    ^bb0(%incoming: i32, %best: i32, %best_index: i64):
      %r = linalg.index 0 : index
      %index = arith.index_cast %r : index to i64
      %maximum = arith.minui %incoming, %best : i32
      %greater = arith.cmpi ult, %incoming, %best : i32
      %equal = arith.cmpi eq, %incoming, %best : i32
      %earlier = arith.cmpi slt, %index, %best_index : i64
      %tie = arith.andi %equal, %earlier : i1
      %take = arith.ori %greater, %tie : i1
      %selected = arith.select %take, %index, %best_index : i64
      linalg.yield %maximum, %selected : i32, i64
    } -> (tensor<8xi32>, tensor<8xi64>)
    return %result#0, %result#1 : tensor<8xi32>, tensor<8xi64>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %init_value, %init_index, %split, %merge, %loop = transform.structured.tile_reduction_using_forall %op
      by num_threads = [4, 0] tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @argmax_dynamic(
// CHECK: scf.forall
// CHECK: linalg.generic
// CHECK: linalg.reduce
// CHECK: ins(%{{.*}}, %{{.*}} :
// CHECK: arith.cmpi sgt
// CHECK: arith.cmpi eq
// CHECK: arith.cmpi slt
// CHECK: arith.select
// CHECK: arith.select
// CHECK: linalg.yield %{{.*}}, %{{.*}} : i32, i64
// CHECK-NOT: linalg.reduce
// CHECK: return
module attributes {transform.with_named_sequence} {
  func.func @argmax_dynamic(%input: tensor<?x8xi32>)
      -> (tensor<8xi32>, tensor<8xi64>) {
    %min = arith.constant -2147483648 : i32
    %last = arith.constant 9223372036854775807 : i64
    %empty_values = tensor.empty() : tensor<8xi32>
    %values = linalg.fill ins(%min : i32)
        outs(%empty_values : tensor<8xi32>) -> tensor<8xi32>
    %empty_indices = tensor.empty() : tensor<8xi64>
    %indices = linalg.fill ins(%last : i64)
        outs(%empty_indices : tensor<8xi64>) -> tensor<8xi64>
    %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(r, c) -> (r, c)>,
        affine_map<(r, c) -> (c)>,
        affine_map<(r, c) -> (c)>
      ],
      iterator_types = ["reduction", "parallel"]
    } ins(%input : tensor<?x8xi32>)
      outs(%values, %indices : tensor<8xi32>, tensor<8xi64>) {
    ^bb0(%incoming: i32, %best: i32, %best_index: i64):
      %r = linalg.index 0 : index
      %index = arith.index_cast %r : index to i64
      %maximum = arith.maxsi %incoming, %best : i32
      %greater = arith.cmpi sgt, %incoming, %best : i32
      %equal = arith.cmpi eq, %incoming, %best : i32
      %earlier = arith.cmpi slt, %index, %best_index : i64
      %tie = arith.andi %equal, %earlier : i1
      %take = arith.ori %greater, %tie : i1
      %selected = arith.select %take, %index, %best_index : i64
      linalg.yield %maximum, %selected : i32, i64
    } -> (tensor<8xi32>, tensor<8xi64>)
    return %result#0, %result#1 : tensor<8xi32>, tensor<8xi64>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %init_value, %init_index, %split, %merge, %loop = transform.structured.tile_reduction_using_forall %op
      by num_threads = [4, 0] tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @argmax_paired_selects(
// CHECK: scf.forall
// CHECK: linalg.generic
// CHECK: linalg.reduce
// CHECK: ins(%{{.*}}, %{{.*}} :
// CHECK: arith.cmpi sgt
// CHECK: arith.cmpi eq
// CHECK: arith.cmpi slt
// CHECK: arith.select
// CHECK: arith.select
// CHECK: linalg.yield %{{.*}}, %{{.*}} : i32, i64
// CHECK-NOT: linalg.reduce
// CHECK: return
module attributes {transform.with_named_sequence} {
  func.func @argmax_paired_selects(%input: tensor<1025x8xi32>)
      -> (tensor<8xi32>, tensor<8xi64>) {
    %min = arith.constant -2147483648 : i32
    %last = arith.constant 9223372036854775807 : i64
    %empty_values = tensor.empty() : tensor<8xi32>
    %values = linalg.fill ins(%min : i32)
        outs(%empty_values : tensor<8xi32>) -> tensor<8xi32>
    %empty_indices = tensor.empty() : tensor<8xi64>
    %indices = linalg.fill ins(%last : i64)
        outs(%empty_indices : tensor<8xi64>) -> tensor<8xi64>
    %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(r, c) -> (r, c)>,
        affine_map<(r, c) -> (c)>,
        affine_map<(r, c) -> (c)>
      ],
      iterator_types = ["reduction", "parallel"]
    } ins(%input : tensor<1025x8xi32>)
      outs(%values, %indices : tensor<8xi32>, tensor<8xi64>) {
    ^bb0(%incoming: i32, %best: i32, %best_index: i64):
      %r = linalg.index 0 : index
      %index = arith.index_cast %r : index to i64
      %greater = arith.cmpi sgt, %incoming, %best : i32
      %equal = arith.cmpi eq, %incoming, %best : i32
      %earlier = arith.cmpi slt, %index, %best_index : i64
      %tie = arith.andi %equal, %earlier : i1
      %take = arith.ori %greater, %tie : i1
      %selected = arith.select %take, %index, %best_index : i64
      %maximum = arith.select %take, %incoming, %best : i32
      linalg.yield %maximum, %selected : i32, i64
    } -> (tensor<8xi32>, tensor<8xi64>)
    return %result#0, %result#1 : tensor<8xi32>, tensor<8xi64>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    %init_value, %init_index, %split, %merge, %loop = transform.structured.tile_reduction_using_forall %op
      by num_threads = [4, 0] tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  func.func @reject_implicit_tie(%input: tensor<1025x8xi32>)
      -> (tensor<8xi32>, tensor<8xi64>) {
    %min = arith.constant -2147483648 : i32
    %last = arith.constant 9223372036854775807 : i64
    %empty_values = tensor.empty() : tensor<8xi32>
    %values = linalg.fill ins(%min : i32)
        outs(%empty_values : tensor<8xi32>) -> tensor<8xi32>
    %empty_indices = tensor.empty() : tensor<8xi64>
    %indices = linalg.fill ins(%last : i64)
        outs(%empty_indices : tensor<8xi64>) -> tensor<8xi64>
    // expected-error @below {{Failed to anaysis the reduction operation.}}
    %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(r, c) -> (r, c)>,
        affine_map<(r, c) -> (c)>,
        affine_map<(r, c) -> (c)>
      ],
      iterator_types = ["reduction", "parallel"]
    } ins(%input : tensor<1025x8xi32>)
      outs(%values, %indices : tensor<8xi32>, tensor<8xi64>) {
    ^bb0(%incoming: i32, %best: i32, %best_index: i64):
      %r = linalg.index 0 : index
      %index = arith.index_cast %r : index to i64
      %maximum = arith.maxsi %incoming, %best : i32
      %greater = arith.cmpi sgt, %incoming, %best : i32
      %equal = arith.cmpi eq, %incoming, %best : i32
      %earlier = arith.cmpi slt, %index, %best_index : i64
      %tie = arith.andi %equal, %earlier : i1
      %take = arith.ori %greater, %greater : i1
      %selected = arith.select %take, %index, %best_index : i64
      linalg.yield %maximum, %selected : i32, i64
    } -> (tensor<8xi32>, tensor<8xi64>)
    return %result#0, %result#1 : tensor<8xi32>, tensor<8xi64>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{could not tile reduction}}
    %init_value, %init_index, %split, %merge, %loop = transform.structured.tile_reduction_using_forall %op
      by num_threads = [4, 0] tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  func.func @reject_wrong_index_dependency(%input: tensor<1025x8xi32>)
      -> (tensor<8xi32>, tensor<8xi64>) {
    %min = arith.constant -2147483648 : i32
    %last = arith.constant 9223372036854775807 : i64
    %empty_values = tensor.empty() : tensor<8xi32>
    %values = linalg.fill ins(%min : i32)
        outs(%empty_values : tensor<8xi32>) -> tensor<8xi32>
    %empty_indices = tensor.empty() : tensor<8xi64>
    %indices = linalg.fill ins(%last : i64)
        outs(%empty_indices : tensor<8xi64>) -> tensor<8xi64>
    // expected-error @below {{Failed to anaysis the reduction operation.}}
    %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(r, c) -> (r, c)>,
        affine_map<(r, c) -> (c)>,
        affine_map<(r, c) -> (c)>
      ],
      iterator_types = ["reduction", "parallel"]
    } ins(%input : tensor<1025x8xi32>)
      outs(%values, %indices : tensor<8xi32>, tensor<8xi64>) {
    ^bb0(%incoming: i32, %best: i32, %best_index: i64):
      %r = linalg.index 0 : index
      %index = arith.index_cast %r : index to i64
      %maximum = arith.maxsi %incoming, %best : i32
      %greater = arith.cmpi sgt, %incoming, %best : i32
      %equal = arith.cmpi eq, %incoming, %best : i32
      %earlier = arith.cmpi slt, %index, %index : i64
      %tie = arith.andi %equal, %earlier : i1
      %take = arith.ori %greater, %tie : i1
      %selected = arith.select %take, %index, %best_index : i64
      linalg.yield %maximum, %selected : i32, i64
    } -> (tensor<8xi32>, tensor<8xi64>)
    return %result#0, %result#1 : tensor<8xi32>, tensor<8xi64>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{could not tile reduction}}
    %init_value, %init_index, %split, %merge, %loop = transform.structured.tile_reduction_using_forall %op
      by num_threads = [4, 0] tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  func.func @reject_different_maps(%input: tensor<1025x8xi32>)
      -> (tensor<8xi32>, tensor<1x8xi64>) {
    %min = arith.constant -2147483648 : i32
    %last = arith.constant 9223372036854775807 : i64
    %empty_values = tensor.empty() : tensor<8xi32>
    %values = linalg.fill ins(%min : i32)
        outs(%empty_values : tensor<8xi32>) -> tensor<8xi32>
    %empty_indices = tensor.empty() : tensor<1x8xi64>
    %indices = linalg.fill ins(%last : i64)
        outs(%empty_indices : tensor<1x8xi64>) -> tensor<1x8xi64>
    // expected-error @below {{Failed to anaysis the reduction operation.}}
    %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(r, c) -> (r, c)>,
        affine_map<(r, c) -> (c)>,
        affine_map<(r, c) -> (0, c)>
      ],
      iterator_types = ["reduction", "parallel"]
    } ins(%input : tensor<1025x8xi32>)
      outs(%values, %indices : tensor<8xi32>, tensor<1x8xi64>) {
    ^bb0(%incoming: i32, %best: i32, %best_index: i64):
      %r = linalg.index 0 : index
      %index = arith.index_cast %r : index to i64
      %maximum = arith.maxsi %incoming, %best : i32
      %greater = arith.cmpi sgt, %incoming, %best : i32
      %equal = arith.cmpi eq, %incoming, %best : i32
      %earlier = arith.cmpi slt, %index, %best_index : i64
      %tie = arith.andi %equal, %earlier : i1
      %take = arith.ori %greater, %tie : i1
      %selected = arith.select %take, %index, %best_index : i64
      linalg.yield %maximum, %selected : i32, i64
    } -> (tensor<8xi32>, tensor<1x8xi64>)
    return %result#0, %result#1 : tensor<8xi32>, tensor<1x8xi64>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{could not tile reduction}}
    %init_value, %init_index, %split, %merge, %loop = transform.structured.tile_reduction_using_forall %op
      by num_threads = [4, 0] tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  func.func private @side_effect(i32)
  func.func @reject_side_effects(%input: tensor<1025x8xi32>)
      -> (tensor<8xi32>, tensor<8xi64>) {
    %min = arith.constant -2147483648 : i32
    %last = arith.constant 9223372036854775807 : i64
    %empty_values = tensor.empty() : tensor<8xi32>
    %values = linalg.fill ins(%min : i32)
        outs(%empty_values : tensor<8xi32>) -> tensor<8xi32>
    %empty_indices = tensor.empty() : tensor<8xi64>
    %indices = linalg.fill ins(%last : i64)
        outs(%empty_indices : tensor<8xi64>) -> tensor<8xi64>
    // expected-error @below {{Failed to anaysis the reduction operation.}}
    %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(r, c) -> (r, c)>,
        affine_map<(r, c) -> (c)>,
        affine_map<(r, c) -> (c)>
      ],
      iterator_types = ["reduction", "parallel"]
    } ins(%input : tensor<1025x8xi32>)
      outs(%values, %indices : tensor<8xi32>, tensor<8xi64>) {
    ^bb0(%incoming: i32, %best: i32, %best_index: i64):
      func.call @side_effect(%incoming) : (i32) -> ()
      %r = linalg.index 0 : index
      %index = arith.index_cast %r : index to i64
      %maximum = arith.maxsi %incoming, %best : i32
      %greater = arith.cmpi sgt, %incoming, %best : i32
      %equal = arith.cmpi eq, %incoming, %best : i32
      %earlier = arith.cmpi slt, %index, %best_index : i64
      %tie = arith.andi %equal, %earlier : i1
      %take = arith.ori %greater, %tie : i1
      %selected = arith.select %take, %index, %best_index : i64
      linalg.yield %maximum, %selected : i32, i64
    } -> (tensor<8xi32>, tensor<8xi64>)
    return %result#0, %result#1 : tensor<8xi32>, tensor<8xi64>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{could not tile reduction}}
    %init_value, %init_index, %split, %merge, %loop = transform.structured.tile_reduction_using_forall %op
      by num_threads = [4, 0] tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  func.func @reject_float(%input: tensor<1025x8xf32>)
      -> (tensor<8xf32>, tensor<8xi64>) {
    %min = arith.constant 0xFF800000 : f32
    %last = arith.constant 9223372036854775807 : i64
    %empty_values = tensor.empty() : tensor<8xf32>
    %values = linalg.fill ins(%min : f32)
        outs(%empty_values : tensor<8xf32>) -> tensor<8xf32>
    %empty_indices = tensor.empty() : tensor<8xi64>
    %indices = linalg.fill ins(%last : i64)
        outs(%empty_indices : tensor<8xi64>) -> tensor<8xi64>
    // expected-error @below {{Failed to anaysis the reduction operation.}}
    %result:2 = linalg.generic {
      indexing_maps = [
        affine_map<(r, c) -> (r, c)>,
        affine_map<(r, c) -> (c)>,
        affine_map<(r, c) -> (c)>
      ],
      iterator_types = ["reduction", "parallel"]
    } ins(%input : tensor<1025x8xf32>)
      outs(%values, %indices : tensor<8xf32>, tensor<8xi64>) {
    ^bb0(%incoming: f32, %best: f32, %best_index: i64):
      %r = linalg.index 0 : index
      %index = arith.index_cast %r : index to i64
      %maximum = arith.maxnumf %incoming, %best : f32
      %greater = arith.cmpf ogt, %incoming, %best : f32
      %equal = arith.cmpf oeq, %incoming, %best : f32
      %earlier = arith.cmpi slt, %index, %best_index : i64
      %tie = arith.andi %equal, %earlier : i1
      %take = arith.ori %greater, %tie : i1
      %selected = arith.select %take, %index, %best_index : i64
      linalg.yield %maximum, %selected : f32, i64
    } -> (tensor<8xf32>, tensor<8xi64>)
    return %result#0, %result#1 : tensor<8xf32>, tensor<8xi64>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{could not tile reduction}}
    %init_value, %init_index, %split, %merge, %loop = transform.structured.tile_reduction_using_forall %op
      by num_threads = [4, 0] tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
