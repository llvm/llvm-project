// RUN: mlir-opt %s --transform-interpreter --split-input-file -canonicalize -cse -verify-diagnostics | FileCheck %s

// The numerator is used by both the reduction and the final division. Fusing
// it before the reduction used to leave the full numerator outside the loop
// in addition to computing its tiles inside the loop. CSE cannot merge those
// differently shaped computations.
// CHECK-LABEL: func.func @softmax_producer_first
// CHECK-NOT: math.exp
// CHECK: scf.forall
// CHECK: %[[EXP:.+]] = linalg.generic
// CHECK: math.exp
// CHECK: %[[SUM:.+]] = linalg.generic
// CHECK-SAME: ins(%[[EXP]] : tensor<1x128xf32>)
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[EXP]], %[[SUM]] : tensor<1x128xf32>, tensor<1xf32>)
// CHECK-NOT: math.exp
// CHECK: return
module attributes {transform.with_named_sequence} {
  func.func @softmax_producer_first(%input: tensor<8x128xf32>, %output: tensor<8x128xf32>)
      -> tensor<8x128xf32> {
    %result = linalg.softmax dimension(1)
        ins(%input : tensor<8x128xf32>) outs(%output : tensor<8x128xf32>)
        -> tensor<8x128xf32>
    return %result : tensor<8x128xf32>
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %softmax = transform.structured.match ops{["linalg.softmax"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %decomposed = transform.structured.decompose_interface %softmax
        : (!transform.any_op) -> !transform.any_op
    %generics = transform.structured.match ops{["linalg.generic"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %parts:4 = transform.split_handle %generics : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %tiled, %container = transform.structured.tile_using_forall %parts#3 tile_sizes [1, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %producers = transform.merge_handles %parts#1, %parts#2 : !transform.any_op
    %fused, %new_container = transform.structured.fuse_into_containing_op
        %producers into %container : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Reversing the handle order must expose the same sharing opportunity.
// CHECK-LABEL: func.func @softmax_consumer_first
// CHECK-NOT: math.exp
// CHECK: scf.forall
// CHECK: %[[EXP:.+]] = linalg.generic
// CHECK: math.exp
// CHECK: %[[SUM:.+]] = linalg.generic
// CHECK-SAME: ins(%[[EXP]] : tensor<1x128xf32>)
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[EXP]], %[[SUM]] : tensor<1x128xf32>, tensor<1xf32>)
// CHECK-NOT: math.exp
// CHECK: return
module attributes {transform.with_named_sequence} {
  func.func @softmax_consumer_first(%input: tensor<8x128xf32>, %output: tensor<8x128xf32>)
      -> tensor<8x128xf32> {
    %result = linalg.softmax dimension(1)
        ins(%input : tensor<8x128xf32>) outs(%output : tensor<8x128xf32>)
        -> tensor<8x128xf32>
    return %result : tensor<8x128xf32>
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %softmax = transform.structured.match ops{["linalg.softmax"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %decomposed = transform.structured.decompose_interface %softmax
        : (!transform.any_op) -> !transform.any_op
    %generics = transform.structured.match ops{["linalg.generic"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %parts:4 = transform.split_handle %generics : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %tiled, %container = transform.structured.tile_using_forall %parts#3 tile_sizes [1, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %producers = transform.merge_handles %parts#2, %parts#1 : !transform.any_op
    %fused, %new_container = transform.structured.fuse_into_containing_op
        %producers into %container : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// The same ordering applies to an ordinary producer chain with a skip edge.
// Repeated producer handles must still be deduplicated.
// CHECK-LABEL: func.func @skip_connection
// CHECK-NOT: linalg.elementwise
// CHECK: scf.forall
// CHECK: %[[EXP:.+]] = linalg.elementwise <exp>
// CHECK: %[[ABS:.+]] = linalg.elementwise <abs> {producer} ins(%[[EXP]]
// CHECK: %[[NEG:.+]] = linalg.elementwise <negf> {producer} ins(%[[ABS]]
// CHECK: linalg.elementwise <add> {consumer} ins(%[[EXP]], %[[NEG]]
// CHECK: return
module attributes {transform.with_named_sequence} {
  func.func @skip_connection(%input: tensor<16xf32>, %output: tensor<16xf32>)
      -> tensor<16xf32> {
    %p = linalg.elementwise <exp> {producer} ins(%input : tensor<16xf32>)
        outs(%output : tensor<16xf32>) -> tensor<16xf32>
    %q = linalg.elementwise <abs> {producer} ins(%p : tensor<16xf32>)
        outs(%output : tensor<16xf32>) -> tensor<16xf32>
    %r = linalg.elementwise <negf> {producer} ins(%q : tensor<16xf32>)
        outs(%output : tensor<16xf32>) -> tensor<16xf32>
    %result = linalg.elementwise <add> {consumer} ins(%p, %r : tensor<16xf32>, tensor<16xf32>)
        outs(%output : tensor<16xf32>) -> tensor<16xf32>
    return %result : tensor<16xf32>
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %producers = transform.structured.match attributes{producer} in %root
        : (!transform.any_op) -> !transform.any_op
    %consumer = transform.structured.match attributes{consumer} in %root
        : (!transform.any_op) -> !transform.any_op
    %tiled, %container = transform.structured.tile_using_forall %consumer tile_sizes [4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %repeated = transform.merge_handles %producers, %producers : !transform.any_op
    %fused, %new_container = transform.structured.fuse_into_containing_op
        %repeated into %container : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// An independently used result of a multi-output producer must be preserved.
// Its first result can still be shared by the selected consumer and the tile.
// CHECK-LABEL: func.func @multi_output_with_external_use
// CHECK: %[[PRODUCER:.+]]:2 = linalg.generic
// CHECK: math.exp
// CHECK: arith.mulf
// CHECK: %[[RESULT:.+]] = scf.forall
// CHECK: %[[EXP:.+]]:2 = linalg.generic
// CHECK: math.exp
// CHECK: %[[ABS:.+]] = linalg.elementwise <abs> {producer} ins(%[[EXP]]#0
// CHECK: linalg.elementwise <add> {consumer} ins(%[[EXP]]#0, %[[ABS]]
// CHECK: return %[[RESULT]], %[[PRODUCER]]#1
module attributes {transform.with_named_sequence} {
  func.func @multi_output_with_external_use(
      %input: tensor<16xf32>, %output: tensor<16xf32>, %side: tensor<16xf32>)
      -> (tensor<16xf32>, tensor<16xf32>) {
    %p:2 = linalg.generic {
        producer,
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%input : tensor<16xf32>)
        outs(%output, %side : tensor<16xf32>, tensor<16xf32>) {
      ^bb0(%in: f32, %out: f32, %other: f32):
        %exp = math.exp %in : f32
        %square = arith.mulf %in, %in : f32
        linalg.yield %exp, %square : f32, f32
    } -> (tensor<16xf32>, tensor<16xf32>)
    %q = linalg.elementwise <abs> {producer} ins(%p#0 : tensor<16xf32>)
        outs(%output : tensor<16xf32>) -> tensor<16xf32>
    %result = linalg.elementwise <add> {consumer} ins(%p#0, %q : tensor<16xf32>, tensor<16xf32>)
        outs(%output : tensor<16xf32>) -> tensor<16xf32>
    return %result, %p#1 : tensor<16xf32>, tensor<16xf32>
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %producers = transform.structured.match attributes{producer} in %root
        : (!transform.any_op) -> !transform.any_op
    %consumer = transform.structured.match attributes{consumer} in %root
        : (!transform.any_op) -> !transform.any_op
    %tiled, %container = transform.structured.tile_using_forall %consumer tile_sizes [4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused, %new_container = transform.structured.fuse_into_containing_op
        %producers into %container : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// A sequential container can request different tiles of a shared producer.
// Ordering must expose both uses without treating the distinct tiles as equal.
// CHECK-LABEL: func.func @different_tiles_in_for
// CHECK-SAME: %[[INPUT:.+]]: tensor<16xf32>
// CHECK-NOT: linalg.elementwise
// CHECK: scf.for %[[I:[a-zA-Z0-9_]+]] =
// CHECK: %[[J:.+]] = arith.addi %[[I]], %{{.+}} : index
// CHECK: %[[X0:.+]] = tensor.extract_slice %[[INPUT]][%[[I]]] [1] [1]
// CHECK: %[[E0:.+]] = linalg.elementwise <exp> {producer} ins(%[[X0]]
// CHECK: %[[X1:.+]] = tensor.extract_slice %[[INPUT]][%[[J]]] [1] [1]
// CHECK: %[[E1:.+]] = linalg.elementwise <exp> {producer} ins(%[[X1]]
// CHECK: %[[ABS:.+]] = linalg.elementwise <abs> {producer} ins(%[[E1]]
// CHECK: linalg.elementwise <add> ins(%[[E0]], %[[ABS]]
// CHECK: return
module attributes {transform.with_named_sequence} {
  func.func @different_tiles_in_for(%input: tensor<16xf32>, %output: tensor<8xf32>)
      -> tensor<8xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %empty = tensor.empty() : tensor<16xf32>
    %p = linalg.elementwise <exp> {producer} ins(%input : tensor<16xf32>)
        outs(%empty : tensor<16xf32>) -> tensor<16xf32>
    %q = linalg.elementwise <abs> {producer} ins(%p : tensor<16xf32>)
        outs(%empty : tensor<16xf32>) -> tensor<16xf32>
    %result = scf.for %i = %c0 to %c8 step %c1 iter_args(%out = %output)
        -> tensor<8xf32> {
      %j = arith.addi %i, %c8 : index
      %left = tensor.extract_slice %p[%i] [1] [1] : tensor<16xf32> to tensor<1xf32>
      %right = tensor.extract_slice %q[%j] [1] [1] : tensor<16xf32> to tensor<1xf32>
      %dest = tensor.extract_slice %out[%i] [1] [1] : tensor<8xf32> to tensor<1xf32>
      %sum = linalg.elementwise <add> ins(%left, %right : tensor<1xf32>, tensor<1xf32>)
          outs(%dest : tensor<1xf32>) -> tensor<1xf32>
      %inserted = tensor.insert_slice %sum into %out[%i] [1] [1]
          : tensor<1xf32> into tensor<8xf32>
      scf.yield %inserted : tensor<8xf32>
    }
    return %result : tensor<8xf32>
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %producers = transform.structured.match attributes{producer} in %root
        : (!transform.any_op) -> !transform.any_op
    %container = transform.structured.match ops{["scf.for"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %fused, %new_container = transform.structured.fuse_into_containing_op
        %producers into %container : (!transform.any_op, !transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// A top-level payload operation has no parent region. Diagnose it instead of
// trying to find a common ancestor region with another selected producer.
module attributes {transform.with_named_sequence} {
  // NO-CHECK-LABEL-ON-EXPECTED-ERROR
  func.func @top_level_producer(%output: tensor<4xf32>) -> tensor<4xf32> {
    %zero = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%zero : f32) outs(%output : tensor<4xf32>)
        -> tensor<4xf32>
    // expected-note @below {{containing op}}
    %result = scf.forall (%i) in (4) shared_outs(%out = %output) -> (tensor<4xf32>) {
      %slice = tensor.extract_slice %fill[%i] [1] [1]
          : tensor<4xf32> to tensor<1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %slice into %out[%i] [1] [1]
            : tensor<1xf32> into tensor<4xf32>
      }
    }
    return %result : tensor<4xf32>
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %fill = transform.structured.match ops{["linalg.fill"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %container = transform.structured.match ops{["scf.forall"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %producers = transform.merge_handles %root, %fill : !transform.any_op
    // expected-error @below {{cannot fuse a top-level operation}}
    %fused, %new_container = transform.structured.fuse_into_containing_op %producers into %container
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
