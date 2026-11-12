// RUN: mlir-opt %s --transform-interpreter --split-input-file -canonicalize -cse -verify-diagnostics | FileCheck %s

// TODO: Process selected consumers before shared producers to avoid leaving
// both a full producer and tiled producer computations after fusion.
// CHECK-LABEL: func.func @softmax_producer_first
// CHECK: %[[FULL:.+]] = linalg.generic {{.*}}iterator_types = ["parallel", "parallel"]}
// CHECK: math.exp
// CHECK: scf.forall
// CHECK: %[[EXP:.+]] = linalg.generic
// CHECK: math.exp
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[FULL]]
// CHECK: %[[SUM:.+]] = linalg.generic
// CHECK-SAME: ins(%[[SLICE]] : tensor<1x128xf32>)
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[EXP]], %[[SUM]] : tensor<1x128xf32>, tensor<1xf32>)
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

// TODO: Fuse the skip-edge producer after its selected consumers.
// Repeated handles must still be deduplicated.
// CHECK-LABEL: func.func @skip_connection
// CHECK: %[[FULL:.+]] = linalg.elementwise <exp>
// CHECK: scf.forall
// CHECK: %[[EXP:.+]] = linalg.elementwise <exp>
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[FULL]]
// CHECK: %[[ABS:.+]] = linalg.elementwise <abs> {producer} ins(%[[SLICE]]
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

// Preserve an externally used result of a multi-output producer.
// TODO: Share the tiled first result with the selected consumer.
// CHECK-LABEL: func.func @multi_output_with_external_use
// CHECK: %[[FULL:.+]]:2 = linalg.generic
// CHECK: math.exp
// CHECK: arith.mulf
// CHECK: %[[RESULT:.+]] = scf.forall
// CHECK: %[[EXP:.+]]:2 = linalg.generic
// CHECK: math.exp
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[FULL]]#0
// CHECK: %[[ABS:.+]] = linalg.elementwise <abs> {producer} ins(%[[SLICE]]
// CHECK: linalg.elementwise <add> {consumer} ins(%[[EXP]]#0, %[[ABS]]
// CHECK: return %[[RESULT]], %[[FULL]]#1

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

// Different tile domains must remain distinct.
// TODO: Fuse both requested producer tiles instead of leaving the full producer.
// CHECK-LABEL: func.func @different_tiles_in_for
// CHECK-SAME: %[[INPUT:.+]]: tensor<16xf32>
// CHECK: %[[FULL:.+]] = linalg.elementwise <exp>
// CHECK: scf.for %[[I:[a-zA-Z0-9_]+]] =
// CHECK: %[[J:.+]] = arith.addi %[[I]], %{{.+}} : index
// CHECK: %[[X0:.+]] = tensor.extract_slice %[[INPUT]][%[[I]]] [1] [1]
// CHECK: %[[E0:.+]] = linalg.elementwise <exp> {producer} ins(%[[X0]]
// CHECK: %[[X1:.+]] = tensor.extract_slice %[[FULL]][%[[J]]] [1] [1]
// CHECK: %[[ABS:.+]] = linalg.elementwise <abs> {producer} ins(%[[X1]]
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
