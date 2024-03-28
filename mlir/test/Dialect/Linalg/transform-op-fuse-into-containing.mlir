// RUN: mlir-opt --transform-interpreter --split-input-file %s -verify-diagnostics | FileCheck %s

#map0 = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0, s1)>

module {
  // CHECK-LABEL: func.func @fuse_tileable_op
  //  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
  //  CHECK-SAME:   %[[IN:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT:[0-9a-z]+]]: tensor<?xf32>
  func.func @fuse_tileable_op(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 4.200000e+01 : f32
    %c0 = arith.constant 0 : index
    %0 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<?xf32>) -> tensor<?xf32>
    %d0 = tensor.dim %arg1, %c0 : tensor<?xf32>
    %1 = affine.apply #map0()[%d0, %arg0]

    // CHECK: scf.forall {{.*}} {
    %2 = scf.forall (%arg3) in (%1) shared_outs(%o = %arg2) -> (tensor<?xf32>) {
      %3 = affine.apply #map1(%arg3)[%arg0]
      %4 = affine.min #map2(%arg3)[%d0, %arg0]
      %5 = tensor.extract_slice %o[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: %[[T0:.*]] = tensor.extract_slice %[[IN]][%{{.*}}] [%{{.*}}] [{{.*}}]
      // CHECK: %[[T1:.*]] = linalg.fill {{.*}} outs(%[[T0]]
      %6 = tensor.extract_slice %0[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: %[[T2:.*]] = linalg.elemwise_unary ins(%[[T1]]
      %7 = linalg.elemwise_unary ins(%6 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %7 into %o[%3] [%4] [1] : tensor<?xf32> into tensor<?xf32>
      }
    }
    // CHECK: }
    func.return %2 : tensor<?xf32>
  }

  // Check no failure when nothing happens.
  func.func @dummy1() { return }
  func.func @dummy2() { return }
  func.func @dummy3() { return }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.fill">
      %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

      // linalg.fill is tileable. The op is tiled and fused.
      transform.structured.fuse_into_containing_op %0 into %1
        : (!transform.op<"linalg.fill">, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.any_op)
        transform.yield
    }
  }
}

// -----

#map0 = affine_map<()[s0] -> (64 ceildiv s0)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0] -> (-(d0 * s0) + 64, s0)>

module {
  // CHECK-LABEL: func.func @fuse_untileable_op
  //  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
  //  CHECK-SAME:   %[[IN:[0-9a-z]+]]: tensor<64xf32>
  //  CHECK-SAME:   %[[OUT:[0-9a-z]+]]: tensor<64xf32>
  func.func @fuse_untileable_op(%arg0: index, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> {
    %0 = tensor.empty(%arg0) : tensor<?xf32>
    %1 = affine.apply #map0()[%arg0]

    // CHECK: scf.forall {{.*}} {
    %2 = scf.forall (%arg3) in (%1) shared_outs(%o = %arg2) -> (tensor<64xf32>) {
      // CHECK: %[[INIT_TENSOR:.*]] = tensor.empty
      %3 = affine.apply #map1(%arg3)[%arg0]
      %4 = affine.min #map2(%arg3)[%arg0]
      %5 = tensor.extract_slice %o[%3] [%4] [1] : tensor<64xf32> to tensor<?xf32>

      // CHECK: %[[T2:.*]] = linalg.elemwise_unary ins(%[[INIT_TENSOR]]
      %7 = linalg.elemwise_unary ins(%0 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %7 into %o[%3] [%4] [1] : tensor<?xf32> into tensor<64xf32>
      }
    }
    // CHECK: }

    func.return %2 : tensor<64xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.empty"]} in %arg1 : (!transform.any_op) -> !transform.op<"tensor.empty">
      %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

      // tensor.empty is not tileable. The op is cloned and fused.
      transform.structured.fuse_into_containing_op %0 into %1
        : (!transform.op<"tensor.empty">, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.any_op)
        transform.yield
    }
  }
}

// -----

module {
  func.func @foo(%0: tensor<f32>) -> tensor<f32> {
    return %0: tensor<f32>
  }

  // CHECK-LABEL: func.func @fuse_tileable_op_rank_reducing
  //  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
  //  CHECK-SAME:   %[[IN:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT:[0-9a-z]+]]: tensor<?xf32>
  func.func @fuse_tileable_op_rank_reducing(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 4.200000e+01 : f32
    %c0 = arith.constant 0 : index
    %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
    %d0 = tensor.dim %arg1, %c0 : tensor<?xf32>

    // CHECK: scf.forall {{.*}} -> (tensor<?xf32>) {
    %2 = scf.forall (%arg3) in (%d0) shared_outs(%o = %0) -> (tensor<?xf32>) {
      %5 = tensor.extract_slice %o[%arg3] [1] [1] : tensor<?xf32> to tensor<f32>

      // CHECK: tensor.extract_slice %{{.*}}[%{{.*}}] [1] [1] : tensor<?xf32> to tensor<1xf32>
      // CHECK: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<1xf32>) -> tensor<1xf32>
      // CHECK: tensor.extract_slice %{{.*}}[0] [1] [1] : tensor<1xf32> to tensor<f32>
      // CHECK: func.call @foo(%{{.*}}) : (tensor<f32>) -> tensor<f32>
      %7 = func.call @foo(%5) : (tensor<f32>) -> tensor<f32>

      scf.forall.in_parallel {
      // CHECK: tensor.parallel_insert_slice %{{.*}} into %{{.*}}[%{{.*}}] [1] [1] : tensor<f32> into tensor<?xf32>
        tensor.parallel_insert_slice %7 into %o[%arg3] [1] [1] : tensor<f32> into tensor<?xf32>
      }
    }
    // CHECK: }
    func.return %2 : tensor<?xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.fill">
      %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

      // linalg.fill is tileable. The op is tiled and fused.
      transform.structured.fuse_into_containing_op %0 into %1
        : (!transform.op<"linalg.fill">, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.any_op)
        transform.yield
    }
  }
}

// -----

#map0 = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0, s1)>

module {
  // CHECK-LABEL: func.func @fuse_tileable_op_through_bbarg
  //  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
  //  CHECK-SAME:   %[[IN:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT:[0-9a-z]+]]: tensor<?xf32>
  func.func @fuse_tileable_op_through_bbarg(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 4.200000e+01 : f32
    %c0 = arith.constant 0 : index
    %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
    %d0 = tensor.dim %arg1, %c0 : tensor<?xf32>
    %1 = affine.apply #map0()[%d0, %arg0]

    // CHECK: scf.forall {{.*}} shared_outs(%[[BBARGOUT:.*]] = %[[OUT]]) -> (tensor<?xf32>) {
    %2 = scf.forall (%arg3) in (%1) shared_outs(%o = %0) -> (tensor<?xf32>) {
      %3 = affine.apply #map1(%arg3)[%arg0]
      %4 = affine.min #map2(%arg3)[%d0, %arg0]
      %5 = tensor.extract_slice %o[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: %[[T0:.*]] = tensor.extract_slice %[[BBARGOUT]][%{{.*}}] [%{{.*}}] [{{.*}}]
      // CHECK: %[[T1:.*]] = linalg.fill {{.*}} outs(%[[T0]]
      %6 = tensor.extract_slice %arg1[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: %[[T2:.*]] = linalg.elemwise_unary {{.*}} outs(%[[T1]]
      %7 = linalg.elemwise_unary ins(%6 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %7 into %o[%3] [%4] [1] : tensor<?xf32> into tensor<?xf32>
      }
    }
    // CHECK: }
    func.return %2 : tensor<?xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op

      // linalg.fill is tileable. The op is tiled and fused.
      transform.structured.fuse_into_containing_op %0 into %1
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.yield
    }
  }
}

// -----

#map0 = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0, s1)>

module {
  // CHECK-LABEL: func.func @fuse_tileable_multi_output_op
  //  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
  //  CHECK-SAME:   %[[IN:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT_1:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT_2:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT_3:[0-9a-z]+]]: tensor<?xf32>
  func.func @fuse_tileable_multi_output_op(%idx: index, %in: tensor<?xf32>, %out_1: tensor<?xf32>, %out_2: tensor<?xf32>, %out_3: tensor<?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 4.200000e+01 : f32
    %c0 = arith.constant 0 : index

    %0:2 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%in : tensor<?xf32>) outs(%out_1, %out_3 : tensor<?xf32>, tensor<?xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %d = arith.addf %a, %b : f32
        %e = arith.addf %d, %c : f32
        linalg.yield %d, %e : f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>)
    %d0 = tensor.dim %out_1, %c0 : tensor<?xf32>

    %1 = affine.apply #map0()[%d0, %idx]

    // CHECK: scf.forall {{.*}} {
    %2 = scf.forall (%i) in (%1) shared_outs(%o = %out_2) -> (tensor<?xf32>) {
      %3 = affine.apply #map1(%i)[%idx]
      %4 = affine.min #map2(%i)[%d0, %idx]
      %5 = tensor.extract_slice %o[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: %[[T0:.*]] = tensor.extract_slice %[[IN]][%{{.*}}] [%{{.*}}] [{{.*}}]
      // CHECK: %[[T1:.*]]:2 = linalg.generic {{.*}} ins(%[[T0]]
      %6 = tensor.extract_slice %0#0[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: %[[T2:.*]] = linalg.elemwise_unary ins(%[[T1]]#0
      %7 = linalg.elemwise_unary ins(%6 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %7 into %o[%3] [%4] [1] : tensor<?xf32> into tensor<?xf32>
      }
    }
    // CHECK: }
    func.return %2 : tensor<?xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.generic">
      %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

      // linalg.generic is tileable. The op is tiled and fused.
      transform.structured.fuse_into_containing_op %0 into %1
        : (!transform.op<"linalg.generic">, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.any_op)
        transform.yield
    }
  }
}

// -----

module {
  // CHECK-LABEL: func.func @fuse_repeated
  func.func @fuse_repeated(%fill: tensor<2xf32>, %output: tensor<2xf32>) -> tensor<2xf32> {
    %c0 = arith.constant 0.0 : f32
    %0 = linalg.fill ins(%c0 : f32) outs(%fill : tensor<2xf32>) -> tensor<2xf32>

    // CHECK: scf.forall
    %1 = scf.forall (%i) in (2) shared_outs(%arg1 = %output) -> (tensor<2xf32>) {
      %2 = tensor.extract_slice %0[%i][1][1] : tensor<2xf32> to tensor<1xf32>
      %3 = tensor.extract_slice %arg1[%i][1][1] : tensor<2xf32> to tensor<1xf32>
      // CHECK: %[[FUSED:.+]] = linalg.fill
      // CHECK: elemwise_unary ins(%[[FUSED]]
      %4 = linalg.elemwise_unary ins(%2 : tensor<1xf32>) outs(%3 : tensor<1xf32>) -> tensor<1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %4 into %arg1[%i][1][1] : tensor<1xf32> into tensor<2xf32>
      }
    }

    return %1 : tensor<2xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op

      // Create a new handle that points to `linalg.fill` twice.
      %2 = transform.merge_handles %0, %0 : !transform.any_op

      // It shouldn't be a problem to fuse this handle.
      transform.structured.fuse_into_containing_op %2 into %1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----

#map0 = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0, s1)>

module {
  // CHECK-LABEL: func.func @fuse_tileable_multi_output_op_multi_use
  //  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
  //  CHECK-SAME:   %[[IN:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT_1:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT_2:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT_3:[0-9a-z]+]]: tensor<?xf32>
  func.func @fuse_tileable_multi_output_op_multi_use(%idx: index, %in: tensor<?xf32>, %out_1: tensor<?xf32>, %out_2: tensor<?xf32>, %out_3: tensor<?xf32>)
    -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
    %cst = arith.constant 4.200000e+01 : f32
    %c0 = arith.constant 0 : index

    // CHECK: %[[G0:.*]]:2 = linalg.generic
    %0:2 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%in : tensor<?xf32>) outs(%out_1, %out_3 : tensor<?xf32>, tensor<?xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %d = arith.addf %a, %b : f32
        %e = arith.addf %d, %c : f32
        linalg.yield %d, %e : f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>)
    %d0 = tensor.dim %out_1, %c0 : tensor<?xf32>

    %1 = affine.apply #map0()[%d0, %idx]

    // CHECK: %[[R0:.*]]:2 = scf.forall (%[[ARG5:.*]]) in (%{{.*}}) shared_outs(%[[ARG6:.*]] = %[[OUT_2]], %[[ARG7:.*]] = %[[OUT_1]])
    // CHECK-SAME: -> (tensor<?xf32>, tensor<?xf32>) {
    // expected-remark @below{{new containing op}}
    %2 = scf.forall (%i) in (%1) shared_outs(%o = %out_2) -> (tensor<?xf32>) {
      // CHECK: %[[I0:.*]] = affine.apply {{.*}}
      %3 = affine.apply #map1(%i)[%idx]
      // CHECK: %[[I1:.*]] = affine.min {{.*}}
      %4 = affine.min #map2(%i)[%d0, %idx]
      %5 = tensor.extract_slice %o[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: %[[T1:.*]]:2 = linalg.generic {{.*}}
      %6 = tensor.extract_slice %0#0[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      %7 = linalg.elemwise_unary ins(%6 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      scf.forall.in_parallel {
        // CHECK: tensor.parallel_insert_slice %[[T1]]#0 into %[[ARG7]][%[[I0]]] [%[[I1]]] [1] : tensor<?xf32> into tensor<?xf32>
        tensor.parallel_insert_slice %7 into %o[%3] [%4] [1] : tensor<?xf32> into tensor<?xf32>
      }
    }
    // CHECK: return %[[R0]]#0, %[[R0]]#1, %[[G0]]#1
    func.return %2, %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
    // CHECK: }
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.generic">
      %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

      // linalg.generic is tileable. The op is tiled and fused.
      %fused, %containing = transform.structured.fuse_into_containing_op %0 into %1
        : (!transform.op<"linalg.generic">, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.any_op)
      transform.debug.emit_remark_at %containing, "new containing op" : !transform.any_op
      transform.yield
    }
  }
}

// -----

#map0 = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0, s1)>

module {
  // CHECK-LABEL: func.func @fuse_tileable_mixed_dominating_uses
  //  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
  //  CHECK-SAME:   %[[IN:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT_1:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT_2:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT_3:[0-9a-z]+]]: tensor<?xf32>
  func.func @fuse_tileable_mixed_dominating_uses(%idx: index, %in: tensor<?xf32>, %out_1: tensor<?xf32>, %out_2: tensor<?xf32>, %out_3: tensor<?xf32>)
    -> (tensor<?xf32>, tensor<?xf32>) {
    %cst = arith.constant 4.200000e+01 : f32
    %c0 = arith.constant 0 : index

    // CHECK: %[[G0:.*]] = linalg.generic
    %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%in : tensor<?xf32>) outs(%out_1 : tensor<?xf32>) {
      ^bb0(%a: f32, %b: f32):
        %d = arith.addf %a, %b : f32
        linalg.yield %d : f32
    } -> tensor<?xf32>
    // CHECK: %[[D0:.*]] = tensor.dim %[[G0]]
    %d0 = tensor.dim %0, %c0 : tensor<?xf32>

    %1 = affine.apply #map0()[%d0, %idx]

    // CHECK: %[[R0:.*]]:2 = scf.forall (%[[ARG5:.*]]) in (%{{.*}}) shared_outs(%[[ARG6:.*]] = %[[OUT_2]], %[[ARG7:.*]] = %[[OUT_1]])
    // CHECK-SAME: -> (tensor<?xf32>, tensor<?xf32>) {
    %2 = scf.forall (%i) in (%1) shared_outs(%o = %out_2) -> (tensor<?xf32>) {
      // CHECK: %[[I0:.*]] = affine.apply {{.*}}
      %3 = affine.apply #map1(%i)[%idx]
      // CHECK: %[[I1:.*]] = affine.min {{.*}}
      %4 = affine.min #map2(%i)[%d0, %idx]
      %5 = tensor.extract_slice %o[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: %[[T1:.*]] = linalg.generic {{.*}}
      %6 = tensor.extract_slice %0[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      %7 = linalg.elemwise_unary ins(%6 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      scf.forall.in_parallel {
        // CHECK: tensor.parallel_insert_slice %[[T1]] into %[[ARG7]][%[[I0]]] [%[[I1]]] [1] : tensor<?xf32> into tensor<?xf32>
        tensor.parallel_insert_slice %7 into %o[%3] [%4] [1] : tensor<?xf32> into tensor<?xf32>
      }
    }
    // CHECK: return %[[R0]]#0, %[[R0]]#1
    func.return %2, %0 : tensor<?xf32>, tensor<?xf32>
    // CHECK: }
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.generic">
      %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

      // linalg.generic is tileable. The op is tiled and fused.
      transform.structured.fuse_into_containing_op %0 into %1
        : (!transform.op<"linalg.generic">, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.any_op)
        transform.yield
    }
  }
}

// -----

#map0 = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0, s1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0)>

module {
  // CHECK-LABEL: func.func @fuse_tileable_reductions
  //  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
  //  CHECK-SAME:   %[[IN:[0-9a-z]+]]: tensor<?x?xf32>
  //  CHECK-SAME:   %[[OUT_1:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT_2:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT_3:[0-9a-z]+]]: tensor<?xf32>
  func.func @fuse_tileable_reductions(%idx: index, %in: tensor<?x?xf32>, %out_1: tensor<?xf32>, %out_2: tensor<?xf32>, %out_3: tensor<?xf32>)
    -> (tensor<?xf32>, tensor<?xf32>) {
    %cst = arith.constant 4.200000e+01 : f32
    %c0 = arith.constant 0 : index

    %0 = linalg.generic {
      indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction"]
      } ins(%in : tensor<?x?xf32>) outs(%out_1 : tensor<?xf32>) {
        ^bb0(%a: f32, %b: f32):
          %d = arith.maximumf %a, %b : f32
          linalg.yield %d : f32
        } -> tensor<?xf32>
    %d0 = tensor.dim %out_1, %c0 : tensor<?xf32>

    %1 = affine.apply #map0()[%d0, %idx]

    // CHECK: %[[R0:.*]]:2 = scf.forall (%[[ARG5:.*]]) in (%{{.*}}) shared_outs(%[[ARG6:.*]] = %[[OUT_2]], %[[ARG7:.*]] = %[[OUT_1]])
    // CHECK-SAME: -> (tensor<?xf32>, tensor<?xf32>) {
    %2 = scf.forall (%i) in (%1) shared_outs(%o = %out_2) -> (tensor<?xf32>) {
      // CHECK: %[[I0:.*]] = affine.apply {{.*}}
      %3 = affine.apply #map1(%i)[%idx]
      // CHECK: %[[I1:.*]] = affine.min {{.*}}
      %4 = affine.min #map2(%i)[%d0, %idx]
      %5 = tensor.extract_slice %o[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: %[[T1:.*]] = linalg.generic {{.*}}
      %6 = tensor.extract_slice %0[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      %7 = linalg.elemwise_unary ins(%6 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      scf.forall.in_parallel {
        // CHECK: tensor.parallel_insert_slice %[[T1]] into %[[ARG7]][%[[I0]]] [%[[I1]]] [1] : tensor<?xf32> into tensor<?xf32>
        tensor.parallel_insert_slice %7 into %o[%3] [%4] [1] : tensor<?xf32> into tensor<?xf32>
      }
    }
    // CHECK: return %[[R0]]#0, %[[R0]]#1
    func.return %2, %0 : tensor<?xf32>, tensor<?xf32>
    // CHECK: }
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.generic">
      %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

      // linalg.generic is tileable. The op is tiled and fused.
      transform.structured.fuse_into_containing_op %0 into %1
        : (!transform.op<"linalg.generic">, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.any_op)
        transform.yield
    }
  }
}

// -----

#map0 = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0, s1)>
#map3 = affine_map<(d0) -> (d0)>

module {
  // CHECK-LABEL: func.func @fuse_tileable_using_new_handle
  //  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
  //  CHECK-SAME:   %[[IN:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT_1:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT_2:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT_3:[0-9a-z]+]]: tensor<?xf32>
  func.func @fuse_tileable_using_new_handle(%idx: index, %in: tensor<?xf32>, %out_1: tensor<?xf32>, %out_2: tensor<?xf32>, %out_3: tensor<?xf32>)
    -> (tensor<?xf32>, tensor<?xf32>) {
    %cst = arith.constant 4.200000e+01 : f32
    %c0 = arith.constant 0 : index

    %0 = linalg.generic {
      indexing_maps = [#map3, #map3], iterator_types = ["parallel"]
      } ins(%in : tensor<?xf32>) outs(%out_1 : tensor<?xf32>) {
        ^bb0(%a: f32, %b: f32):
          %d = arith.addf %a, %b : f32
          linalg.yield %d : f32
        } -> tensor<?xf32>

    %1 = linalg.generic {
      indexing_maps = [#map3, #map3], iterator_types = ["parallel"]
      } ins(%0 : tensor<?xf32>) outs(%out_1 : tensor<?xf32>) {
        ^bb0(%a: f32, %b: f32):
          %d = arith.mulf %a, %b : f32
          linalg.yield %d : f32
        } -> tensor<?xf32>
    %d0 = tensor.dim %out_1, %c0 : tensor<?xf32>

    %2 = affine.apply #map0()[%d0, %idx]

    // CHECK: %[[R0:.*]]:2 = scf.forall (%[[ARG5:.*]]) in (%{{.*}}) shared_outs(%[[ARG6:.*]] = %[[OUT_2]], %[[ARG7:.*]] = %[[OUT_1]])
    // CHECK-SAME: -> (tensor<?xf32>, tensor<?xf32>) {
    %3 = scf.forall (%i) in (%2) shared_outs(%o = %out_2) -> (tensor<?xf32>) {
      // CHECK: %[[I0:.*]] = affine.apply {{.*}}
      %4 = affine.apply #map1(%i)[%idx]
      // CHECK: %[[I1:.*]] = affine.min {{.*}}
      %5 = affine.min #map2(%i)[%d0, %idx]
      %6 = tensor.extract_slice %o[%4] [%5] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: %[[T1:.*]] = linalg.generic {{.*}}
      // CHECK: %[[T2:.*]] = linalg.generic {{.*}}
      %7 = tensor.extract_slice %1[%4] [%5] [1] : tensor<?xf32> to tensor<?xf32>

      %8 = linalg.elemwise_unary ins(%7 : tensor<?xf32>) outs(%6 : tensor<?xf32>) -> tensor<?xf32>
      scf.forall.in_parallel {
        // CHECK: tensor.parallel_insert_slice %[[T2]] into %[[ARG7]][%[[I0]]] [%[[I1]]] [1] : tensor<?xf32> into tensor<?xf32>
        tensor.parallel_insert_slice %8 into %o[%2] [%5] [1] : tensor<?xf32> into tensor<?xf32>
      }
    }
    // CHECK: return %[[R0]]#0, %[[R0]]#1
    func.return %3, %1 : tensor<?xf32>, tensor<?xf32>
    // CHECK: }
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.generic">
      %add, %reduce = transform.split_handle %0 : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.generic">, !transform.op<"linalg.generic">)
      %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

      %fused_ops, %new_forall = transform.structured.fuse_into_containing_op %reduce into %1
        : (!transform.op<"linalg.generic">, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.op<"scf.forall">)
      %fused_ops_2, %new_forall_2 = transform.structured.fuse_into_containing_op %add into %new_forall
        : (!transform.op<"linalg.generic">, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.op<"scf.forall">)
        transform.yield
    }
  }
}

// -----

#map = affine_map<()[s0] -> (64 ceildiv s0)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0] -> (-(d0 * s0) + 64, s0)>
module {
  // CHECK-LABEL: func.func @fuse_tileable_consumer
  // CHECK-SAME: %[[CHUNK_SIZE:[0-9a-z]+]]: index
  // CHECK-SAME: %[[IN:[0-9a-z]+]]: tensor<64xf32>
  // CHECK-SAME: %[[OUT:[0-9a-z]+]]: tensor<64xf32>
  func.func @fuse_tileable_consumer(%arg0: index, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> {
    // CHECK: %[[SLICE:.*]] = tensor.empty(%[[CHUNK_SIZE]]) : tensor<?xf32>
    %0 = tensor.empty(%arg0) : tensor<?xf32>
    %1 = affine.apply #map()[%arg0]
    // CHECK: %[[EMPTY0:[0-9a-z]+]] = tensor.empty() : tensor<64xf32>
    %2 = tensor.empty() : tensor<64xf32>
    // CHECK: %[[EMPTY1:[0-9a-z]+]] = tensor.empty() : tensor<64xf32>
    %3 = tensor.empty() : tensor<64xf32>
    // CHECK: %[[RES:[0-9a-z]+]]:2 = scf.forall {{.*}} shared_outs(%[[LOOP_ARG0:.*]] = %[[OUT]], %[[LOOP_ARG1:.*]] = %[[EMPTY1]]
    %4 = scf.forall (%arg3) in (%1) shared_outs(%arg4 = %arg2) -> (tensor<64xf32>) {
      %6 = affine.apply #map1(%arg3)[%arg0]
      %7 = affine.min #map2(%arg3)[%arg0]
      // CHECK: %[[T0:.*]] = tensor.extract_slice %[[LOOP_ARG0]][%{{.*}}] [%{{.*}}] [{{.*}}]
      %extracted_slice = tensor.extract_slice %arg4[%6] [%7] [1] : tensor<64xf32> to tensor<?xf32>
      // CHECK: %[[T1:[0-9a-z]+]] = linalg.elemwise_unary
      %8 = linalg.elemwise_unary ins(%0 : tensor<?xf32>) outs(%extracted_slice : tensor<?xf32>) -> tensor<?xf32>

      // CHECK: %[[T2:.*]] = tensor.extract_slice %[[EMPTY0]][%{{.*}}] [%{{.*}}] [{{.*}}]
      // CHECK: %[[T3:.*]] = tensor.extract_slice %[[LOOP_ARG1]][%{{.*}}] [%{{.*}}] [{{.*}}]
      // CHECK: %[[T4:.*]] = linalg.elemwise_binary {{.*}} ins(%[[T1]], %[[T2]] : {{.*}} outs(%[[T3]]

      scf.forall.in_parallel {
        // CHECK: tensor.parallel_insert_slice %[[T4]] into %[[LOOP_ARG1]]
        // CHECK: tensor.parallel_insert_slice %[[T1]] into %[[LOOP_ARG0]]
        tensor.parallel_insert_slice %8 into %arg4[%6] [%7] [1] : tensor<?xf32> into tensor<64xf32>
      }
    }
    // CHECK: %[[ORI_OUTPUT:.*]] = linalg.elemwise_binary
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%4, %2 : tensor<64xf32>, tensor<64xf32>) outs(%3 : tensor<64xf32>) -> tensor<64xf32>
    // CHECK: return %[[RES]]#1
    return %5 : tensor<64xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.op<"scf.forall">
      %1 = transform.structured.match ops{["linalg.elemwise_binary"]} in %arg0 : (!transform.any_op) -> !transform.op<"linalg.elemwise_binary">
      %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1 into %0 : (!transform.op<"linalg.elemwise_binary">, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----

#map = affine_map<(d0) -> (d0 * -50 + 123, 50)>
#map1 = affine_map<(d0) -> (d0 * -16 + 789, 16)>
#map2 = affine_map<(d0) -> (d0 * 50)>
#map3 = affine_map<(d0) -> (d0 * 16)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d1, d0)>
module {
  // CHECK-LABEL: func.func @fuse_consumer_multi_output
  // CHECK-SAME: %[[IN0:[0-9a-z]+]]: tensor<123x456xf32>
  // CHECK-SAME: %[[IN1:[0-9a-z]+]]: tensor<456x789xf32>
  // CHECK-SAME: %[[OUT:[0-9a-z]+]]: tensor<123x789xf32>
  func.func @fuse_consumer_multi_output(%arg0: tensor<123x456xf32>, %arg1: tensor<456x789xf32>, %arg2: tensor<123x789xf32>) -> (tensor<123x789xf32>, tensor<789x123xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    // CHECK: %[[INIT:.*]] = linalg.fill
    %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<123x789xf32>) -> tensor<123x789xf32>
    // CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<123x789xf32>
    %1 = tensor.empty() : tensor<123x789xf32>
    // CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<789x123xf32>
    %2 = tensor.empty() : tensor<789x123xf32>
    // CHECK: %[[RES:[0-9a-z]+]]:3 = scf.forall {{.*}} shared_outs(%[[LOOP_ARG0:.*]] = %[[INIT]], %[[LOOP_ARG1:.*]] = %[[EMPTY0]], %[[LOOP_ARG2:.*]] = %[[EMPTY1]]
    %3 = scf.forall (%arg3, %arg4) in (3, 50) shared_outs(%arg5 = %0) -> (tensor<123x789xf32>) {
      %5 = affine.min #map(%arg3)
      %6 = affine.min #map1(%arg4)
      %7 = affine.apply #map2(%arg3)
      %8 = affine.apply #map3(%arg4)
      %9 = affine.apply #map2(%arg3)
      %10 = affine.apply #map3(%arg4)
      // CHECK: %[[EXTRACT_IN0:.*]] = tensor.extract_slice %[[IN0]]
      %extracted_slice = tensor.extract_slice %arg0[%7, 0] [%5, 456] [1, 1] : tensor<123x456xf32> to tensor<?x456xf32>
      // CHECK: %[[EXTRACT_IN1:.*]] = tensor.extract_slice %[[IN1]]
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %8] [456, %6] [1, 1] : tensor<456x789xf32> to tensor<456x?xf32>
      // CHECK: %[[EXTRACT_OUT:.*]] = tensor.extract_slice %[[LOOP_ARG0]]
      %extracted_slice_1 = tensor.extract_slice %arg5[%9, %10] [%5, %6] [1, 1] : tensor<123x789xf32> to tensor<?x?xf32>
      // CHECK: %[[MATMUL_RES:.*]] = linalg.matmul ins(%[[EXTRACT_IN0]], %[[EXTRACT_IN1]] {{.*}} outs(%[[EXTRACT_OUT]]
      %11 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<?x456xf32>, tensor<456x?xf32>) outs(%extracted_slice_1 : tensor<?x?xf32>) -> tensor<?x?xf32>

      // CHECK: %[[EXTRACT_EMPTY0:.*]] = tensor.extract_slice %[[LOOP_ARG1]]
      // CHECK: %[[EXTRACT_EMPTY1:.*]] = tensor.extract_slice %[[LOOP_ARG2]]
      // CHECK: %[[GENERIC_RES:.*]]:2 = linalg.generic {{.*}} ins(%[[MATMUL_RES]] : tensor<?x?xf32>) outs(%[[EXTRACT_EMPTY0]], %[[EXTRACT_EMPTY1]]

      %12 = affine.apply #map2(%arg3)
      %13 = affine.apply #map3(%arg4)
      scf.forall.in_parallel {
        // CHECK: tensor.parallel_insert_slice %[[GENERIC_RES]]#0 into %[[LOOP_ARG1]]
        // CHECK: tensor.parallel_insert_slice %[[GENERIC_RES]]#1 into %[[LOOP_ARG2]]
        // CHECK: tensor.parallel_insert_slice %[[MATMUL_RES]] into %[[LOOP_ARG0]]
        tensor.parallel_insert_slice %11 into %arg5[%12, %13] [%5, %6] [1, 1] : tensor<?x?xf32> into tensor<123x789xf32>
      }
    }
    // CHECK: %[[ORI_OUTPUT:.*]]:2 = linalg.generic
    %4:2 = linalg.generic {indexing_maps = [#map4, #map4, #map5], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<123x789xf32>) outs(%1, %2 : tensor<123x789xf32>, tensor<789x123xf32>) {
    ^bb0(%in: f32, %out: f32, %out_0: f32):
      %5 = arith.addf %in, %out : f32
      %6 = arith.addf %5, %out_0 : f32
      linalg.yield %5, %6 : f32, f32
    } -> (tensor<123x789xf32>, tensor<789x123xf32>)
    // CHECK: return %[[RES]]#1, %[[RES]]#2
    return %4#0, %4#1 : tensor<123x789xf32>, tensor<789x123xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.op<"scf.forall">
      %1 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.op<"linalg.generic">
      %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %1 into %0 : (!transform.op<"linalg.generic">, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----

// This is a regression test. Make sure that the transform succeeds and valid
// IR is generated.

module {
  // CHECK-LABEL: func.func @softmax_dispatch_0_generic_16x128x128_f32
  func.func @softmax_dispatch_0_generic_16x128x128_f32() -> tensor<16x128x128xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<5.000000e+00> : tensor<16x128x128xf32>
    %cst_1 = arith.constant 5.000000e+00 : f32
    %1 = tensor.empty() : tensor<16x128xf32>
    %2 = tensor.empty() : tensor<16x128x128xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<16x128xf32>) -> tensor<16x128xf32>
    %5 = linalg.generic {producer, indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cst : tensor<16x128x128xf32>) outs(%4 : tensor<16x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.maximumf %in, %out : f32
      linalg.yield %8 : f32
    } -> tensor<16x128xf32>
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %7 = scf.forall (%arg0, %arg1) in (16, 32) shared_outs(%arg2 = %2) -> (tensor<16x128x128xf32>) {
      %11 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg1)
      %extracted_slice = tensor.extract_slice %5[%arg0, %11] [1, 4] [1, 1] : tensor<16x128xf32> to tensor<1x4xf32>
      %extracted_slice_3 = tensor.extract_slice %2[%arg0, %11, 0] [1, 4, 128] [1, 1, 1] : tensor<16x128x128xf32> to tensor<1x4x128xf32>
      %extracted_slice_4 = tensor.extract_slice %3[%arg0, %11] [1, 4] [1, 1] : tensor<16x128xf32> to tensor<1x4xf32>
      %15:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<1x4xf32>) outs(%extracted_slice_3, %extracted_slice_4 : tensor<1x4x128xf32>, tensor<1x4xf32>) {
      ^bb0(%in: f32, %out: f32, %out_9: f32):
        %22 = arith.subf %cst_1, %in : f32
        %23 = math.exp %22 : f32
        %24 = arith.addf %23, %out_9 : f32
        linalg.yield %23, %24 : f32, f32
      } -> (tensor<1x4x128xf32>, tensor<1x4xf32>)
      %extracted_slice_5 = tensor.extract_slice %5[%arg0, %11] [1, 4] [1, 1] : tensor<16x128xf32> to tensor<1x4xf32>
      %extracted_slice_6 = tensor.extract_slice %2[%arg0, %11, 0] [1, 4, 128] [1, 1, 1] : tensor<16x128x128xf32> to tensor<1x4x128xf32>
      %extracted_slice_7 = tensor.extract_slice %3[%arg0, %11] [1, 4] [1, 1] : tensor<16x128xf32> to tensor<1x4xf32>
      %19:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice_5 : tensor<1x4xf32>) outs(%extracted_slice_6, %extracted_slice_7 : tensor<1x4x128xf32>, tensor<1x4xf32>) {
      ^bb0(%in: f32, %out: f32, %out_9: f32):
        %22 = arith.subf %cst_1, %in : f32
        %23 = math.exp %22 : f32
        %24 = arith.addf %23, %out_9 : f32
        linalg.yield %23, %24 : f32, f32
      } -> (tensor<1x4x128xf32>, tensor<1x4xf32>)
      %extracted_slice_8 = tensor.extract_slice %arg2[%arg0, %11, 0] [1, 4, 128] [1, 1, 1] : tensor<16x128x128xf32> to tensor<1x4x128xf32>
      %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15#0, %19#1 : tensor<1x4x128xf32>, tensor<1x4xf32>) outs(%extracted_slice_8 : tensor<1x4x128xf32>) {
      ^bb0(%in: f32, %in_9: f32, %out: f32):
        %22 = arith.divf %in, %in_9 : f32
        linalg.yield %22 : f32
      } -> tensor<1x4x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %20 into %arg2[%arg0, %11, 0] [1, 4, 128] [1, 1, 1] : tensor<1x4x128xf32> into tensor<16x128x128xf32>
      }
    }
    return %7 : tensor<16x128x128xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match attributes{producer} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.generic">
      %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">
      transform.structured.fuse_into_containing_op %0 into %1
        : (!transform.op<"linalg.generic">, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.any_op)
        transform.yield
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
// Tests below are expected to fail.
////////////////////////////////////////////////////////////////////////////////

// -----

// NO-CHECK-LABEL-ON-EXPECTED-ERROR
func.func @copy_1d_1024xf16(%arg0: tensor<123x456xf32>, %arg1: tensor<456x789xf32>, %arg2 : tensor<123x789xf32>) -> tensor<123x789xf32> {
  %0 = arith.constant 0.000000e+00 : f32
  %1 = linalg.fill ins(%0 : f32) outs(%arg2 : tensor<123x789xf32>) -> tensor<123x789xf32>
  // expected-note @below {{containing op}}
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<123x456xf32>, tensor<456x789xf32>) outs(%1 : tensor<123x789xf32>) -> tensor<123x789xf32>
  return %2 : tensor<123x789xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %tiled_op, %forall_op = transform.structured.tile_using_forall %1
      num_threads [] tile_sizes [50, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // Note that we pass in %tiled_op, which isn't a container op.
    // expected-error @+2 {{could not find next target to fuse into container}}
    %fused_op, %new_containing_op =
      transform.structured.fuse_into_containing_op %0 into %tiled_op
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)
          transform.yield
  }
}
