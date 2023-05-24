// RUN: mlir-opt --test-transform-dialect-interpreter --split-input-file %s | FileCheck %s

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

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.fill">
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

    // linalg.fill is tileable. The op is tiled and fused.
    transform.structured.fuse_into_containing_op %0 into %1
      : (!transform.op<"linalg.fill">, !transform.op<"scf.forall">) -> !transform.any_op
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

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["tensor.empty"]} in %arg1 : (!transform.any_op) -> !transform.op<"tensor.empty">
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

    // tensor.empty is not tileable. The op is cloned and fused.
    transform.structured.fuse_into_containing_op %0 into %1
      : (!transform.op<"tensor.empty">, !transform.op<"scf.forall">) -> !transform.any_op
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

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.fill">
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

    // linalg.fill is tileable. The op is tiled and fused.
    transform.structured.fuse_into_containing_op %0 into %1
      : (!transform.op<"linalg.fill">, !transform.op<"scf.forall">) -> !transform.any_op
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

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // linalg.fill is tileable. The op is tiled and fused.
    transform.structured.fuse_into_containing_op %0 into %1
      : (!transform.any_op, !transform.any_op) -> !transform.any_op
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

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.generic">
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

    // linalg.generic is tileable. The op is tiled and fused.
    transform.structured.fuse_into_containing_op %0 into %1
      : (!transform.op<"linalg.generic">, !transform.op<"scf.forall">) -> !transform.any_op
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

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // Create a new handle that points to `linalg.fill` twice.
    %2 = transform.merge_handles %0, %0 : !transform.any_op

    // It shouldn't be a problem to fuse this handle.
    transform.structured.fuse_into_containing_op %2 into %1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
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

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.generic">
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

    // linalg.generic is tileable. The op is tiled and fused.
    transform.structured.fuse_into_containing_op %0 into %1
      : (!transform.op<"linalg.generic">, !transform.op<"scf.forall">) -> !transform.any_op
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

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.generic">
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

    // linalg.generic is tileable. The op is tiled and fused.
    transform.structured.fuse_into_containing_op %0 into %1
      : (!transform.op<"linalg.generic">, !transform.op<"scf.forall">) -> !transform.any_op
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
          %d = arith.maxf %a, %b : f32
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

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.op<"linalg.generic">
    %1 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.op<"scf.forall">

    // linalg.generic is tileable. The op is tiled and fused.
    transform.structured.fuse_into_containing_op %0 into %1
      : (!transform.op<"linalg.generic">, !transform.op<"scf.forall">) -> !transform.any_op
  }
}
