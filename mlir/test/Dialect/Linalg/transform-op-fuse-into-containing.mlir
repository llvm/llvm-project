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

    // CHECK: scf.foreach_thread {{.*}} {
    %2 = scf.foreach_thread (%arg3) in (%1) shared_outs(%o = %arg2) -> (tensor<?xf32>) {
      %3 = affine.apply #map1(%arg3)[%arg0]
      %4 = affine.min #map2(%arg3)[%d0, %arg0]
      %5 = tensor.extract_slice %o[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: %[[T0:.*]] = tensor.extract_slice %[[IN]][%{{.*}}] [%{{.*}}] [{{.*}}]
      // CHECK: %[[T1:.*]] = linalg.fill {{.*}} outs(%[[T0]]
      %6 = tensor.extract_slice %0[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: %[[T2:.*]] = linalg.elemwise_unary ins(%[[T1]]
      %7 = linalg.elemwise_unary ins(%6 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      scf.foreach_thread.perform_concurrently {
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

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.fill"]} in %arg1
      %1 = transform.structured.match ops{["scf.foreach_thread"]} in %arg1

      // linalg.fill is tileable. The op is tiled and fused.
      transform.structured.fuse_into_containing_op %0 into %1
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

    // CHECK: scf.foreach_thread {{.*}} {
    %2 = scf.foreach_thread (%arg3) in (%1) shared_outs(%o = %arg2) -> (tensor<64xf32>) {
      // CHECK: %[[INIT_TENSOR:.*]] = tensor.empty
      %3 = affine.apply #map1(%arg3)[%arg0]
      %4 = affine.min #map2(%arg3)[%arg0]
      %5 = tensor.extract_slice %o[%3] [%4] [1] : tensor<64xf32> to tensor<?xf32>

      // CHECK: %[[T2:.*]] = linalg.elemwise_unary ins(%[[INIT_TENSOR]]
      %7 = linalg.elemwise_unary ins(%0 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      scf.foreach_thread.perform_concurrently {
        tensor.parallel_insert_slice %7 into %o[%3] [%4] [1] : tensor<?xf32> into tensor<64xf32>
      }
    }
    // CHECK: }

    func.return %2 : tensor<64xf32>
  }

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["tensor.empty"]} in %arg1
      %1 = transform.structured.match ops{["scf.foreach_thread"]} in %arg1

      // tensor.empty is not tileable. The op is cloned and fused.
      transform.structured.fuse_into_containing_op %0 into %1
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

    // CHECK: scf.foreach_thread {{.*}} shared_outs(%[[BBARGOUT:.*]] = %[[OUT]]) -> (tensor<?xf32>) {
    %2 = scf.foreach_thread (%arg3) in (%1) shared_outs(%o = %0) -> (tensor<?xf32>) {
      %3 = affine.apply #map1(%arg3)[%arg0]
      %4 = affine.min #map2(%arg3)[%d0, %arg0]
      %5 = tensor.extract_slice %o[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: %[[T0:.*]] = tensor.extract_slice %[[BBARGOUT]][%{{.*}}] [%{{.*}}] [{{.*}}]
      // CHECK: %[[T1:.*]] = linalg.fill {{.*}} outs(%[[T0]]
      %6 = tensor.extract_slice %arg1[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: %[[T2:.*]] = linalg.elemwise_unary {{.*}} outs(%[[T1]]
      %7 = linalg.elemwise_unary ins(%6 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      scf.foreach_thread.perform_concurrently {
        tensor.parallel_insert_slice %7 into %o[%3] [%4] [1] : tensor<?xf32> into tensor<?xf32>
      }
    }
    // CHECK: }
    func.return %2 : tensor<?xf32>
  }

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1
    %1 = transform.structured.match ops{["scf.foreach_thread"]} in %arg1

    // linalg.fill is tileable. The op is tiled and fused.
    transform.structured.fuse_into_containing_op %0 into %1
  }
}
