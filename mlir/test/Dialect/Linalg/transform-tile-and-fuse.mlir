// RUN: mlir-opt %s --transform-interpreter --split-input-file -canonicalize | FileCheck %s

// This is a simple tile-and-fuse example with a single fusion group.

module {
  // CHECK: func @foo
  // CHECK:   scf.forall {{.*}} {
  // CHECK:     linalg.fill
  // CHECK:     linalg.matmul
  // CHECK:     linalg.generic
  // CHECK:   }
  func.func @foo(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?xf32>,
                 %D: tensor<?x?xf32>, %sz0: index, %sz1: index)
      -> tensor<?x?xf32>
  {
    %cst = arith.constant 0.000000e+00 : f32
    %5 = linalg.fill
        {__producer__}
        ins(%cst : f32)
        outs(%D : tensor<?x?xf32>) -> tensor<?x?xf32>
    %6 = linalg.matmul
        {__producer__}
        ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %7 = linalg.generic
        {__root__,
         indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                          affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]
        }
        ins(%C, %6 : tensor<?xf32>, tensor<?x?xf32>)
        outs(%D : tensor<?x?xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %16 = arith.maximumf %arg3, %cst : f32
      %17 = arith.cmpf ogt, %arg2, %cst : f32
      %18 = arith.select %17, %cst, %16 : f32
      linalg.yield %18 : f32
    } -> tensor<?x?xf32>
    return %7 : tensor<?x?xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      // Find the root and all producers.
      %root = transform.structured.match attributes{"__root__"} in %arg1 : (!transform.any_op) -> !transform.any_op
      %producers = transform.structured.match attributes{"__producer__"} in %arg1 : (!transform.any_op) -> !transform.any_op

      // Tile the root.
      %tiled_op, %forall_op = transform.structured.tile_using_forall %root num_threads [10, 20]
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

      // Fuse all producers.
      transform.structured.fuse_into_containing_op %producers into %forall_op
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.yield
    }
  }
}

// -----

// Inverse the order of the payload ops passed to the tile_using_forall
// op. Fusion should still work.

module {
  // CHECK: func @foo
  // CHECK:   scf.forall {{.*}} {
  // CHECK:     linalg.fill
  // CHECK:     linalg.matmul
  // CHECK:     linalg.generic
  // CHECK:   }
  func.func @foo(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?xf32>,
                 %D: tensor<?x?xf32>, %sz0: index, %sz1: index)
      -> tensor<?x?xf32>
  {
    %cst = arith.constant 0.000000e+00 : f32
    %5 = linalg.fill
        {__producer__}
        ins(%cst : f32)
        outs(%D : tensor<?x?xf32>) -> tensor<?x?xf32>
    %6 = linalg.matmul
        {__producer__}
        ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
    %7 = linalg.generic
        {__root__,
         indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                          affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>],
         iterator_types = ["parallel", "parallel"]
        }
        ins(%C, %6 : tensor<?xf32>, tensor<?x?xf32>)
        outs(%D : tensor<?x?xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      %16 = arith.maximumf %arg3, %cst : f32
      %17 = arith.cmpf ogt, %arg2, %cst : f32
      %18 = arith.select %17, %cst, %16 : f32
      linalg.yield %18 : f32
    } -> tensor<?x?xf32>
    return %7 : tensor<?x?xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      // Find the root and all producers.
      %root = transform.structured.match attributes{"__root__"} in %arg1 : (!transform.any_op) -> !transform.any_op
      %producers = transform.structured.match attributes{"__producer__"} in %arg1 : (!transform.any_op) -> !transform.any_op
      %reversed_producers = transform.test_reverse_payload_ops %producers : (!transform.any_op) -> !transform.any_op

      // Tile the root.
      %tiled_op, %forall_op = transform.structured.tile_using_forall %root num_threads [10, 20]
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

      // Fuse all producers.
      transform.structured.fuse_into_containing_op %reversed_producers into %forall_op
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.yield
    }
  }
}
