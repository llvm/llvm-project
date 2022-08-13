// RUN: mlir-opt %s --test-transform-dialect-interpreter --split-input-file -canonicalize | FileCheck %s

// This is a simple tile-and-fuse example with a single fusion group.

module {
  // CHECK: func @foo
  // CHECK:   scf.foreach_thread {{.*}} {
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
      %16 = arith.maxf %arg3, %cst : f32
      %17 = arith.cmpf ogt, %arg2, %cst : f32
      %18 = arith.select %17, %cst, %16 : f32
      linalg.yield %18 : f32
    } -> tensor<?x?xf32>
    return %7 : tensor<?x?xf32>
  }

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
      // Find the root and all producers.
      %root = transform.structured.match attributes{"__root__"} in %arg1
      %producers = transform.structured.match attributes{"__producer__"} in %arg1

      // Tile the root.
      %foreach_thread_op, %tiled_op = transform.structured.tile_to_foreach_thread_op %root num_threads [10, 20]

      // Fuse all producers.
      transform.structured.fuse_into_containing_op %producers into %foreach_thread_op
    }
  }
}

// -----

// Inverse the order of the payload ops passed to the tile_to_foreach_thread_op
// op. Fusion should still work.

module {
  // CHECK: func @foo
  // CHECK:   scf.foreach_thread {{.*}} {
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
      %16 = arith.maxf %arg3, %cst : f32
      %17 = arith.cmpf ogt, %arg2, %cst : f32
      %18 = arith.select %17, %cst, %16 : f32
      linalg.yield %18 : f32
    } -> tensor<?x?xf32>
    return %7 : tensor<?x?xf32>
  }

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
      // Find the root and all producers.
      %root = transform.structured.match attributes{"__root__"} in %arg1
      %producers = transform.structured.match attributes{"__producer__"} in %arg1
      %reversed_producers = transform.test_reverse_payload_ops %producers

      // Tile the root.
      %foreach_thread_op, %tiled_op = transform.structured.tile_to_foreach_thread_op %root num_threads [10, 20]

      // Fuse all producers.
      transform.structured.fuse_into_containing_op %reversed_producers into %foreach_thread_op
    }
  }
}
