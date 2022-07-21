// RUN: mlir-opt %s --test-transform-dialect-interpreter -canonicalize | FileCheck %s

// Offset per thread:
// CHECK-DAG: affine_map<(d0)[s0] -> (d0 * (s0 ceildiv 10))>
// Per thread tile size.
// CHECK-DAG: affine_map<(d0)[s0] -> (-(d0 * (s0 ceildiv 10)) + s0, s0 ceildiv 10)>
// CHECK-DAG: affine_map<(d0)[s0] -> (d0 * (s0 ceildiv 20))>
// CHECK-DAG: affine_map<(d0)[s0] -> (-(d0 * (s0 ceildiv 20)) + s0, s0 ceildiv 20)>

module {
// CHECK-LABEL: matmul(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor<?x?xf32>
  func.func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  //  CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index
  //  CHECK-DAG: %[[C20:.*]] = arith.constant 20 : index
  //      CHECK: scf.foreach_thread ({{.*}}) in (%[[C10]], %[[C20]]) -> (tensor<?x?xf32>) {
  //      CHECK:   %[[tA:.*]] = tensor.extract_slice %[[A]]{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:   %[[tB:.*]] = tensor.extract_slice %[[B]]{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:   %[[tC:.*]] = tensor.extract_slice %[[C]]{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:   %[[RES:.*]] = linalg.matmul
  // CHECK-SAME:      ins(%[[tA]], %[[tB]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[tC]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-NEXT:   scf.foreach_thread.perform_concurrently {
  // CHECK-NEXT:     tensor.parallel_insert_slice %[[RES]] into %[[C]]{{.*}} :
  // CHECK-SAME:       tensor<?x?xf32> into tensor<?x?xf32>
  // CHECK-NEXT:   }
  // CHECK-NEXT: } {thread_dim_mapping = [1, 0]}
    %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                      outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
    return %0 : tensor<?x?xf32>
  }

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 {
    ^bb1(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
      %1:2 = transform.structured.tile_to_foreach_thread_op %0 [10, 20] (mapped to dims [1, 0])
    }
  }
}
