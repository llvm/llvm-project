// RUN: mlir-opt %s --test-transform-dialect-interpreter -canonicalize -split-input-file | FileCheck %s

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
      %1:2 = transform.structured.tile_to_foreach_thread_op %0 num_threads [10, 20] (mapped to dims [1, 0])
    }
  }
}

// -----

// Tests that dimension 0 can eliminate affine.min/max, dimension 1 cannot.

// CHECK-DAG: #[[$map0:.+]] = affine_map<(d0) -> (d0 * -15 + 300, 15)>
// CHECK-DAG: #[[$map1:.+]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-DAG: #[[$map3:.+]] = affine_map<(d0) -> (d0 * 15)>

// CHECK-LABEL: matmul_static(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor
func.func @matmul_static(%A: tensor<100x200xf32>, %B: tensor<200x300xf32>, %C: tensor<100x300xf32>) -> tensor<100x300xf32> {  
  //  CHECK-DAG: %[[c10:.+]] = arith.constant 10 : index
  //  CHECK-DAG: %[[c21:.+]] = arith.constant 21 : index
  //      CHECK: scf.foreach_thread (%[[IV0:.+]], %[[IV1:.+]]) in (%[[c10]], %[[c21]])
  //      CHECK:   %[[TSMIN:.+]] = affine.min #[[$map0]](%[[IV1]])
  //      CHECK:   %[[TS:.+]] = affine.max #[[$map1]](%[[TSMIN]])
  //  CHECK-NOT:   affine.min
  //  CHECK-NOT:   affine.max
  //      CHECK:   %[[LB0:.+]] = affine.apply #[[$map2]](%[[IV0]])
  //      CHECK:   %[[tA:.+]] = tensor.extract_slice %[[A]][%[[LB0]], 0] [10, 200] [1, 1] :
  //      CHECK:   %[[LB1:.+]] = affine.apply #[[$map3]](%[[IV1]])
  //      CHECK:   %[[tB:.+]] = tensor.extract_slice %[[B]][0, %[[LB1]]] [200, %[[TS]]] [1, 1] :
  //      CHECK:   %[[LB0:.+]] = affine.apply #[[$map2]](%[[IV0]])
  //      CHECK:   %[[LB1:.+]] = affine.apply #[[$map3]](%[[IV1]])
  //      CHECK:   %[[tC:.+]] = tensor.extract_slice %[[C]][%[[LB0]], %[[LB1]]] [10, %[[TS]]] [1, 1] :
  //      CHECK:   linalg.matmul
  //      CHECK:   scf.foreach_thread.perform_concurrently
  // CHECK-NEXT:    tensor.parallel_insert_slice
  %0 = linalg.matmul ins(%A, %B : tensor<100x200xf32>, tensor<200x300xf32>)
                    outs(%C : tensor<100x300xf32>) -> (tensor<100x300xf32>)
  return %0 : tensor<100x300xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1:2 = transform.structured.tile_to_foreach_thread_op %0 num_threads [10, 21]
  }
}


// -----

// CHECK-DAG: #[[$map0:.+]] = affine_map<()[s0] -> (s0 ceildiv 10)>
// CHECK-DAG: #[[$map1:.+]] = affine_map<()[s0] -> (s0 ceildiv 20)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<(d0)[s0] -> (d0 * -10 + s0, 10)>
// CHECK-DAG: #[[$map4:.+]] = affine_map<(d0)[s0] -> (d0 * -20 + s0, 20)>
// CHECK-DAG: #[[$map5:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-DAG: #[[$map6:.+]] = affine_map<(d0) -> (d0 * 20)>

// CHECK-LABEL: matmul_tile_size_dynamic(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor<?x?xf32>
func.func @matmul_tile_size_dynamic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {  
  //      CHECK: %[[M:.+]] = tensor.dim %[[A]], %c0 :
  //      CHECK: %[[N:.+]] = tensor.dim %[[B]], %c1 : 
  //      CHECK: %[[NT0:.+]] = affine.apply #map0()[%[[M]]]
  //      CHECK: %[[NT1:.+]] = affine.apply #map1()[%[[N]]]
  //      CHECK: %[[M:.+]] = tensor.dim %[[A]], %c0 :
  //      CHECK: %[[N:.+]] = tensor.dim %[[B]], %c1 :
  //      CHECK: scf.foreach_thread (%[[IV0:.+]], %[[IV1:.+]]) in (%[[NT0]], %[[NT1]])
  //      CHECK:   %[[TS0:.+]] = affine.min #[[$map2]](%[[IV0]])[%[[M]]]  
  //      CHECK:   %[[TS1:.+]] = affine.min #[[$map4]](%[[IV1]])[%[[N]]]
  //      CHECK:   %[[LB0:.+]] = affine.apply #[[$map5]](%[[IV0]])
  //      CHECK    tensor.extract_slice %[[A]]
  //      CHECK:   %[[LB1:.+]] = affine.apply #[[$map6]](%[[IV1]])
  //      CHECK    tensor.extract_slice %[[B]]
  //      CHECK:   %[[LB0:.+]] = affine.apply #[[$map5]](%[[IV0]])
  //      CHECK:   %[[LB1:.+]] = affine.apply #[[$map6]](%[[IV1]])
  //      CHECK    tensor.extract_slice %[[C]]
  //      CHECK:   linalg.matmul
  //      CHECK:   scf.foreach_thread.perform_concurrently
  // CHECK-NEXT:    tensor.parallel_insert_slice
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
  return %0 : tensor<?x?xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):    
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1:2 = transform.structured.tile_to_foreach_thread_op %0 tile_sizes [10, 20]
  }
}

// -----

// Tests that dimension 0 can eliminate affine.min/max, dimension 1 cannot.

// CHECK-DAG: #[[$map0:.+]] = affine_map<(d0) -> (d0 * -21 + 300, 21)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-DAG: #[[$map3:.+]] = affine_map<(d0) -> (d0 * 21)>

// CHECK-LABEL: matmul_tile_size_static(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor
func.func @matmul_tile_size_static(%A: tensor<100x200xf32>, %B: tensor<200x300xf32>, %C: tensor<100x300xf32>) -> tensor<100x300xf32> {
  //  CHECK-DAG: %[[c10:.+]] = arith.constant 10 :
  //  CHECK-DAG: %[[c15:.+]] = arith.constant 15 :
  //      CHECK: scf.foreach_thread (%[[IV0:.+]], %[[IV1:.+]]) in (%[[c10]], %[[c15]])
  //      CHECK:   %[[TS:.+]] = affine.min #[[$map0]](%[[IV1]])  
  //  CHECK-NOT:   affine.max
  //  CHECK-NOT:   affine.min
  //      CHECK:   %[[LB0:.+]] = affine.apply #[[$map2]](%[[IV0]])
  //      CHECK:   %[[tA:.+]] = tensor.extract_slice %[[A]][%[[LB0]], 0] [10, 200] [1, 1] :
  //      CHECK:   %[[LB1:.+]] = affine.apply #[[$map3]](%[[IV1]])
  //      CHECK:   %[[tB:.+]] = tensor.extract_slice %[[B]][0, %[[LB1]]] [200, %[[TS]]] [1, 1] :
  //      CHECK:   %[[LB0:.+]] = affine.apply #[[$map2]](%[[IV0]])
  //      CHECK:   %[[LB1:.+]] = affine.apply #[[$map3]](%[[IV1]])
  //      CHECK:   %[[tC:.+]] = tensor.extract_slice %[[C]][%[[LB0]], %[[LB1]]] [10, %[[TS]]] [1, 1] :
  //      CHECK:   linalg.matmul
  //      CHECK:   scf.foreach_thread.perform_concurrently
  // CHECK-NEXT:    tensor.parallel_insert_slice
  %0 = linalg.matmul ins(%A, %B : tensor<100x200xf32>, tensor<200x300xf32>)
                    outs(%C : tensor<100x300xf32>) -> (tensor<100x300xf32>)
  return %0 : tensor<100x300xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1:2 = transform.structured.tile_to_foreach_thread_op %0 tile_sizes [10, 21]
  }
}
