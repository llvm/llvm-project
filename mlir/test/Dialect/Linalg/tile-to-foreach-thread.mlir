// RUN: mlir-opt %s --test-transform-dialect-interpreter -canonicalize -cse -split-input-file | FileCheck %s

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
  //      CHECK:   scf.foreach_thread.perform_concurrently {
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
    transform.sequence %arg0 failures(propagate) {
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
  //      CHECK:   %[[LB1:.+]] = affine.apply #[[$map3]](%[[IV1]])
  //      CHECK:   %[[tA:.+]] = tensor.extract_slice %[[A]][%[[LB0]], 0] [10, 200] [1, 1] :
  //      CHECK:   %[[tB:.+]] = tensor.extract_slice %[[B]][0, %[[LB1]]] [200, %[[TS]]] [1, 1] :
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
  transform.sequence %arg0 failures(propagate) {
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
  //      CHECK: scf.foreach_thread (%[[IV0:.+]], %[[IV1:.+]]) in (%[[NT0]], %[[NT1]])
  //      CHECK:   %[[TS0:.+]] = affine.min #[[$map2]](%[[IV0]])[%[[M]]]  
  //      CHECK:   %[[TS1:.+]] = affine.min #[[$map4]](%[[IV1]])[%[[N]]]
  //      CHECK:   %[[LB0:.+]] = affine.apply #[[$map5]](%[[IV0]])
  //      CHECK    tensor.extract_slice %[[A]]
  //      CHECK:   %[[LB1:.+]] = affine.apply #[[$map6]](%[[IV1]])
  //      CHECK    tensor.extract_slice %[[B]]
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
  transform.sequence %arg0 failures(propagate) {
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
  //      CHECK:   %[[LB1:.+]] = affine.apply #[[$map3]](%[[IV1]])
  //      CHECK:   %[[tA:.+]] = tensor.extract_slice %[[A]][%[[LB0]], 0] [10, 200] [1, 1] :
  //      CHECK:   %[[tB:.+]] = tensor.extract_slice %[[B]][0, %[[LB1]]] [200, %[[TS]]] [1, 1] :
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
  transform.sequence %arg0 failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1:2 = transform.structured.tile_to_foreach_thread_op %0 tile_sizes [10, 21]
  }
}

// -----

module {
  func.func @extract_source(%A: tensor<4xf32>, %B: tensor<16xf32>) -> tensor<4xf32> {
    %B1 = tensor.extract_slice %B[10] [4] [1] : tensor<16xf32> to tensor<4xf32>
    %result = linalg.generic {indexing_maps = [
      affine_map<(d0) -> (d0)>,affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%A : tensor<4xf32>) outs(%B1 : tensor<4xf32>) {
      ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
        %2 = arith.addf %arg3, %arg3 : f32
        linalg.yield %2 : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
      %1:2 = transform.structured.tile_to_foreach_thread_op %0 num_threads [2] (mapped to dims [0])
    }
  }
}
// CHECK-DAG: #[[$map0:.+]] = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: extract_source(
//       CHECK:  %[[C2:.*]] = arith.constant 2 : index
//       CHECK:  scf.foreach_thread (%[[ARG:.*]]) in (%[[C2]]) -> (tensor<4xf32>) {
//       CHECK:    %[[OFF:.*]] = affine.apply #[[$map0]](%[[ARG]])
//       CHECK:    scf.foreach_thread.perform_concurrently {
//       CHECK:      tensor.parallel_insert_slice %{{.*}} into %{{.*}}[%[[OFF]]] [2] [1] : tensor<2xf32> into tensor<4xf32>

// -----

// In this test case, matmul dims and tile size are dynamic.

// CHECK-DAG: #[[$map0:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
// CHECK-DAG: #[[$map1:.+]] = affine_map<()[s0] -> (s0 ceildiv 20)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0, s1)>
// CHECK-DAG: #[[$map3:.+]] = affine_map<(d0)[s0] -> (d0 * -20 + s0, 20)>
// CHECK-DAG: #[[$map4:.+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: #[[$map5:.+]] = affine_map<(d0) -> (d0 * 20)>

// CHECK-LABEL: matmul_tile_size_dynamic_dynamic(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor<?x?xf32>
func.func @matmul_tile_size_dynamic_dynamic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  //  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  //  CHECK-DAG: %[[tile_size:.*]] = "test.dummy"()
  //  CHECK-DAG: %[[M:.+]] = tensor.dim %[[A]], %[[c0]] :
  //  CHECK-DAG: %[[N:.+]] = tensor.dim %[[B]], %c1 :
  //  CHECK-DAG: %[[NT0:.+]] = affine.apply #[[$map0]]()[%[[M]], %[[tile_size]]]
  //  CHECK-DAG: %[[NT1:.+]] = affine.apply #[[$map1]]()[%[[N]]]
  //      CHECK: scf.foreach_thread (%[[IV0:.+]], %[[IV1:.+]]) in (%[[NT0]], %[[NT1]])
  //      CHECK    tensor.extract_slice %[[A]]
  //      CHECK    tensor.extract_slice %[[B]]
  //      CHECK    tensor.extract_slice %[[C]]
  //      CHECK:   linalg.matmul
  //      CHECK:   scf.foreach_thread.perform_concurrently
  // CHECK-NEXT:    tensor.parallel_insert_slice
  %tile_size = "test.dummy"() : () -> (index)
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
  return %0 : tensor<?x?xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %sz = transform.structured.match ops{["test.dummy"]} in %arg1
    %1:2 = transform.structured.tile_to_foreach_thread_op %0 tile_sizes [%sz, 20]
  }
}
