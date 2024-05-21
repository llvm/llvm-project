// RUN: mlir-opt %s --transform-interpreter -canonicalize -cse -split-input-file -verify-diagnostics | FileCheck %s

// Offset per thread:
// CHECK-DAG: affine_map<(d0)[s0] -> (d0 * (s0 ceildiv 10))>
// Per thread tile size.
// CHECK-DAG: affine_map<(d0)[s0] -> (s0 ceildiv 10, -(d0 * (s0 ceildiv 10)) + s0)>
// CHECK-DAG: affine_map<(d0)[s0] -> (d0 * (s0 ceildiv 20))>
// CHECK-DAG: affine_map<(d0)[s0] -> (s0 ceildiv 20, -(d0 * (s0 ceildiv 20)) + s0)>

module {
// CHECK-LABEL: matmul(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor<?x?xf32>
  func.func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  //      CHECK: scf.forall ({{.*}}) in (10, 20) shared_outs(%[[C_BLK:.*]] = %[[C]]) -> (tensor<?x?xf32>) {
  //      CHECK:   %[[tA:.*]] = tensor.extract_slice %[[A]]{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:   %[[tB:.*]] = tensor.extract_slice %[[B]]{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:   %[[tC:.*]] = tensor.extract_slice %[[C_BLK]]{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:   %[[RES:.*]] = linalg.matmul
  // CHECK-SAME:      ins(%[[tA]], %[[tB]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[tC]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  //      CHECK:   scf.forall.in_parallel {
  // CHECK-NEXT:     tensor.parallel_insert_slice %[[RES]] into %[[C_BLK]]{{.*}} :
  // CHECK-SAME:       tensor<?x?xf32> into tensor<?x?xf32>
  // CHECK-NEXT:   }
  // CHECK-NEXT: } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                      outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
    return %0 : tensor<?x?xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1:2 = transform.structured.tile_using_forall %0 num_threads [10, 20] (mapping = [ #gpu.thread<y>, #gpu.thread<x> ] )
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
     transform.yield
    }
  }
}

// -----

module {
  // CHECK-LABEL: func @matmul_memref(
  //       CHECK:   scf.forall (%{{.*}}, %{{.*}}) in (10, 20) {
  //       CHECK:     memref.subview
  //       CHECK:     memref.subview
  //       CHECK:     memref.subview
  //       CHECK:     linalg.matmul
  //       CHECK:   } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  func.func @matmul_memref(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
    linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                  outs(%C : memref<?x?xf32>)
    return
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1:2 = transform.structured.tile_using_forall %0 num_threads [10, 20] (mapping = [ #gpu.thread<y>, #gpu.thread<x> ] )
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----

module {
  // CHECK-LABEL: func @copy_memref(
  //       CHECK:   scf.forall (%{{.*}}, %{{.*}}) in (10, 20) {
  //       CHECK:     memref.subview
  //       CHECK:     memref.subview
  //       CHECK:     linalg.copy
  //       CHECK:   } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  func.func @copy_memref(%A: memref<?x?xf32>, %B: memref<?x?xf32>) {
    linalg.copy ins(%A: memref<?x?xf32>)
                outs(%B : memref<?x?xf32>)
    return
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1:2 = transform.structured.tile_using_forall %0 num_threads [10, 20] (mapping = [ #gpu.thread<y>, #gpu.thread<x> ] )
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}

// -----

// In this test case, matmul dims and tile size are dynamic.

// CHECK-DAG: #[[$map0:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<(d0)[s0, s1] -> (s0, -(d0 * s0) + s1)>
// CHECK-DAG: #[[$map4:.+]] = affine_map<(d0)[s0] -> (d0 * s0)>

// CHECK-LABEL: matmul_tile_size_dynamic_dynamic(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor<?x?xf32>
func.func @matmul_tile_size_dynamic_dynamic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  //  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  //  CHECK-DAG: %[[tile_size_1:.*]] = "test.dummy"()
  //  CHECK-DAG: %[[tile_size_2:.*]] = "test.dummy"()
  //  CHECK-DAG: %[[M:.+]] = tensor.dim %[[A]], %[[c0]] :
  //  CHECK-DAG: %[[N:.+]] = tensor.dim %[[B]], %c1 :
  //  CHECK-DAG: %[[NT0:.+]] = affine.apply #[[$map0]]()[%[[M]], %[[tile_size_1]]]
  //  CHECK-DAG: %[[NT1:.+]] = affine.apply #[[$map0]]()[%[[N]], %[[tile_size_2]]]
  //      CHECK: scf.forall (%[[IV0:.+]], %[[IV1:.+]]) in (%[[NT0]], %[[NT1]]) shared_outs(%[[C_BLK:.*]] = %[[C]])
  //      CHECK:   tensor.extract_slice %[[A]]
  //      CHECK:   tensor.extract_slice %[[B]]
  //      CHECK:   tensor.extract_slice %[[C_BLK]]
  //      CHECK:   linalg.matmul
  //      CHECK:   scf.forall.in_parallel
  // CHECK-NEXT:    tensor.parallel_insert_slice
  %tile_size_1 = "test.dummy"() : () -> (index)
  %tile_size_2 = "test.dummy"() : () -> (index)
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sz = transform.structured.match ops{["test.dummy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_forall %0 tile_sizes *(%sz)
           : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Tests that dimension 0 can eliminate affine.min/max, dimension 1 cannot.

// CHECK-DAG: #[[$map0:.+]] = affine_map<(d0) -> (15, d0 * -15 + 300)>
// CHECK-DAG: #[[$map1:.+]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-DAG: #[[$map3:.+]] = affine_map<(d0) -> (d0 * 15)>

// CHECK-LABEL: matmul_static(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor
func.func @matmul_static(%A: tensor<100x200xf32>, %B: tensor<200x300xf32>, %C: tensor<100x300xf32>) -> tensor<100x300xf32> {
  //      CHECK: scf.forall (%[[IV0:.+]], %[[IV1:.+]]) in (10, 21) shared_outs(%[[C_BLK:.*]] = %[[C]])
  //      CHECK:   %[[TSMIN:.+]] = affine.min #[[$map0]](%[[IV1]])
  //      CHECK:   %[[TS:.+]] = affine.max #[[$map1]](%[[TSMIN]])
  //  CHECK-NOT:   affine.min
  //  CHECK-NOT:   affine.max
  //      CHECK:   %[[LB0:.+]] = affine.apply #[[$map2]](%[[IV0]])
  //      CHECK:   %[[LB1:.+]] = affine.apply #[[$map3]](%[[IV1]])
  //      CHECK:   %[[tA:.+]] = tensor.extract_slice %[[A]][%[[LB0]], 0] [10, 200] [1, 1] :
  //      CHECK:   %[[tB:.+]] = tensor.extract_slice %[[B]][0, %[[LB1]]] [200, %[[TS]]] [1, 1] :
  //      CHECK:   %[[tC:.+]] = tensor.extract_slice %[[C_BLK]][%[[LB0]], %[[LB1]]] [10, %[[TS]]] [1, 1] :
  //      CHECK:   linalg.matmul
  //      CHECK:   scf.forall.in_parallel
  // CHECK-NEXT:    tensor.parallel_insert_slice
  %0 = linalg.matmul ins(%A, %B : tensor<100x200xf32>, tensor<200x300xf32>)
                    outs(%C : tensor<100x300xf32>) -> (tensor<100x300xf32>)
  return %0 : tensor<100x300xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_forall %0 num_threads [10, 21]
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 10)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 20)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0)[s0] -> (d0 * -10 + s0, 10)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0)[s0] -> (d0 * -20 + s0, 20)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0) -> (d0 * 20)>

//       CHECK: matmul_tile_size_dynamic(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor<?x?xf32>
func.func @matmul_tile_size_dynamic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  //      CHECK: %[[M:.+]] = tensor.dim %[[A]], %c0 :
  //      CHECK: %[[N:.+]] = tensor.dim %[[B]], %c1 :
  //      CHECK: %[[NT0:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
  //      CHECK: %[[NT1:.+]] = affine.apply #[[MAP1]]()[%[[N]]]
  //      CHECK: scf.forall (%[[IV0:.+]], %[[IV1:.+]]) in (%[[NT0]], %[[NT1]]) shared_outs(%[[C_BLK:.*]] = %[[C]])
  //      CHECK:   %[[TS0:.+]] = affine.min #[[MAP2]](%[[IV0]])[%[[M]]]
  //      CHECK:   %[[TS1:.+]] = affine.min #[[MAP3]](%[[IV1]])[%[[N]]]
  //      CHECK:   %[[LB0:.+]] = affine.apply #[[MAP4]](%[[IV0]])
  //      CHECK:   %[[LB1:.+]] = affine.apply #[[MAP5]](%[[IV1]])
  //      CHECK:   tensor.extract_slice %[[A]]
  //      CHECK:   tensor.extract_slice %[[B]]
  //      CHECK:   tensor.extract_slice %[[C_BLK]]
  //      CHECK:   linalg.matmul
  //      CHECK:   scf.forall.in_parallel
  // CHECK-NEXT:    tensor.parallel_insert_slice
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_forall %0 tile_sizes [10, 20]
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// -----

// Tests that dimension 0 can eliminate affine.min/max, dimension 1 cannot.

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0) -> (d0 * -21 + 300, 21)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 21)>

//       CHECK: matmul_tile_size_static(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor
func.func @matmul_tile_size_static(%A: tensor<100x200xf32>, %B: tensor<200x300xf32>, %C: tensor<100x300xf32>) -> tensor<100x300xf32> {
  //      CHECK: scf.forall (%[[IV0:.+]], %[[IV1:.+]]) in (10, 15) shared_outs(%[[C_BLK:.*]] = %[[C]])
  //      CHECK:   %[[TS:.+]] = affine.min #[[MAP0]](%[[IV1]])
  //  CHECK-NOT:   affine.max
  //  CHECK-NOT:   affine.min
  //      CHECK:   %[[LB0:.+]] = affine.apply #[[MAP1]](%[[IV0]])
  //      CHECK:   %[[LB1:.+]] = affine.apply #[[MAP2]](%[[IV1]])
  //      CHECK:   %[[tA:.+]] = tensor.extract_slice %[[A]][%[[LB0]], 0] [10, 200] [1, 1] :
  //      CHECK:   %[[tB:.+]] = tensor.extract_slice %[[B]][0, %[[LB1]]] [200, %[[TS]]] [1, 1] :
  //      CHECK:   %[[tC:.+]] = tensor.extract_slice %[[C_BLK]][%[[LB0]], %[[LB1]]] [10, %[[TS]]] [1, 1] :
  //      CHECK:   linalg.matmul
  //      CHECK:   scf.forall.in_parallel
  // CHECK-NEXT:    tensor.parallel_insert_slice
  %0 = linalg.matmul ins(%A, %B : tensor<100x200xf32>, tensor<200x300xf32>)
                    outs(%C : tensor<100x300xf32>) -> (tensor<100x300xf32>)
  return %0 : tensor<100x300xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_forall %0 tile_sizes [10, 21]
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
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

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1:2 = transform.structured.tile_using_forall %0 num_threads [2] ( mapping = [#gpu.thread<x>])
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}
// CHECK-DAG: #[[$map0:.+]] = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: extract_source(
//       CHECK:  scf.forall (%[[ARG:.*]]) in (2) shared_outs(%{{.*}} = %{{.*}}) -> (tensor<4xf32>) {
//       CHECK:    %[[OFF:.*]] = affine.apply #[[$map0]](%[[ARG]])
//       CHECK:    scf.forall.in_parallel {
//       CHECK:      tensor.parallel_insert_slice %{{.*}} into %{{.*}}[%[[OFF]]] [2] [1] : tensor<2xf32> into tensor<4xf32>

// -----

// In this test case, matmul dims and tile size are dynamic.

// CHECK-DAG: #[[$map0:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
// CHECK-DAG: #[[$map1:.+]] = affine_map<()[s0] -> (s0 ceildiv 20)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<(d0)[s0, s1] -> (s0, -(d0 * s0) + s1)>
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
  //      CHECK: scf.forall (%[[IV0:.+]], %[[IV1:.+]]) in (%[[NT0]], %[[NT1]]) shared_outs(%[[C_BLK:.*]] = %[[C]])
  //      CHECK:   tensor.extract_slice %[[A]]
  //      CHECK:   tensor.extract_slice %[[B]]
  //      CHECK:   tensor.extract_slice %[[C_BLK]]
  //      CHECK:   linalg.matmul
  //      CHECK:   scf.forall.in_parallel
  // CHECK-NEXT:    tensor.parallel_insert_slice
  %tile_size = "test.dummy"() : () -> (index)
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sz = transform.structured.match ops{["test.dummy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1:2 = transform.structured.tile_using_forall %0 tile_sizes [%sz, 20]
           : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-DAG: #[[$map0:.+]] = affine_map<(d0) -> (d0 * -15 + 100, 15)>
// CHECK-DAG: #[[$map1:.+]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<(d0) -> (d0 * 15)>
// CHECK-DAG: #[[$map3:.+]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: tile_output_multi_1d_static(
//  CHECK-SAME:   %[[IN1:[0-9a-z]+]]: tensor<100xf32>
//  CHECK-SAME:   %[[IN2:[0-9a-z]+]]: tensor<100xf32>
//  CHECK-SAME:   %[[ORGOUT1:[0-9a-z]+]]: tensor<100xf32>
//  CHECK-SAME:   %[[ORGOUT2:[0-9a-z]+]]: tensor<100xf32>
  func.func @tile_output_multi_1d_static(%IN1: tensor<100xf32>, %IN2: tensor<100xf32>,
                                         %OUT1: tensor<100xf32>, %OUT2: tensor<100xf32>)
                                         -> (tensor<100xf32>, tensor<100xf32>) {
//      CHECK: scf.forall (%[[IV0:.+]]) in (7) shared_outs(%[[OUT1:[0-9a-z]+]] = %[[ORGOUT1]], %[[OUT2:[0-9a-z]+]] = %[[ORGOUT2]])
//      CHECK:   %[[TSMIN:.+]] = affine.min #[[$map0]](%[[IV0]])
//      CHECK:   %[[TS:.+]] = affine.max #[[$map1]](%[[TSMIN]])
//  CHECK-NOT:   affine.min
//  CHECK-NOT:   affine.max
//      CHECK:   %[[LB:.+]] = affine.apply #[[$map2]](%[[IV0]])
//      CHECK:   %[[tIN1:.+]] = tensor.extract_slice %[[IN1]][%[[LB]]] [%[[TS]]] [1] :
//      CHECK:   %[[tIN2:.+]] = tensor.extract_slice %[[IN2]][%[[LB]]] [%[[TS]]] [1] :
//      CHECK:   %[[tOUT1:.+]] = tensor.extract_slice %[[OUT1]][%[[LB]]] [%[[TS]]] [1] :
//      CHECK:   %[[tOUT2:.+]] = tensor.extract_slice %[[OUT2]][%[[LB]]] [%[[TS]]] [1] :
//      CHECK:   %[[RES1:[0-9]+]]:[[RES2:[0-9]+]] = linalg.generic
//      CHECK:   scf.forall.in_parallel
// CHECK-NEXT:    tensor.parallel_insert_slice %[[RES1]]#0 into %[[OUT1]][%[[LB]]] [%[[TS]]] [1] :
// CHECK-NEXT:    tensor.parallel_insert_slice %[[RES1]]#1 into %[[OUT2]][%[[LB]]] [%[[TS]]] [1] :
    %res1, %res2 = linalg.generic
    {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%IN1, %IN2 : tensor<100xf32>, tensor<100xf32>)
      outs(%OUT1, %OUT2 : tensor<100xf32>, tensor<100xf32>)
    {
      ^bb0(%a1: f32, %a2: f32, %a3: f32, %a4: f32):
        %1 = arith.addf %a1, %a3 : f32
        %2 = arith.addf %a2, %a4 : f32
        linalg.yield %1, %2 : f32,f32
    } -> (tensor<100xf32>, tensor<100xf32>)
    return %res1, %res2 : tensor<100xf32>, tensor<100xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %tiled_generic, %forall = transform.structured.tile_using_forall %0 num_threads [7]
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }

// -----

// CHECK-DAG: #[[$map0:.+]] = affine_map<(d0) -> (d0 * 75)>
// CHECK-DAG: #[[$map1:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<(d0, d1) -> (d1, d0)
// CHECK-DAG: #[[$map3:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[$map4:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: tile_output_multi_1d2d_static(
//  CHECK-SAME:   %[[IN1:[0-9a-z]+]]: tensor<100xf32>
//  CHECK-SAME:   %[[IN2:[0-9a-z]+]]: tensor<100x300xf32>
//  CHECK-SAME:   %[[IN3:[0-9a-z]+]]: tensor<300xf32>
//  CHECK-SAME:   %[[ORGOUT1:[0-9a-z]+]]: tensor<300x100xf32>
//  CHECK-SAME:   %[[ORGOUT2:[0-9a-z]+]]: tensor<300xf32>
  func.func @tile_output_multi_1d2d_static(%IN1: tensor<100xf32>, %IN2: tensor<100x300xf32>, %IN3: tensor<300xf32>,
                     %OUT1: tensor<300x100xf32>, %OUT2: tensor<300xf32>)
                     -> (tensor<300x100xf32>, tensor<300xf32>) {
//      CHECK: scf.forall (%[[IV0:.+]]) in (4) shared_outs(%[[OUT1:[0-9a-z]+]] = %[[ORGOUT1]], %[[OUT2:[0-9a-z]+]] = %[[ORGOUT2]])
//      CHECK:   %[[LB:.+]] = affine.apply #[[$map0]](%[[IV0]])
//      CHECK:   %[[tIN1:.+]] = tensor.extract_slice %[[IN2]][0, %[[LB]]] [100, 75]
//      CHECK:   %[[tIN2:.+]] = tensor.extract_slice %[[IN3]][%[[LB]]] [75]
//      CHECK:   %[[tOUT1:.+]] = tensor.extract_slice %[[OUT1]][%[[LB]], 0] [75, 100]
//      CHECK:   %[[tOUT2:.+]] = tensor.extract_slice %[[OUT2]][%[[LB]]] [75]
//      CHECK:   %[[RES1:[0-9]+]]:[[RES2:[0-9]+]] = linalg.generic
//      CHECK:   scf.forall.in_parallel
// CHECK-NEXT:    tensor.parallel_insert_slice %[[RES1]]#0 into %[[OUT1]][%[[LB]], 0] [75, 100]
// CHECK-NEXT:    tensor.parallel_insert_slice %[[RES1]]#1 into %[[OUT2]][%[[LB]]] [75]
    %res2, %res3 = linalg.generic {
      indexing_maps = [affine_map<(d0,d1) -> (d1)>,
                       affine_map<(d0,d1) -> (d1,d0)>,
                       affine_map<(d0,d1) -> (d0)>,
                       affine_map<(d0,d1) -> (d0,d1)>,
                       affine_map<(d0,d1) -> (d0)>
                       ],
      iterator_types = ["parallel", "parallel"]
    } ins(%IN1, %IN2, %IN3 : tensor<100xf32>, tensor<100x300xf32>, tensor<300xf32>)
      outs(%OUT1, %OUT2: tensor<300x100xf32>, tensor<300xf32>)  {
      ^bb0(%i1: f32, %i2: f32, %i3: f32, %o1: f32, %o2: f32):
        %1 = arith.addf %i1, %o1 : f32
        %2 = arith.addf %i2, %1 : f32
        %3 = arith.addf %i3, %2 : f32
        linalg.yield %3, %i3 : f32, f32
    } -> (tensor<300x100xf32>, tensor<300xf32>)

    return %res2, %res3 : tensor<300x100xf32>, tensor<300xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%IN_MAT2: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %IN_MAT2 : (!transform.any_op) -> !transform.any_op
      %tiled_generic, %forall = transform.structured.tile_using_forall %0 num_threads [4]
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }

// -----

// CHECK-DAG: #[[$map0:.+]] = affine_map<()[s0] -> (s0 ceildiv 10)>
// CHECK-DAG: #[[$map1:.+]] = affine_map<()[s0] -> (s0 ceildiv 20)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<(d0)[s0] -> (d0 * -10 + s0, 10)>
// CHECK-DAG: #[[$map3:.+]] = affine_map<(d0)[s0] -> (d0 * -20 + s0, 20)>
// CHECK-DAG: #[[$map4:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-DAG: #[[$map5:.+]] = affine_map<(d0) -> (d0 * 20)>

// CHECK-LABEL: matmul_tile_size_dynamic(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor<?x?xf32>
func.func @matmul_tile_size_dynamic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  //      CHECK: %[[c1:.*]] = arith.constant 1 : index
  //      CHECK: %[[c0:.*]] = arith.constant 0 : index
  //      CHECK: %[[M:.+]] = tensor.dim %[[A]], %[[c0]] :
  //      CHECK: %[[N:.+]] = tensor.dim %[[B]], %[[c1]] :
  //      CHECK: %[[NT0:.+]] = affine.apply #map()[%[[M]]]
  //      CHECK: %[[NT1:.+]] = affine.apply #map1()[%[[N]]]
  //      CHECK: %[[K:.+]] = tensor.dim %[[A]], %[[c1]] :
  //      CHECK: scf.forall (%[[IV0:.+]], %[[IV1:.+]]) in (%[[NT0]], %[[NT1]]) shared_outs(%[[C_BLK:.*]] = %[[C]])
  //      CHECK:   %[[TS0:.+]] = affine.min #[[$map2]](%[[IV0]])[%[[M]]]
  //      CHECK:   %[[TS1:.+]] = affine.min #[[$map3]](%[[IV1]])[%[[N]]]
  //      CHECK:   %[[LB0:.+]] = affine.apply #[[$map4]](%[[IV0]])
  //      CHECK:   %[[LB1:.+]] = affine.apply #[[$map5]](%[[IV1]])
  //      CHECK:   tensor.extract_slice %[[A]][%[[LB0]], 0] [%[[TS0]], %[[K]]] [1, 1] :
  //      CHECK:   tensor.extract_slice %[[B]][0, %[[LB1]]] [%[[K]], %[[TS1]]] [1, 1] :
  //      CHECK:   tensor.extract_slice %[[C_BLK]][%[[LB0]], %[[LB1]]] [%[[TS0]], %[[TS1]]] [1, 1] :
  //      CHECK:   linalg.matmul
  //      CHECK:   scf.forall.in_parallel
  // CHECK-NEXT:    tensor.parallel_insert_slice
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sz = transform.param.constant 10 : i64 -> !transform.param<i64>
    %1:2 = transform.structured.tile_using_forall %0 tile_sizes [%sz, 20]
           : (!transform.any_op, !transform.param<i64>) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @matmul_tile_size_dynamic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul_transpose_b"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %c10 = transform.param.constant 10 : i64 -> !transform.param<i64>
    %c20 = transform.param.constant 20 : i64 -> !transform.param<i64>
    %sz = transform.merge_handles %c10, %c20 : !transform.param<i64>
    // expected-error @below {{requires exactly one parameter associated}}
    %1:2 = transform.structured.tile_using_forall %0 tile_sizes [%sz, 20]
           : (!transform.any_op, !transform.param<i64>) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-DAG: #[[$map0:.+]] = affine_map<()[s0] -> (s0 ceildiv 10)>
// CHECK-DAG: #[[$map1:.+]] = affine_map<()[s0] -> (s0 ceildiv 20)>
// CHECK-DAG: #[[$map2:.+]] = affine_map<(d0)[s0] -> (d0 * -10 + s0, 10)>
// CHECK-DAG: #[[$map3:.+]] = affine_map<(d0)[s0] -> (d0 * -20 + s0, 20)>
// CHECK-DAG: #[[$map4:.+]] = affine_map<(d0) -> (d0 * 10)>
// CHECK-DAG: #[[$map5:.+]] = affine_map<(d0) -> (d0 * 20)>

// CHECK-LABEL: matmul_tile_size_dynamic(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor<?x?xf32>
func.func @matmul_tile_size_dynamic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  //      CHECK: %[[c1:.*]] = arith.constant 1 : index
  //      CHECK: %[[c0:.*]] = arith.constant 0 : index
  //      CHECK: %[[M:.+]] = tensor.dim %[[A]], %[[c0]] :
  //      CHECK: %[[N:.+]] = tensor.dim %[[B]], %[[c1]] :
  //      CHECK: %[[NT0:.+]] = affine.apply #map()[%[[M]]]
  //      CHECK: %[[NT1:.+]] = affine.apply #map1()[%[[N]]]
  //      CHECK: %[[K:.+]] = tensor.dim %[[A]], %[[c1]] :
  //      CHECK: scf.forall (%[[IV0:.+]], %[[IV1:.+]]) in (%[[NT0]], %[[NT1]]) shared_outs(%[[C_BLK:.*]] = %[[C]])
  //      CHECK:   %[[TS0:.+]] = affine.min #[[$map2]](%[[IV0]])[%[[M]]]
  //      CHECK:   %[[TS1:.+]] = affine.min #[[$map3]](%[[IV1]])[%[[N]]]
  //      CHECK:   %[[LB0:.+]] = affine.apply #[[$map4]](%[[IV0]])
  //      CHECK:   %[[LB1:.+]] = affine.apply #[[$map5]](%[[IV1]])
  //      CHECK:   tensor.extract_slice %[[A]][%[[LB0]], 0] [%[[TS0]], %[[K]]] [1, 1] :
  //      CHECK:   tensor.extract_slice %[[B]][0, %[[LB1]]] [%[[K]], %[[TS1]]] [1, 1] :
  //      CHECK:   tensor.extract_slice %[[C_BLK]][%[[LB0]], %[[LB1]]] [%[[TS0]], %[[TS1]]] [1, 1] :
  //      CHECK:   linalg.matmul
  //      CHECK:   scf.forall.in_parallel
  // CHECK-NEXT:    tensor.parallel_insert_slice
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %c10 = transform.param.constant 10 : i64 -> !transform.any_param
    %c20 = transform.param.constant 20 : i64 -> !transform.any_param
    %sz = transform.merge_handles %c10, %c20 : !transform.any_param
    %1:2 = transform.structured.tile_using_forall %0 tile_sizes *(%sz)
           : (!transform.any_op, !transform.any_param) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @matmul_tile_size_dynamic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %sz = transform.param.constant "[10 : i64, 20 : i64]" -> !transform.any_param
    // expected-error @below {{expected the parameter to be associated with an integer attribute}}
    %1:2 = transform.structured.tile_using_forall %0 tile_sizes *(%sz)
           : (!transform.any_op, !transform.any_param) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @tile_thread_safety1(%arg0: tensor<100x300xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
  // expected-warning@below {{tiling is not thread safe at axis #1}}
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<100x300xf32>) outs(%arg1 : tensor<100xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<100xf32>
  return %0 : tensor<100xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %forall, %tiled_generic = transform.structured.tile_using_forall %0 num_threads [4, 2]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>

func.func @tile_thread_safety2(%arg0: tensor<100x300x8xf32>, %arg1: tensor<300x8xf32>) -> tensor<300x8xf32> {
  // expected-warning@below {{tiling is not thread safe at axis #0}}
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel", "parallel"]} ins(%arg0 : tensor<100x300x8xf32>) outs(%arg1 : tensor<300x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<300x8xf32>
  return %0 : tensor<300x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %forall, %tiled_generic = transform.structured.tile_using_forall %0 num_threads [8]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>

func.func @tile_thread_safety3(%arg0: tensor<100x300x8xf32>, %arg1: tensor<100x8xf32>) -> tensor<100x8xf32> {
  // expected-warning@below {{tiling is not thread safe at axis #1}}
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg0 : tensor<100x300x8xf32>) outs(%arg1 : tensor<100x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<100x8xf32>
  return %0 : tensor<100x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %forall, %tiled_generic = transform.structured.tile_using_forall %0 num_threads [8, 4, 2]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>

func.func @tile_thread_safety4(%arg0: tensor<100x300x8xf32>, %arg1: tensor<100x8xf32>, %arg2 : tensor<8xf32>) -> (tensor<100x8xf32>, tensor<8xf32>) {
  // expected-warning@+2 {{tiling is not thread safe at axis #0}}
  // expected-warning@below {{tiling is not thread safe at axis #1}}
  %0:2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel"]} ins(%arg0 : tensor<100x300x8xf32>) outs(%arg1, %arg2 : tensor<100x8xf32>, tensor<8xf32>) {
  ^bb0(%in: f32, %out1: f32, %out2: f32):
    %1 = arith.addf %in, %out1 : f32
    %2 = arith.addf %in, %out2 : f32
    linalg.yield %1, %2 : f32, f32
  } -> (tensor<100x8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<100x8xf32>, tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %forall, %tiled_generic = transform.structured.tile_using_forall %0 num_threads [8, 4, 2]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

func.func @tile_thread_safety5(%arg0: tensor<100x300xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
  // expected-warning@below {{tiling is not thread safe at axis #1}}
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<100x300xf32>) outs(%arg1 : tensor<100xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<100xf32>
  return %0 : tensor<100xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %forall, %tiled_generic = transform.structured.tile_using_forall %0 tile_sizes [10, 1]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @tile_thread_safety6(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-warning@below {{tiling is not thread safe at axis #2}}
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %forall, %tiled_generic = transform.structured.tile_using_forall %0 num_threads [2, 0, 8]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
