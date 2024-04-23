// RUN: mlir-opt %s -split-input-file -test-linalg-fuse-consumer | FileCheck %s

#map = affine_map<()[s0] -> (64 ceildiv s0)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0] -> (-(d0 * s0) + 64, s0)>
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
  } {"containing"}
  // CHECK: %[[ORI_OUTPUT:.*]] = linalg.elemwise_binary
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>, "consumer"} ins(%4, %2 : tensor<64xf32>, tensor<64xf32>) outs(%3 : tensor<64xf32>) -> tensor<64xf32>
  // CHECK: return %[[RES]]#1
  return %5 : tensor<64xf32>
}
// -----

#map = affine_map<(d0) -> (d0 * -50 + 123, 50)>
#map1 = affine_map<(d0) -> (d0 * -16 + 789, 16)>
#map2 = affine_map<(d0) -> (d0 * 50)>
#map3 = affine_map<(d0) -> (d0 * 16)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d1, d0)>
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
  } {"containing"}
  // CHECK: %[[ORI_OUTPUT:.*]]:2 = linalg.generic
  %4:2 = linalg.generic {"consumer", indexing_maps = [#map4, #map4, #map5], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<123x789xf32>) outs(%1, %2 : tensor<123x789xf32>, tensor<789x123xf32>) {
  ^bb0(%in: f32, %out: f32, %out_0: f32):
    %5 = arith.addf %in, %out : f32
    %6 = arith.addf %5, %out_0 : f32
    linalg.yield %5, %6 : f32, f32
  } -> (tensor<123x789xf32>, tensor<789x123xf32>)
  // CHECK: return %[[RES]]#1, %[[RES]]#2
  return %4#0, %4#1 : tensor<123x789xf32>, tensor<789x123xf32>
}


