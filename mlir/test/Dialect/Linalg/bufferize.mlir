// RUN: mlir-opt --one-shot-bufferize="dialect-filter=linalg,bufferization copy-before-write unknown-type-conversion=identity-layout-map" -canonicalize -cse -split-input-file %s | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>

// In-depth checking of a basic case, this is testing
// - bufferization.to_memref / bufferization.to_tensor materializations are
//   properly inserted
// - payload is correctly carried over
// - affine maps are correctly carried over
// Later tests will not check all these details.

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func @basic(
// CHECK-SAME:                %[[TENSOR:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-DAG:       %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]] : memref<4xf32>
// CHECK-DAG:       %[[RESULT_MEMREF:.*]] = memref.alloc() {{.*}} : memref<4xf32>
// CHECK:           linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
// CHECK-SAME:      ins(%[[MEMREF]] : memref<4xf32>)
// CHECK-SAME:      outs(%[[RESULT_MEMREF]] : memref<4xf32>) {
// CHECK:           ^bb0(%[[RESULT1:.*]]: f32, %[[UNUSED:.*]]: f32):
// CHECK:             %[[DIM1:.*]] = math.exp %[[RESULT1]] : f32
// CHECK:             linalg.yield %[[DIM1]] : f32
// CHECK:           }
// CHECK:           %[[RESULT:.*]] = bufferization.to_tensor %[[RESULT_MEMREF]] : memref<4xf32>
// CHECK:           return %[[RESULT]] : tensor<4xf32>
func.func @basic(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = linalg.generic {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel"]
    } ins(%arg0 : tensor<4xf32>)
      outs(%arg0 : tensor<4xf32>) {
      ^bb0(%gen_arg1: f32, %out: f32):
        %tmp1 = math.exp %gen_arg1 : f32
        linalg.yield %tmp1 : f32
    } -> tensor<4xf32>
    return %0 : tensor<4xf32>
}


// -----

#map0 = affine_map<(d0) -> (d0)>

// Same as above but with tensor.empty op.

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @empty_tensor(
// CHECK-SAME:      %[[IN:.*]]: tensor<?xf32>, %[[SIZE:.*]]: index)
// CHECK-DAG:     %[[MEMREF:.*]] = bufferization.to_memref %[[IN]] : memref<?xf32>
// CHECK-DAG:     %[[OUT_BUF:.*]] = memref.alloc(%[[SIZE]]) {{.*}} : memref<?xf32>
// CHECK:         linalg.generic
// CHECK-SAME:    ins(%[[MEMREF]] : memref<?xf32>)
// CHECK-SAME:    outs(%[[OUT_BUF]] : memref<?xf32>) {
func.func @empty_tensor(%in : tensor<?xf32>, %size: index) -> tensor<?xf32> {
  %init = tensor.empty(%size) : tensor<?xf32>
  %0 = linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]
  } ins(%in : tensor<?xf32>)
    outs(%init : tensor<?xf32>) {
    ^bb0(%gen_arg1: f32, %out: f32):
      %tmp1 = math.exp %gen_arg1 : f32
      linalg.yield %tmp1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}


// -----

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL:   func @multiple_results
// CHECK:           %[[RESULT0:.*]] = memref.alloc() {{.*}} : memref<4xf32>
// CHECK:           %[[RESULT1:.*]] = memref.alloc() {{.*}} : memref<4xf32>
// CHECK:           linalg.generic
// CHECK-SAME:      ins(%{{.*}} : memref<4xf32>)
// CHECK-SAME:      outs(%[[RESULT0]], %[[RESULT1]] : memref<4xf32>, memref<4xf32>)
// CHECK-NEXT: ^bb0(%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: f32):
func.func @multiple_results(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %0, %1 = linalg.generic {
      indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel"]
    } ins(%arg0 : tensor<4xf32>)
      outs (%arg0, %arg0 : tensor<4xf32>, tensor<4xf32>) {
      ^bb0(%gen_arg1: f32, %out1: f32, %out2: f32):
        %tmp1 = math.exp %gen_arg1 : f32
        linalg.yield %tmp1, %tmp1 : f32, f32
    } -> (tensor<4xf32>, tensor<4xf32>)
    return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// -----

#map_2d = affine_map<(d0, d1) -> (d0, d1)>

// Check that the allocs properly consider the different shapes of the output
// operands. The permuted indexing maps translate to different output shapes.

// CHECK-LABEL:   func @dynamic_results(
// CHECK-SAME:                          %[[ARG:.*]]: tensor<?x?xf32>
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[C0]] : tensor<?x?xf32>
// CHECK-DAG:       %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[C1]] : tensor<?x?xf32>
// CHECK-DAG:       %[[RESULT0:.*]] = memref.alloc(%[[DIM0]], %[[DIM1]]) {{.*}} : memref<?x?xf32>
// CHECK-DAG:       %[[RESULT1:.*]] = memref.alloc(%[[DIM0]], %[[DIM1]]) {{.*}} : memref<?x?xf32>
// CHECK-DAG:       %[[MEMREF_ARG:.*]] = bufferization.to_memref %[[ARG]] : memref<?x?xf32>
// CHECK:           linalg.generic
// CHECK-SAME:      ins(%[[MEMREF_ARG]] : memref<?x?xf32>)
// CHECK-SAME:      outs(%[[RESULT0]], %[[RESULT1]] : memref<?x?xf32>, memref<?x?xf32>)
func.func @dynamic_results(%arg0: tensor<?x?xf32>)
         -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    %0, %1 = linalg.generic {
      indexing_maps = [#map_2d, #map_2d, #map_2d],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<?x?xf32>)
      outs (%arg0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>) {
      ^bb0(%gen_arg1: f32, %out1: f32, %out2: f32):
        %tmp1 = math.exp %gen_arg1 : f32
        linalg.yield %tmp1, %tmp1 : f32, f32
    } -> (tensor<?x?xf32>, tensor<?x?xf32>)
    return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

#accesses = [
  affine_map<(i, j, k) -> (j, i, k)>,
  affine_map<(i, j, k) -> (i, j)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// Check the bufferization of init tensors.

// CHECK-LABEL:   func @generic_with_init_tensor(
// CHECK-SAME:                                   %[[ARG0_TENSOR:.*]]: tensor<2x3x4xvector<3x4xi4>>,
// CHECK-SAME:                                   %[[ARG1_TENSOR:.*]]: tensor<3x2xf32>) -> tensor<3x2xf32> {
// CHECK-DAG:       %[[INIT_BUFFER:.*]] = memref.alloc() {{.*}} : memref<3x2xf32>
// CHECK-DAG:       %[[ARG0_MEMREF:.*]] = bufferization.to_memref %[[ARG0_TENSOR]] : memref<2x3x4xvector<3x4xi4>>
// CHECK-DAG:       %[[ARG1_MEMREF:.*]] = bufferization.to_memref %[[ARG1_TENSOR]] : memref<3x2xf32>
// CHECK:           memref.copy %[[ARG1_MEMREF]], %[[INIT_BUFFER]] : memref<3x2xf32> to memref<3x2xf32>
// CHECK:           linalg.generic
// CHECK-SAME:      ins(%[[ARG0_MEMREF]] : memref<2x3x4xvector<3x4xi4>>)
// CHECK-SAME:      outs(%[[INIT_BUFFER]] : memref<3x2xf32>) {
func.func @generic_with_init_tensor(%arg0: tensor<2x3x4xvector<3x4xi4>>,
  %arg1: tensor<3x2xf32>) -> (tensor<3x2xf32>) {

  %0 = linalg.generic #trait
    ins(%arg0 : tensor<2x3x4xvector<3x4xi4>>)
   outs(%arg1 : tensor<3x2xf32>) {
    ^bb(%v0: vector<3x4xi4>, %v1: f32) :
      linalg.yield %v1 : f32
  } -> tensor<3x2xf32>

  return %0 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: func @bufferize_fill(
// CHECK-SAME:    %[[IN:.*]]: tensor<?xf32>
func.func @bufferize_fill(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0.0 : f32
  // CHECK: %[[ALLOC:.*]] = memref.alloc
  // CHECK: linalg.fill ins(%cst : f32) outs(%[[ALLOC]] : memref<?xf32>)
  // CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] : memref<?xf32>
  // CHECK: return %[[TENSOR]]
  %0 = linalg.fill ins(%c0 : f32) outs(%arg0 : tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL:   func @bufferize_dot
func.func @bufferize_dot(%in: tensor<4xf32>, %out: tensor<f32>) -> tensor<f32> {
  %dot = linalg.dot ins(%in, %in : tensor<4xf32>, tensor<4xf32>)
                    outs(%out : tensor<f32>) -> tensor<f32>
  return %dot : tensor<f32>
  // CHECK: %[[ALLOC:.*]] = memref.alloc
  // TODO: The copy is not necessary.
  // CHECK: memref.copy {{.*}}, %[[ALLOC]]
  // CHECK: linalg.dot ins(%{{.*}}, %{{.*}} : memref<4xf32>, memref<4xf32>)
  // CHECK-SAME:       outs(%[[ALLOC:.*]] : memref<f32>)
  // CHECK: %[[OUT_TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] : memref<f32>
  // CHECK: return %[[OUT_TENSOR]]
}

// -----

// CHECK-LABEL: func @bufferize_softmax(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<2x16x32xf32>, %[[arg1:.*]]: tensor<2x16x32xf32>
//       CHECK:   %[[m0:.*]] = bufferization.to_memref %[[arg0]]
//       CHECK:   %[[alloc:.*]] = memref.alloc()
//   CHECK-NOT:   memref.copy
//       CHECK:   linalg.softmax dimension(2) ins(%[[m0]] : {{.*}}) outs(%[[alloc:.*]] : {{.*}})
//       CHECK:   %[[result:.*]] = bufferization.to_tensor %[[alloc]]
//       CHECK:   return %[[result]]
func.func @bufferize_softmax(%arg0: tensor<2x16x32xf32>, %arg1: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
  %1 = linalg.softmax dimension(2)
      ins(%arg0 : tensor<2x16x32xf32>)
      outs(%arg1: tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %1 : tensor<2x16x32xf32>
}
