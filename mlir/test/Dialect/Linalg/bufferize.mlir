// RUN: mlir-opt -linalg-bufferize  -canonicalize -cse -split-input-file %s | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>

// In-depth checking of a basic case, this is testing
// - memref.buffer_cast / memref.tensor_load materializations are properly inserted
// - payload is correctly carried over
// - affine maps are correctly carried over
// Later tests will not check all these details.

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func @basic(
// CHECK-SAME:                %[[TENSOR:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:           %[[MEMREF:.*]] = memref.buffer_cast %[[TENSOR]] : memref<4xf32>
// CHECK:           %[[RESULT_MEMREF:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
// CHECK-SAME:      ins(%[[MEMREF]] : memref<4xf32>)
// CHECK-SAME:      outs(%[[RESULT_MEMREF]] : memref<4xf32>) {
// CHECK:           ^bb0(%[[RESULT1:.*]]: f32, %[[UNUSED:.*]]: f32):
// CHECK:             %[[DIM1:.*]] = math.exp %[[RESULT1]] : f32
// CHECK:             linalg.yield %[[DIM1]] : f32
// CHECK:           }
// CHECK:           %[[RESULT:.*]] = memref.tensor_load %[[RESULT_MEMREF]] : memref<4xf32>
// CHECK:           return %[[RESULT]] : tensor<4xf32>
func @basic(%arg0: tensor<4xf32>) -> tensor<4xf32> {
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

// Same as above but with linalg.init_tensor op.

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @init_tensor(
// CHECK-SAME:      %[[IN:.*]]: tensor<?xf32>, %[[SIZE:.*]]: index)
// CHECK:         %[[MEMREF:.*]] = memref.buffer_cast %[[IN]] : memref<?xf32>
// CHECK:         %[[OUT_BUF:.*]] = memref.alloc(%[[SIZE]]) : memref<?xf32>
// CHECK:         linalg.generic
// CHECK-SAME:    ins(%[[MEMREF]] : memref<?xf32>)
// CHECK-SAME:    outs(%[[OUT_BUF]] : memref<?xf32>) {
func @init_tensor(%in : tensor<?xf32>, %size: index) -> tensor<?xf32> {
  %init = linalg.init_tensor [%size] : tensor<?xf32>
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
// CHECK:           %[[RESULT0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           %[[RESULT1:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           linalg.generic
// CHECK-SAME:      ins(%{{.*}} : memref<4xf32>)
// CHECK-SAME:      outs(%[[RESULT0]], %[[RESULT1]] : memref<4xf32>, memref<4xf32>)
// CHECK-NEXT: ^bb0(%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: f32):
func @multiple_results(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
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
// CHECK-DAG:       %[[C0:.*]] = constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = constant 1 : index
// CHECK:           %[[MEMREF_ARG:.*]] = memref.buffer_cast %[[ARG]] : memref<?x?xf32>
// CHECK:           %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[RESULT0:.*]] = memref.alloc(%[[DIM0]], %[[DIM1]]) : memref<?x?xf32>
// CHECK:           %[[RESULT1:.*]] = memref.alloc(%[[DIM0]], %[[DIM1]]) : memref<?x?xf32>
// CHECK:           linalg.generic
// CHECK-SAME:      ins(%[[MEMREF_ARG]] : memref<?x?xf32>)
// CHECK-SAME:      outs(%[[RESULT0]], %[[RESULT1]] : memref<?x?xf32>, memref<?x?xf32>)
func @dynamic_results(%arg0: tensor<?x?xf32>)
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
// CHECK:           %[[ARG0_MEMREF:.*]] = memref.buffer_cast %[[ARG0_TENSOR]] : memref<2x3x4xvector<3x4xi4>>
// CHECK:           %[[ARG1_MEMREF:.*]] = memref.buffer_cast %[[ARG1_TENSOR]] : memref<3x2xf32>
// CHECK:           %[[INIT_BUFFER:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:           linalg.copy(%[[ARG1_MEMREF]], %[[INIT_BUFFER]]) : memref<3x2xf32>, memref<3x2xf32>
// CHECK:           linalg.generic
// CHECK-SAME:      ins(%[[ARG0_MEMREF]] : memref<2x3x4xvector<3x4xi4>>)
// CHECK-SAME:      outs(%[[INIT_BUFFER]] : memref<3x2xf32>) {
func @generic_with_init_tensor(%arg0: tensor<2x3x4xvector<3x4xi4>>,
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

// CHECK-DAG: #[[$MAP0:[0-9a-z]*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[$MAP1:[0-9a-z]*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1 * 2)>

func private @make_index() -> index

// CHECK-LABEL: func @bufferize_slice(
//  CHECK-SAME:   %[[T:[0-9a-z]*]]: tensor<?x?xf32>
func @bufferize_slice(%t : tensor<?x?xf32>) -> (tensor<2x3xf32>, tensor<2x?xf32>) {
  //      CHECK: %[[IDX:.*]] = call @make_index() : () -> index
  %i0 = call @make_index() : () -> index

  //      CHECK: %[[M:.*]] = memref.buffer_cast %[[T]] : memref<?x?xf32>
  // CHECK-NEXT: %[[A0:.*]] = memref.alloc() : memref<2x3xf32>
  // CHECK-NEXT: %[[SM0:.*]] = memref.subview %[[M]][0, 0] [2, 3] [1, 1]
  // CHECK-SAME:   memref<?x?xf32> to memref<2x3xf32, #[[$MAP0]]>
  // CHECK-NEXT: linalg.copy(%[[SM0]], %[[A0]]) : memref<2x3xf32, #[[$MAP0]]>, memref<2x3xf32>
  // CHECK-NEXT: %[[RT0:.*]] = memref.tensor_load %[[A0]] : memref<2x3xf32>
  %st0 = tensor.extract_slice %t[0, 0][2, 3][1, 1] : tensor<?x?xf32> to tensor<2x3xf32>

  // CHECK-NEXT: %[[A1:.*]] = memref.alloc(%[[IDX]]) : memref<2x?xf32>
  // CHECK-NEXT: %[[SM1:.*]] = memref.subview %[[M]][0, %[[IDX]]] [2, %[[IDX]]] [1, 2]
  // CHECK-SAME:   memref<?x?xf32> to memref<2x?xf32, #[[$MAP1]]>
  // CHECK-NEXT: linalg.copy(%[[SM1]], %[[A1]]) : memref<2x?xf32, #[[$MAP1]]>, memref<2x?xf32>
  // CHECK-NEXT: %[[RT1:.*]] = memref.tensor_load %[[A1]] : memref<2x?xf32>
  %st1 = tensor.extract_slice %t[0, %i0][2, %i0][1, 2] : tensor<?x?xf32> to tensor<2x?xf32>

  // CHECK-NEXT: return %[[RT0]], %[[RT1]]
  return %st0, %st1 : tensor<2x3xf32>, tensor<2x?xf32>
}

// -----

// CHECK-DAG: #[[$MAP0:[0-9a-z]*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[$MAP1:[0-9a-z]*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1 * 2)>

func private @make_index() -> index

// CHECK-LABEL: func @bufferize_insert_slice(
//  CHECK-SAME:   %[[T:[0-9a-z]*]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ST0:[0-9a-z]*]]: tensor<2x3xf32>
//  CHECK-SAME:   %[[ST1:[0-9a-z]*]]: tensor<2x?xf32>
func @bufferize_insert_slice(%t : tensor<?x?xf32>, %st0 : tensor<2x3xf32>, %st1 : tensor<2x?xf32>) ->
    (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  // CHECK-DAG: %[[C0:.*]] = constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = constant 1 : index
  %i0 = call @make_index() : () -> index
  // CHECK: %[[IDX:.*]] = call @make_index() : () -> index


  // CHECK-DAG: %[[M:.*]] = memref.buffer_cast %[[T]] : memref<?x?xf32>
  // CHECK-DAG: %[[SM0:.*]] = memref.buffer_cast %[[ST0]] : memref<2x3xf32>
  // CHECK-NEXT: %[[DIM0:.*]] = tensor.dim %[[T]], %[[C0]] : tensor<?x?xf32>
  // CHECK-NEXT: %[[DIM1:.*]] = tensor.dim %[[T]], %[[C1]] : tensor<?x?xf32>
  // CHECK-NEXT: %[[M_COPY0:.*]] = memref.alloc(%[[DIM0]], %[[DIM1]]) : memref<?x?xf32>
  // CHECK-NEXT: linalg.copy(%[[M]], %[[M_COPY0]]) : memref<?x?xf32>, memref<?x?xf32>
  // CHECK-NEXT: %[[SUBVIEW0:.*]] = memref.subview %[[M_COPY0]][0, 0] [2, 3] [1, 1]
  // CHECK-SAME:   memref<?x?xf32> to memref<2x3xf32, #[[$MAP0]]>
  // CHECK-NEXT: linalg.copy(%[[SM0]], %[[SUBVIEW0]]) : memref<2x3xf32>, memref<2x3xf32, #[[$MAP0]]>
  // CHECK-NEXT: %[[RT0:.*]] = memref.tensor_load %[[M_COPY0]] : memref<?x?xf32>
  %t0 = tensor.insert_slice %st0 into %t[0, 0][2, 3][1, 1] : tensor<2x3xf32> into tensor<?x?xf32>

  //  CHECK-DAG: %[[SM1:.*]] = memref.buffer_cast %[[ST1]] : memref<2x?xf32>
  // CHECK-NEXT: %[[M_COPY1:.*]] = memref.alloc(%[[DIM0]], %[[DIM1]]) : memref<?x?xf32>
  // CHECK-NEXT: linalg.copy(%[[M]], %[[M_COPY1]]) : memref<?x?xf32>, memref<?x?xf32>
  // CHECK-NEXT: %[[SUBVIEW1:.*]] = memref.subview %[[M_COPY1]][0, %[[IDX]]] [2, %[[IDX]]] [1, 2]
  // CHECK-SAME:   memref<?x?xf32> to memref<2x?xf32, #[[$MAP1]]>
  // CHECK-NEXT: linalg.copy(%[[SM1]], %[[SUBVIEW1]]) : memref<2x?xf32>, memref<2x?xf32, #[[$MAP1]]>
  // CHECK-NEXT: %[[RT1:.*]] = memref.tensor_load %[[M_COPY1]] : memref<?x?xf32>
  %t1 = tensor.insert_slice %st1 into %t[0, %i0][2, %i0][1, 2] : tensor<2x?xf32> into tensor<?x?xf32>

  //     CHECK: return %[[RT0]], %[[RT1]]
  return %t0, %t1: tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @bufferize_fill(
// CHECK-SAME:    %[[IN:.*]]: tensor<?xf32>
func @bufferize_fill(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = constant 0.0 : f32
  // CHECK: %[[MEMREF:.*]] = memref.buffer_cast %[[IN]] : memref<?xf32>
  // CHECK: linalg.fill(%cst, %[[MEMREF]]) : f32, memref<?xf32>
  // CHECK: %[[TENSOR:.*]] = memref.tensor_load %[[MEMREF]] : memref<?xf32>
  // CHECK: return %[[TENSOR]]
  %0 = linalg.fill(%c0, %arg0) : f32, tensor<?xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @bufferize_tensor_collapse_shape(
// CHECK-SAME:    %[[IN:.*]]: tensor<4x5xf32>
func @bufferize_tensor_collapse_shape(%arg0: tensor<4x5xf32>) -> tensor<20xf32> {
  %out = linalg.tensor_collapse_shape %arg0 [[0, 1]] :
     tensor<4x5xf32> into tensor<20xf32>
  return %out : tensor<20xf32>
}
// CHECK: %[[MEMREF:.*]] = memref.buffer_cast %[[IN]] : memref<4x5xf32>
// CHECK: %[[RESHAPE:.*]] = memref.collapse_shape %[[MEMREF]] {{\[}}[0, 1]]
// CHECK-SAME: : memref<4x5xf32> into memref<20xf32>
// CHECK: %[[TENSOR:.*]] = memref.tensor_load %[[RESHAPE]] : memref<20xf32>
// CHECK: return %[[TENSOR]]

// -----

// CHECK-LABEL:   func @pad_tensor_dynamic_shape(
// CHECK-SAME:                                   %[[IN:.*]]: tensor<4x?x2x?xf32>,
// CHECK-SAME:                                   %[[OFFSET:.*]]: index) -> tensor<4x?x?x?xf32> {
func @pad_tensor_dynamic_shape(%arg0: tensor<4x?x2x?xf32>, %arg1: index) -> tensor<4x?x?x?xf32> {
  %c0 = constant 0 : index
  %cst = constant 0.0 : f32
  %out = linalg.pad_tensor %arg0 low[%c0, %c0, %arg1, %c0] high[%c0, %c0, %c0, %arg1]  {
  ^bb0(%gen_arg1: index, %gen_arg2: index, %gen_arg3: index, %gen_arg4: index):  // no predecessors
    linalg.yield %cst : f32
  } : tensor<4x?x2x?xf32> to tensor<4x?x?x?xf32>
  return %out : tensor<4x?x?x?xf32>
}

// CHECK:           %[[C3:.*]] = constant 3 : index
// CHECK:           %[[C2:.*]] = constant 2 : index
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           %[[CST:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[DIM1:.*]] = tensor.dim %[[IN]], %[[C1]] : tensor<4x?x2x?xf32>
// CHECK:           %[[OUT_DIM2:.*]] = addi %[[OFFSET]], %[[C2]] : index
// CHECK:           %[[DIM3:.*]] = tensor.dim %[[IN]], %[[C3]] : tensor<4x?x2x?xf32>
// CHECK:           %[[OUT_DIM3:.*]] = addi %[[DIM3]], %[[OFFSET]] : index
// CHECK:           %[[FILLED:.*]] = memref.alloc(%[[DIM1]], %[[OUT_DIM2]], %[[OUT_DIM3]]) : memref<4x?x?x?xf32>
// CHECK:           linalg.fill(%[[CST]], %[[FILLED]]) : f32, memref<4x?x?x?xf32>
// CHECK:           %[[IN_MEMREF:.*]] = memref.buffer_cast %[[IN]] : memref<4x?x2x?xf32>
// CHECK:           %[[OUT:.*]] = memref.alloc(%[[DIM1]], %[[OUT_DIM2]], %[[OUT_DIM3]]) : memref<4x?x?x?xf32>
// CHECK:           linalg.copy(%[[FILLED]], %[[OUT]]) : memref<4x?x?x?xf32>, memref<4x?x?x?xf32>
// CHECK:           %[[INTERIOR:.*]] = memref.subview %[[OUT]][0, 0, %[[OFFSET]], 0] [4, %[[DIM1]], 2, %[[DIM3]]] [1, 1, 1, 1] : memref<4x?x?x?xf32> to memref<4x?x2x?xf32, #map>
// CHECK:           linalg.copy(%[[IN_MEMREF]], %[[INTERIOR]]) : memref<4x?x2x?xf32>, memref<4x?x2x?xf32, #map>
// CHECK:           %[[OUT_TENSOR:.*]] = memref.tensor_load %[[OUT]] : memref<4x?x?x?xf32>
// CHECK:           return %[[OUT_TENSOR]] : tensor<4x?x?x?xf32>
// CHECK:         }


// -----

// CHECK-LABEL:   func @vector_transfer
func @vector_transfer(%in: tensor<4xf32>, %out: tensor<4xf32>) {
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f32
  %read = vector.transfer_read %in[%c0], %cst {in_bounds = [true]}
      : tensor<4xf32>, vector<4xf32>
  %tanh = math.tanh %read : vector<4xf32>
  %write = vector.transfer_write %tanh, %out[%c0] {in_bounds = [true]}
      : vector<4xf32>, tensor<4xf32>
  return
  // CHECK: vector.transfer_read {{.*}} : memref<4xf32>, vector<4xf32>
  // CHECK: vector.transfer_write {{.*}} : vector<4xf32>, memref<4xf32>
}

// -----

//      CHECK:  func @tiled_dot
func @tiled_dot(%A: tensor<10xf32>, %B: tensor<10xf32>,
                %C: tensor<f32>) -> tensor<f32> {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c10 = constant 10 : index

  %dot = linalg.tiled_loop (%i) = (%c0) to (%c10) step (%c2)
       ins (%A_ = %A: tensor<10xf32>, %B_ = %B: tensor<10xf32>)
       outs (%C_ = %C: tensor<f32>)
       iterators["reduction"] {
    %A_sub = tensor.extract_slice %A_[%i] [%c2] [1]
      : tensor<10xf32> to tensor<?xf32>
    %B_sub = tensor.extract_slice %B_[%i] [%c2] [1]
      : tensor<10xf32> to tensor<?xf32>
    %dot_sub = linalg.dot ins(%A_sub, %B_sub : tensor<?xf32>, tensor<?xf32>)
                          outs(%C_ : tensor<f32>) -> tensor<f32>
    linalg.yield %dot_sub : tensor<f32>
  }
  // CHECK: linalg.tiled_loop
  // CHECK-SAME: ins (%[[A:.*]] = %{{.*}}: memref<10xf32>, %[[B:.*]] = %{{.*}}: memref<10xf32>)
  // CHECK-SAME: outs (%[[C:.*]] = %{{.*}}: memref<f32>)
  //   CHECK-NOT:   alloc
  //   CHECK:       %[[SV_A:.*]] = memref.subview %[[A]]
  //   CHECK:       %[[SV_B:.*]] = memref.subview %[[B]]
  //   CHECK:       linalg.dot ins(%[[SV_A]], %[[SV_B]]
  //   CHECK-SAME:             outs(%[[C]] : memref<f32>)
  //   CHECK:   linalg.yield
  return %dot : tensor<f32>
}
