// RUN: mlir-opt -convert-linalg-on-tensors-to-buffers -buffer-placement -split-input-file %s | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @multiple_results
func @multiple_results(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %0, %1 = linalg.generic {
      indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel"]
    } ins(%arg0 : tensor<4xf32>) {
      ^bb0(%gen_arg1: f32):
        %tmp1 = exp %gen_arg1 : f32
        linalg.yield %tmp1, %tmp1 : f32, f32
    } -> tensor<4xf32>, tensor<4xf32>
    return %0, %1 : tensor<4xf32>, tensor<4xf32>
}
//      CHECK: (%[[NEW_ARG0:.*]]: [[TYPE:.*]], %[[ARG1_RESULT:.*]]: [[TYPE]], %[[ARG2_RESULT:.*]]: [[TYPE]])
//      CHECK: %[[FIRST_ALLOC:.*]] = alloc() : [[TYPE]]
//      CHECK: %[[SECOND_ALLOC:.*]] = alloc() : [[TYPE]]
//      CHECK: linalg.generic
// CHECK-SAME: ins(%[[NEW_ARG0]] : [[TYPE]]
// CHECK-SAME: outs(%[[FIRST_ALLOC]], %[[SECOND_ALLOC]] : [[TYPE]], [[TYPE]]
// CHECK-NEXT: ^{{[a-z0-9_]*}}
// CHECK-SAME: %{{.*}}: f32, %{{.*}}: f32, %{{.*}}: f32
// CHECK-NEXT: %{{.*}} = exp
// CHECK-NEXT: linalg.yield
//      CHECK: linalg.copy(%[[FIRST_ALLOC]], %[[ARG1_RESULT]])
//      CHECK: dealloc %[[FIRST_ALLOC]]
//      CHECK: linalg.copy(%[[SECOND_ALLOC]], %[[ARG2_RESULT]])
//      CHECK: dealloc %[[SECOND_ALLOC]]
//      CHECK: return

// -----

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @chained_operations
func @chained_operations(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = linalg.generic {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel"]
    } ins(%arg0 : tensor<4xf32>) {
      ^bb0(%gen_arg1: f32):
        %tmp1 = exp %gen_arg1 : f32
        linalg.yield %tmp1 : f32
    } -> tensor<4xf32>

    %1 = linalg.generic {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel"]
    } ins(%0 : tensor<4xf32>) {
      ^bb0(%gen_arg2: f32):
        %tmp2 = exp %gen_arg2 : f32
        linalg.yield %tmp2 : f32
    } -> tensor<4xf32>
    return %1 : tensor<4xf32>
}
//      CHECK: (%[[NEW_ARG0:.*]]: [[TYPE:.*]], %[[ARG1_RESULT:.*]]: [[TYPE]])
//      CHECK: %[[FIRST_ALLOC:.*]] = alloc() : [[TYPE]]
//      CHECK: linalg.generic
// CHECK-SAME: ins(%[[NEW_ARG0]] : [[TYPE]]
// CHECK-SAME: outs(%[[FIRST_ALLOC]] : [[TYPE]]
//      CHECK: ^{{[a-z0-9_]*}}
// CHECK-SAME: %{{.*}}: f32, %{{.*}}: f32
//      CHECK: %[[SECOND_ALLOC:.*]] = alloc() : [[TYPE]]
//      CHECK: linalg.generic
// CHECK-SAME: ins(%[[FIRST_ALLOC]] : [[TYPE]]
// CHECK-SAME: outs(%[[SECOND_ALLOC]] : [[TYPE]]
//      CHECK: ^{{[a-z0-9_]*}}
// CHECK-SAME: %{{.*}}: f32, %{{.*}}: f32
//      CHECK: dealloc %[[FIRST_ALLOC]]
//      CHECK: linalg.copy(%[[SECOND_ALLOC]], %[[ARG1_RESULT]])
//      CHECK: dealloc %[[SECOND_ALLOC]]
//      CHECK: return

// -----

// CHECK-LABEL: func @no_linalg_op
func @no_linalg_op(%arg0: f32) -> (f32, f32) {
  %0 = mulf %arg0, %arg0 : f32
  return %0, %0 : f32, f32
}
// CHECK: (%[[NEW_ARG0:.*]]: [[TYPE:.*]]) -> ([[TYPE]], [[TYPE]])
// CHECK: %[[RESULT:.*]] = mulf %[[NEW_ARG0]], %[[NEW_ARG0]] : [[TYPE]]
// CHECK: return %[[RESULT]], %[[RESULT]] : [[TYPE]], [[TYPE]]

// -----

#map_2d = affine_map<(d0, d1) -> (d0, d1)>
#map_2d_inv = affine_map<(d0, d1) -> (d1, d0)>

func @dynamic_results(%arg0: tensor<?x?xf32>)
         -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    %0, %1 = linalg.generic {
      indexing_maps = [#map_2d, #map_2d, #map_2d_inv],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<?x?xf32>) {
      ^bb0(%gen_arg1: f32):
        %tmp1 = exp %gen_arg1 : f32
        linalg.yield %tmp1, %tmp1 : f32, f32
    } -> tensor<?x?xf32>, tensor<?x?xf32>
    return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}

// CHECK: #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func @dynamic_results
// CHECK-SAME: (%[[INPUT:.*]]: [[TYPE:.*]], %[[OUT_1:.*]]: [[TYPE]], %[[OUT_2:.*]]: [[TYPE]]) {
// CHECK: %[[C0:.*]] = constant 0 : index
// CHECK: %[[DIM_0:.*]] = dim %[[INPUT]], %[[C0]] : [[TYPE]]
// CHECK: %[[C1:.*]] = constant 1 : index
// CHECK: %[[DIM_1:.*]] = dim %[[INPUT]], %[[C1]] : [[TYPE]]
// CHECK: %[[OUT_BUF_1:.*]] = alloc(%[[DIM_0]], %[[DIM_1]]) : [[TYPE]]
// CHECK: %[[OUT_BUF_2:.*]] = alloc(%[[DIM_1]], %[[DIM_0]]) : [[TYPE]]

// CHECK: linalg.generic {indexing_maps = [#map0, #map0, #map1], {{.*}}}
// CHECK-SAME: ins(%[[INPUT]] : [[TYPE]])
// CHECK-SAME: outs(%[[OUT_BUF_1]], %[[OUT_BUF_2]] : [[TYPE]], [[TYPE]]) {

// CHECK: linalg.copy(%[[OUT_BUF_1]], %[[OUT_1]]) : [[TYPE]], [[TYPE]]
// CHECK: dealloc %[[OUT_BUF_1]] : [[TYPE]]
// CHECK: linalg.copy(%[[OUT_BUF_2]], %[[OUT_2]]) : [[TYPE]], [[TYPE]]
// CHECK: dealloc %[[OUT_BUF_2]] : [[TYPE]]
// CHECK: return

// -----

func @foo() -> tensor<4xf32> {
// CHECK-LABEL: func @foo(
//  CHECK-SAME:   %[[A:[0-9a-z]*]]: memref<4xf32>) {

  %0 = constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
//  CHECK-NEXT:   %[[CST:.*]] = constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : vector<4xf32>
//  CHECK-NEXT:   %[[ALLOC:.*]] = alloc() : memref<vector<4xf32>>
//  CHECK-NEXT:   store %[[CST]], %[[ALLOC]][] : memref<vector<4xf32>>
//  CHECK-NEXT:   %[[RES:.*]] = vector.type_cast %[[ALLOC]] : memref<vector<4xf32>> to memref<4xf32>

  return %0 : tensor<4xf32>
//  CHECK-NEXT:   linalg.copy(%[[RES]], %[[A]]) : memref<4xf32>, memref<4xf32>
//  CHECK-NEXT:   dealloc %[[ALLOC]] : memref<vector<4xf32>>
//  CHECK-NEXT:   return
}

func @bar() {
// CHECK-LABEL: func @bar() {

  %0 = call @foo() : () -> tensor<4xf32>
//  CHECK-NEXT:   %[[ALLOC:.*]] = alloc() : memref<4xf32>
//  CHECK-NEXT:   call @foo(%[[ALLOC]]) : (memref<4xf32>) -> ()

  // Instead of relying on tensor_store which introduces aliasing, we rely on
  // the conversion of print_memref_f32(tensor<*xf32>) to
  // print_memref_f32(memref<*xf32>).
  // Note that this is skipping a step and we would need at least some function
  // attribute to declare that this conversion is valid (e.g. when we statically
  // know that things will play nicely at the C ABI boundary).
  %unranked = tensor_cast %0 : tensor<4xf32> to tensor<*xf32>
//  CHECK-NEXT:   %[[UNRANKED:.*]] = memref_cast %[[ALLOC]] :
//  CHECK-SAME:     memref<4xf32> to memref<*xf32>

  call @print_memref_f32(%unranked) : (tensor<*xf32>) -> ()
//  CHECK-NEXT:   call @print_memref_f32(%[[UNRANKED]]) : (memref<*xf32>) -> ()

  return
//  CHECK-NEXT:   dealloc %[[ALLOC]] : memref<4xf32>
//  CHECK-NEXT:   return
}

// This gets converted to a function operating on memref<*xf32>.
// Note that this is skipping a step and we would need at least some function
// attribute to declare that this conversion is valid (e.g. when we statically
// know that things will play nicely at the C ABI boundary).
func @print_memref_f32(%ptr : tensor<*xf32>)
// CHECK-LABEL: func @print_memref_f32(memref<*xf32>)

// -----

#accesses = [
  affine_map<(i, j, k) -> (j, i, k)>,
  affine_map<(i, j, k) -> (i, j)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

func @generic_with_init_tensor(%arg0: tensor<2x3x4xvector<3x4xi4>>,
  %arg1: tensor<3x2xf32>) -> (tensor<3x2xf32>) {

  %0 = linalg.generic #trait
    ins(%arg0 : tensor<2x3x4xvector<3x4xi4>>)
   init(%arg1 : tensor<3x2xf32>) {
    ^bb(%v0: vector<3x4xi4>, %v1: f32) :
      %f0 = constant 0.0 : f32
      linalg.yield %f0 : f32
  } -> tensor<3x2xf32>

  return %0 : tensor<3x2xf32>
}
// CHECK-LABEL: func @generic_with_init_tensor
//  CHECK-SAME: (%[[ARG0:.*]]: memref<2x3x4xvector<3x4xi4>>, %[[ARG1:.*]]: memref<3x2xf32>, %[[RESULT0:.*]]: memref<3x2xf32>) {
//  CHECK-NEXT: linalg.generic
//       CHECK: linalg.copy(%[[ARG1]], %[[RESULT0]])
//  CHECK-NEXT: return
//   CHECK-NOT: %

// -----

#accesses = [
  affine_map<(i, j, k) -> (j, i, k)>,
  affine_map<(i, j, k) -> (i, j)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

func @init_tensor_with_2_uses(%arg0: tensor<2x3x4xvector<3x4xi4>>,
  %arg1: tensor<3x2xf32>) -> (tensor<3x2xf32>, tensor<3x2xf32>) {

  %0 = linalg.generic #trait
    ins(%arg0 : tensor<2x3x4xvector<3x4xi4>>)
   init(%arg1 : tensor<3x2xf32>) {
    ^bb(%v0: vector<3x4xi4>, %v1: f32) :
      %f0 = constant 0.0 : f32
      linalg.yield %f0 : f32
  } -> tensor<3x2xf32>

  %1 = linalg.generic #trait
    ins(%arg0 : tensor<2x3x4xvector<3x4xi4>>)
   init(%arg1 : tensor<3x2xf32>) {
    ^bb(%v0: vector<3x4xi4>, %v1: f32) :
      %f0 = constant 0.0 : f32
      linalg.yield %f0 : f32
  } -> tensor<3x2xf32>

  return %0, %1 : tensor<3x2xf32>, tensor<3x2xf32>
}
// CHECK-LABEL: func @init_tensor_with_2_uses
//  CHECK-SAME: (%[[ARG0:.*]]: memref<2x3x4xvector<3x4xi4>>, %[[ARG1:.*]]: memref<3x2xf32>, %[[RESULT0:.*]]: memref<3x2xf32>, %[[RESULT1:.*]]: memref<3x2xf32>) {
//  CHECK-NEXT: %[[ALLOC0:.*]] = alloc
//  CHECK-NEXT: linalg.copy(%[[ARG1]], %[[ALLOC0]])
//  CHECK-NEXT: linalg.generic
//  CHECK-SAME: outs(%[[ALLOC0]]
//  CHECK-NEXT: ^bb
//  CHECK-NEXT:   constant
//  CHECK-NEXT:   yield
//  CHECK-NEXT: }
//  CHECK-NEXT: %[[ALLOC1:.*]] = alloc
//  CHECK-NEXT: linalg.copy(%[[ARG1]], %[[ALLOC1]])
//  CHECK-NEXT: linalg.generic
//  CHECK-SAME: outs(%[[ALLOC1]]
//  CHECK-NEXT: ^bb
//  CHECK-NEXT:   constant
//  CHECK-NEXT:   yield
//  CHECK-NEXT: }
//  CHECK-NEXT: linalg.copy(%[[ALLOC0]], %[[RESULT0]])
//  CHECK-NEXT: dealloc
//  CHECK-NEXT: linalg.copy(%[[ALLOC1]], %[[RESULT1]])
//  CHECK-NEXT: dealloc
//  CHECK-NEXT: return
//   CHECK-NOT: %

// -----

#accesses = [
  affine_map<(i, j, k) -> (j, i, k)>,
  affine_map<(i, j, k) -> (i, j)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

func @init_tensor_with_1_use_def_chain(%arg0: tensor<2x3x4xvector<3x4xi4>>,
  %arg1: tensor<3x2xf32>) -> (tensor<3x2xf32>) {

  %0 = linalg.generic #trait
    ins(%arg0 : tensor<2x3x4xvector<3x4xi4>>)
   init(%arg1 : tensor<3x2xf32>) {
    ^bb(%v0: vector<3x4xi4>, %v1: f32) :
      %f0 = constant 0.0 : f32
      linalg.yield %f0 : f32
  } -> tensor<3x2xf32>

  %1 = linalg.generic #trait
    ins(%arg0 : tensor<2x3x4xvector<3x4xi4>>)
   init(%0 : tensor<3x2xf32>) {
    ^bb(%v0: vector<3x4xi4>, %v1: f32) :
      %f0 = constant 0.0 : f32
      linalg.yield %f0 : f32
  } -> tensor<3x2xf32>

  return %1 : tensor<3x2xf32>
}
// CHECK-LABEL: func @init_tensor_with_1_use_def_chain
//  CHECK-SAME: (%[[ARG0:.*]]: memref<2x3x4xvector<3x4xi4>>, %[[ARG1:.*]]: memref<3x2xf32>, %[[RESULT0:.*]]: memref<3x2xf32>) {
//  CHECK-NEXT: linalg.generic
//  CHECK-NEXT: ^bb
//  CHECK-NEXT:   constant
//  CHECK-NEXT:   yield
//  CHECK-NEXT: }
//  CHECK-NEXT: linalg.generic
//  CHECK-NEXT: ^bb
//  CHECK-NEXT:   constant
//  CHECK-NEXT:   yield
//  CHECK-NEXT: }
//  CHECK-NEXT: linalg.copy(%[[ARG1]], %[[RESULT0]])
//  CHECK-NEXT: return
//   CHECK-NOT: %
