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
