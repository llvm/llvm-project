// RUN: mlir-opt  %s --transform-interpreter -verify-diagnostics \
// RUN:     --split-input-file | FileCheck %s

//     CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
//     CHECK: func.func @convert_affine_matmul_buffer(%arg0: memref<16x8xf32>, %arg1: memref<8x32xf32>, %arg2: memref<16x32xf32>)
// CHECK-DAG:   affine.for %[[arg3:.*]] = 0 to 16
// CHECK-DAG:   affine.for %[[arg4:.*]] = 0 to 32
// CHECK-DAG:   affine.for %[[arg5:.*]] = 0 to 8
// CHECK-DAG:   %[[v0:.*]] = affine.apply #[[MAP]](%[[arg3]])
// CHECK-DAG:   %[[v1:.*]] = affine.apply #[[MAP]](%[[arg5]])
// CHECK-DAG:   %[[v2:.*]] = affine.load %arg0[%[[v0]], %[[v1]]] : memref<16x8xf32>
// CHECK-DAG:   %[[v3:.*]] = affine.apply #[[MAP]](%[[arg5]])
// CHECK-DAG:   %[[v4:.*]] = affine.apply #[[MAP]](%[[arg4]])
// CHECK-DAG:   %[[v5:.*]] = affine.load %arg1[%[[v3]], %[[v4]]] : memref<8x32xf32>
// CHECK-DAG:   %[[v6:.*]] = affine.apply #[[MAP]](%[[arg3]])
// CHECK-DAG:   %[[v7:.*]] = affine.apply #[[MAP]](%[[arg4]])
// CHECK-DAG:   %[[v8:.*]] = affine.load %arg2[%[[v6]], %[[v7]]] : memref<16x32xf32>
// CHECK-DAG:   %[[v9:.*]] = affine.apply #[[MAP]](%[[arg3]])
// CHECK-DAG:   %[[v10:.*]] = affine.apply #[[MAP]](%[[arg4]])
// CHECK-DAG:   %[[v11:.*]] = arith.mulf %[[v2]], %[[v5]] : f32
// CHECK-DAG:   %[[v12:.*]] = arith.addf %[[v8]], %[[v11]] : f32
//     CHECK:   affine.store %[[v12]], %arg2[%[[v9]], %[[v10]]] : memref<16x32xf32>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @convert_affine_matmul_buffer(%arg0: memref<16x8xf32>, %arg1: memref<8x32xf32>, %arg2: memref<16x32xf32>) {
  linalg.generic
    {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : memref<16x8xf32>, memref<8x32xf32>) outs(%arg2 : memref<16x32xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %0 = arith.mulf %in, %in_0 : f32
        %1 = arith.addf %out, %0 : f32
        linalg.yield %1 : f32
    }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.linalg_to_affine %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

//     CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
//     CHECK: func.func @convert_affine_tc_buffer(%arg0: memref<16x8x4xf32>, %arg1: memref<8x4x32xf32>, %arg2: memref<16x32xf32>)
// CHECK-DAG:   affine.for %[[arg3:.*]] = 0 to 16
// CHECK-DAG:   affine.for %[[arg4:.*]] = 0 to 32
// CHECK-DAG:   affine.for %[[arg5:.*]] = 0 to 8
// CHECK-DAG:   affine.for %[[arg6:.*]] = 0 to 4
// CHECK-DAG:   %[[v0:.*]] = affine.apply #[[MAP]](%[[arg3]])
// CHECK-DAG:   %[[v1:.*]] = affine.apply #[[MAP]](%[[arg5]])
// CHECK-DAG:   %[[v2:.*]] = affine.apply #[[MAP]](%[[arg6]])
// CHECK-DAG:   %[[v3:.*]] = affine.load %arg0[%[[v0]], %[[v1]], %[[v2]]] : memref<16x8x4xf32>
// CHECK-DAG:   %[[v4:.*]] = affine.apply #[[MAP]](%[[arg5]])
// CHECK-DAG:   %[[v5:.*]] = affine.apply #[[MAP]](%[[arg6]])
// CHECK-DAG:   %[[v6:.*]] = affine.apply #[[MAP]](%[[arg4]])
// CHECK-DAG:   %[[v7:.*]] = affine.load %arg1[%[[v4]], %[[v5]], %[[v6]]] : memref<8x4x32xf32>
// CHECK-DAG:   %[[v8:.*]] = affine.apply #[[MAP]](%[[arg3]])
// CHECK-DAG:   %[[v9:.*]] = affine.apply #[[MAP]](%[[arg4]])
// CHECK-DAG:   %[[v10:.*]] = affine.load %arg2[%[[v8]], %[[v9]]] : memref<16x32xf32>
// CHECK-DAG:   %[[v11:.*]] = affine.apply #[[MAP]](%[[arg3]])
// CHECK-DAG:   %[[v12:.*]] = affine.apply #[[MAP]](%[[arg4]])
// CHECK-DAG:   %[[v13:.*]] = arith.mulf %[[v3]], %[[v7]] : f32
// CHECK-DAG:   %[[v14:.*]] = arith.addf %[[v10]], %[[v13]] : f32
//     CHECK:   affine.store %[[v14]], %arg2[%[[v11]], %[[v12]]] : memref<16x32xf32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @convert_affine_tc_buffer(%arg0: memref<16x8x4xf32>, %arg1: memref<8x4x32xf32>, %arg2: memref<16x32xf32>) {
  linalg.generic
    {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0, %arg1 : memref<16x8x4xf32>, memref<8x4x32xf32>) outs(%arg2 : memref<16x32xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %0 = arith.mulf %in, %in_0 : f32
        %1 = arith.addf %out, %0 : f32
        linalg.yield %1 : f32
    }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.linalg_to_affine %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

