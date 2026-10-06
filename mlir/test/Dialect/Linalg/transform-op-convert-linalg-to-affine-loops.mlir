// RUN: mlir-opt  %s --transform-interpreter -verify-diagnostics \
// RUN:     --split-input-file | FileCheck %s

//     CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
//     CHECK: func.func @convert_affine_matmul_buffer(%arg0: memref<16x8xf32>, %arg1: memref<8x32xf32>, %arg2: memref<16x32xf32>)
//     CHECK:   affine.for %[[arg3:.*]] = 0 to 16
//     CHECK:   affine.for %[[arg4:.*]] = 0 to 32
//     CHECK:   affine.for %[[arg5:.*]] = 0 to 8
//     CHECK:   %[[v0:.*]] = affine.apply #[[MAP]](%[[arg3]])
//     CHECK:   %[[v1:.*]] = affine.apply #[[MAP]](%[[arg5]])
//     CHECK:   %[[v2:.*]] = affine.load %arg0[%[[v0]], %[[v1]]] : memref<16x8xf32>
//     CHECK:   %[[v3:.*]] = affine.apply #[[MAP]](%[[arg5]])
//     CHECK:   %[[v4:.*]] = affine.apply #[[MAP]](%[[arg4]])
//     CHECK:   %[[v5:.*]] = affine.load %arg1[%[[v3]], %[[v4]]] : memref<8x32xf32>
//     CHECK:   %[[v6:.*]] = affine.apply #[[MAP]](%[[arg3]])
//     CHECK:   %[[v7:.*]] = affine.apply #[[MAP]](%[[arg4]])
//     CHECK:   %[[v8:.*]] = affine.load %arg2[%[[v6]], %[[v7]]] : memref<16x32xf32>
//     CHECK:   %[[v9:.*]] = affine.apply #[[MAP]](%[[arg3]])
//     CHECK:   %[[v10:.*]] = affine.apply #[[MAP]](%[[arg4]])
//     CHECK:   %[[v11:.*]] = arith.mulf %[[v2]], %[[v5]] : f32
//     CHECK:   %[[v12:.*]] = arith.addf %[[v8]], %[[v11]] : f32
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
    %1 = transform.structured.to_affine %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

//     CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
//     CHECK: func.func @convert_affine_tc_buffer(%arg0: memref<16x8x4xf32>, %arg1: memref<8x4x32xf32>, %arg2: memref<16x32xf32>)
//     CHECK:   affine.for %[[arg3:.*]] = 0 to 16
//     CHECK:   affine.for %[[arg4:.*]] = 0 to 32
//     CHECK:   affine.for %[[arg5:.*]] = 0 to 8
//     CHECK:   affine.for %[[arg6:.*]] = 0 to 4
//     CHECK:   %[[v0:.*]] = affine.apply #[[MAP]](%[[arg3]])
//     CHECK:   %[[v1:.*]] = affine.apply #[[MAP]](%[[arg5]])
//     CHECK:   %[[v2:.*]] = affine.apply #[[MAP]](%[[arg6]])
//     CHECK:   %[[v3:.*]] = affine.load %arg0[%[[v0]], %[[v1]], %[[v2]]] : memref<16x8x4xf32>
//     CHECK:   %[[v4:.*]] = affine.apply #[[MAP]](%[[arg5]])
//     CHECK:   %[[v5:.*]] = affine.apply #[[MAP]](%[[arg6]])
//     CHECK:   %[[v6:.*]] = affine.apply #[[MAP]](%[[arg4]])
//     CHECK:   %[[v7:.*]] = affine.load %arg1[%[[v4]], %[[v5]], %[[v6]]] : memref<8x4x32xf32>
//     CHECK:   %[[v8:.*]] = affine.apply #[[MAP]](%[[arg3]])
//     CHECK:   %[[v9:.*]] = affine.apply #[[MAP]](%[[arg4]])
//     CHECK:   %[[v10:.*]] = affine.load %arg2[%[[v8]], %[[v9]]] : memref<16x32xf32>
//     CHECK:   %[[v11:.*]] = affine.apply #[[MAP]](%[[arg3]])
//     CHECK:   %[[v12:.*]] = affine.apply #[[MAP]](%[[arg4]])
//     CHECK:   %[[v13:.*]] = arith.mulf %[[v3]], %[[v7]] : f32
//     CHECK:   %[[v14:.*]] = arith.addf %[[v10]], %[[v13]] : f32
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
    %1 = transform.structured.to_affine %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

//     CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
//     CHECK: func.func @convert_affine_nored_buffer(%arg0: memref<16x8xf32>, %arg1: memref<8x16xf32>, %arg2: memref<16x8xf32>)
//     CHECK:   affine.for %[[arg3:.*]] = 0 to 16
//     CHECK:   affine.for %[[arg4:.*]] = 0 to 8
//     CHECK:   %[[v0:.*]] = affine.apply #[[MAP]](%[[arg3]])
//     CHECK:   %[[v1:.*]] = affine.apply #[[MAP]](%[[arg4]])
//     CHECK:   %[[v2:.*]] = affine.load %arg0[%[[v0]], %[[v1]]] : memref<16x8xf32>
//     CHECK:   %[[v3:.*]] = affine.apply #[[MAP]](%[[arg4]])
//     CHECK:   %[[v4:.*]] = affine.apply #[[MAP]](%[[arg3]])
//     CHECK:   %[[v5:.*]] = affine.load %arg1[%[[v3]], %[[v4]]] : memref<8x16xf32>
//     CHECK:   %[[v6:.*]] = affine.apply #[[MAP]](%[[arg3]])
//     CHECK:   %[[v7:.*]] = affine.apply #[[MAP]](%[[arg4]])
//     CHECK:   %[[v8:.*]] = affine.load %arg2[%[[v6]], %[[v7]]] : memref<16x8xf32>
//     CHECK:   %[[v9:.*]] = affine.apply #[[MAP]](%[[arg3]])
//     CHECK:   %[[v10:.*]] = affine.apply #[[MAP]](%[[arg4]])
//     CHECK:   %[[v11:.*]] = arith.mulf %[[v2]], %[[v5]] : f32
//     CHECK:   %[[v12:.*]] = arith.addf %[[v8]], %[[v11]] : f32
//     CHECK:   affine.store %[[v12]], %arg2[%[[v9]], %[[v10]]] : memref<16x8xf32>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

func.func @convert_affine_nored_buffer(%arg0: memref<16x8xf32>, %arg1: memref<8x16xf32>, %arg2: memref<16x8xf32>) {
  linalg.generic
    {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<16x8xf32>, memref<8x16xf32>) outs(%arg2 : memref<16x8xf32>) {
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
    %1 = transform.structured.to_affine %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}



// -----

//     CHECK: func.func @convert_affine_0dim_generic_buffer(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>)
//     CHECK:   %[[v0:.*]] = affine.load %arg0[] : memref<f32>
//     CHECK:   %[[v1:.*]] = affine.load %arg1[] : memref<f32>
//     CHECK:   %[[v2:.*]] = affine.load %arg2[] : memref<f32>
//     CHECK:   %[[v3:.*]] = arith.mulf %[[v0]], %[[v1]] : f32
//     CHECK:   %[[v4:.*]] = arith.addf %[[v2]], %[[v3]] : f32
//     CHECK:   affine.store %[[v4]], %arg2[] : memref<f32>
#map  = affine_map<() -> ()>
#map1 = affine_map<() -> ()>
#map2 = affine_map<() -> ()>

func.func @convert_affine_0dim_generic_buffer(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>) {
  linalg.generic
    {indexing_maps = [#map, #map1, #map2], iterator_types = []}
    ins(%arg0, %arg1 : memref<f32>, memref<f32>) outs(%arg2 : memref<f32>) {
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
    %1 = transform.structured.to_affine %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

