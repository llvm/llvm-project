// RUN: mlir-opt %s -linalg-fusion-for-tensor-ops -split-input-file | FileCheck %s --dump-input-on-failure

// CHECK-DAG: [[MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @add_mul_fusion
func @add_mul_fusion(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %0 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} %arg0, %arg1 {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = addf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  }: tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  // CHECK: linalg.generic {args_in = 3 : i64, args_out = 1 : i64
  // CHECK-SAME: indexing_maps = {{\[}}[[MAP0]], [[MAP0]], [[MAP0]], [[MAP0]]{{\]}}
  %2 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} %0, %arg2 {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32):       // no predecessors
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = addf [[ARG0]], [[ARG1]]
      // CHECK-NOT: linalg.yield
      // CHECK: mulf [[T1]], [[ARG2]]
      // CHECK: linalg.yield
      %3 = mulf %arg5, %arg6 : f32
      linalg.yield %3 : f32
    }: tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// -----

// CHECK-DAG: [[MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[MAP1:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d1, d0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @transpose_add_mul_fusion
func @transpose_add_mul_fusion(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %0 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} %arg0, %arg1 {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = addf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  }: tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  // CHECK: linalg.generic {args_in = 3 : i64, args_out = 1 : i64
  // CHECK-SAME: indexing_maps = {{\[}}[[MAP0]], [[MAP1]], [[MAP0]], [[MAP0]]{{\]}}
  %2 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} %0, %arg2 {
    ^bb0(%arg5: f32, %arg6: f32):       // no predecessors
      %3 = mulf %arg5, %arg6 : f32
      linalg.yield %3 : f32
    }: tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// -----

// CHECK-DAG: [[MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[MAP1:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d1, d0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @add_transpose_mul_fusion
func @add_transpose_mul_fusion(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %0 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} %arg0, %arg1 {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = addf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  }: tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  // CHECK: linalg.generic {args_in = 3 : i64, args_out = 1 : i64
  // CHECK-SAME: indexing_maps = {{\[}}[[MAP1]], [[MAP0]], [[MAP0]], [[MAP0]]{{\]}}
  %2 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel"]} %0, %arg2 {
    ^bb0(%arg5: f32, %arg6: f32):       // no predecessors
      %3 = mulf %arg5, %arg6 : f32
      linalg.yield %3 : f32
    }: tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// -----

// CHECK-DAG: [[MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[MAP1:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: [[MAP2:#[a-zA-Z0-9_]*]] = affine_map<(d0) -> (d0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @add_broadcast_mul_fusion
func @add_broadcast_mul_fusion(%arg0: tensor<?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %0 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} %arg0, %arg1 {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = addf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  }: tensor<?xf32>, tensor<?xf32> -> tensor<?xf32>
  // CHECK: linalg.generic {args_in = 3 : i64, args_out = 1 : i64
  // CHECK-SAME: indexing_maps = {{\[}}[[MAP1]], [[MAP1]], [[MAP0]], [[MAP0]]
  %2 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel"]} %0, %arg2 {
    ^bb0(%arg5: f32, %arg6: f32):       // no predecessors
      %3 = mulf %arg5, %arg6 : f32
      linalg.yield %3 : f32
    }: tensor<?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
