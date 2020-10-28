// RUN: mlir-opt %s -linalg-fusion-for-tensor-ops -split-input-file | FileCheck %s

// CHECK-DAG: [[$MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
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
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP0]], [[$MAP0]], [[$MAP0]], [[$MAP0]]{{\]}}
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

// CHECK-DAG: [[$MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[$MAP1:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d1, d0)>
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
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP0]], [[$MAP1]], [[$MAP0]], [[$MAP0]]{{\]}}
  %2 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} %0, %arg2 {
    ^bb0(%arg5: f32, %arg6: f32):       // no predecessors
      %3 = mulf %arg5, %arg6 : f32
      linalg.yield %3 : f32
    }: tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// -----

// CHECK-DAG: [[$MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[$MAP1:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d1, d0)>
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
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP1]], [[$MAP0]], [[$MAP0]], [[$MAP0]]{{\]}}
  %2 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel"]} %0, %arg2 {
    ^bb0(%arg5: f32, %arg6: f32):       // no predecessors
      %3 = mulf %arg5, %arg6 : f32
      linalg.yield %3 : f32
    }: tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// -----

// CHECK-DAG: [[$MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[$MAP1:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0)>
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
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP1]], [[$MAP1]], [[$MAP0]], [[$MAP0]]
  %2 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel"]} %0, %arg2 {
    ^bb0(%arg5: f32, %arg6: f32):       // no predecessors
      %3 = mulf %arg5, %arg6 : f32
      linalg.yield %3 : f32
    }: tensor<?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<() -> ()>
#map0 = affine_map<() -> ()>

// CHECK-LABEL: @add_mul_scalar_fusion
func @add_mul_scalar_fusion(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32>
{
  %0 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = []} %arg0, %arg1 {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = addf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  }: tensor<f32>, tensor<f32> -> tensor<f32>
  // CHECK: linalg.generic {args_in = 3 : i64, args_out = 1 : i64
  // CHECK: addf
  // CHECK: mulf
  %1 = linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = []} %0, %arg2 {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  }: tensor<f32>, tensor<f32> -> tensor<f32>

  return %1 : tensor<f32>
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 4 + d2, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @generic_op_reshape_producer_fusion(%arg0 : tensor<?x?x?xf32>,
                                         %arg1 : tensor<?x?x4x?xf32>) ->
                                         tensor<?x?x4x?xf32>
{
  %0 = linalg.tensor_reshape %arg0 [affine_map<(i, j, k, l) -> (i)>,
                                    affine_map<(i, j, k, l) -> (j, k)>,
                                    affine_map<(i, j, k, l) -> (l)>] :
    tensor<?x?x?xf32> into tensor<?x?x4x?xf32>
  %1 = linalg.generic
    {args_in = 2 : i64, args_out = 1 : i64,
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    %0, %arg1 {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  }: tensor<?x?x4x?xf32>, tensor<?x?x4x?xf32> -> tensor<?x?x4x?xf32>
  return %1 : tensor<?x?x4x?xf32>
}

// CHECK-LABEL: func @generic_op_reshape_producer_fusion
//       CHECK: linalg.generic
//  CHECK-SAME:   args_in = 2
//  CHECK-SAME:   args_out = 1
//  CHECK-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]]
//   CHECK-NOT: linalg.generic

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 20 + d2 * 5 + d3)>

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @generic_op_reshape_consumer_fusion(%arg0 : tensor<?x?x4x5xf32>,
                                         %arg1 : tensor<?x?x4x5xf32>) ->
                                         tensor<?x?xf32>
{
  %0 = linalg.generic
    {args_in = 2 : i64, args_out = 1 : i64,
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    %arg0, %arg1 {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  }: tensor<?x?x4x5xf32>, tensor<?x?x4x5xf32> -> tensor<?x?x4x5xf32>
  %1 = linalg.tensor_reshape %0 [affine_map<(i, j, k, l) -> (i)>,
                                 affine_map<(i, j, k, l) -> (j, k, l)>] :
    tensor<?x?x4x5xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @generic_op_reshape_consumer_fusion
//       CHECK: linalg.generic
//  CHECK-SAME:   args_in = 2
//  CHECK-SAME:   args_out = 1
//  CHECK-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP1]]]
//   CHECK-NOT: linalg.generic

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @generic_op_reshape_consumer_nofusion(%arg0 : tensor<?x?x?x5xf32>,
                                           %arg1 : tensor<?x?x?x5xf32>) ->
                                           tensor<?x?xf32>
{
  %0 = linalg.generic
    {args_in = 2 : i64, args_out = 1 : i64,
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    %arg0, %arg1 {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  }: tensor<?x?x?x5xf32>, tensor<?x?x?x5xf32> -> tensor<?x?x?x5xf32>
  %1 = linalg.tensor_reshape %0 [affine_map<(i, j, k, l) -> (i)>,
                                 affine_map<(i, j, k, l) -> (j, k, l)>] :
    tensor<?x?x?x5xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @generic_op_reshape_consumer_nofusion
//       CHECK: linalg.tensor_reshape

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>

func @generic_op_reshape_consumer_expanding(%arg0: tensor<264x4xf32>)
                                            -> tensor<8x33x4xf32> {
  %cst = constant dense<2.000000e+00> : tensor<264x4xf32>
  %0 = linalg.generic
    {args_in = 2 : i64, args_out = 1 : i64,
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel"]}
    %arg0, %cst {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %2 = mulf %arg1, %arg2 : f32
      linalg.yield %2 : f32
    }: tensor<264x4xf32>, tensor<264x4xf32> -> tensor<264x4xf32>
  %1 = linalg.tensor_reshape %0 [#map1, #map2] :
    tensor<264x4xf32> into tensor<8x33x4xf32>
  return %1 : tensor<8x33x4xf32>
}

// The reshape op in `%arg0` is folded into the indexing map of generic op.
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0 * 33 + d1, d2)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//       CHECK: func @generic_op_reshape_consumer_expanding
//   CHECK-NOT:   linalg.tensor_reshape
//       CHECK:   %[[CST:.*]] = constant {{.*}} : f32
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]]]
//       CHECK:   tensor<264x4xf32> -> tensor<8x33x4xf32>
//   CHECK-NOT:   linalg.tensor_reshape

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_op_constant_fusion(%arg0 : tensor<5x?x?xf32>) -> tensor<5x?x?xf32>
{
  %0 = constant dense<42.0> : tensor<5xf32>
  %1 = linalg.generic
       {args_in = 2 : i64, args_out = 1 : i64,
         indexing_maps = [#map0, #map1, #map1],
         iterator_types = ["parallel", "parallel", "parallel"]}
       %0, %arg0 {
       ^bb0(%arg1: f32, %arg2: f32):
         %2 = mulf %arg1, %arg2 : f32
         linalg.yield %2 : f32
       }: tensor<5xf32>, tensor<5x?x?xf32> -> tensor<5x?x?xf32>
  return %1 : tensor<5x?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @generic_op_constant_fusion
//       CHECK:   %[[CST:.*]] = constant {{.*}} : f32
//       CHECK:   linalg.generic
//  CHECK-SAME:     args_in = 1 : i64
//  CHECK-SAME:     args_out = 1 : i64
//       CHECK:   ^{{.*}}(%[[ARG1:.*]]: f32)
//       CHECK:     mulf %[[CST]], %[[ARG1]]

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @indexed_generic_op_constant_fusion(%arg0 : tensor<5x?x?xf32>)
                                         -> tensor<5x?x?xf32>
{
  %0 = constant dense<42.0> : tensor<5xf32>
  %1 = linalg.indexed_generic
       {args_in = 2 : i64, args_out = 1 : i64,
         indexing_maps = [#map0, #map1, #map1],
         iterator_types = ["parallel", "parallel", "parallel"]}
       %0, %arg0 {
       ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: f32, %arg5 : f32):
         %2 = mulf %arg4, %arg5 : f32
         linalg.yield %2 : f32
       }: tensor<5xf32>, tensor<5x?x?xf32> -> tensor<5x?x?xf32>
  return %1 : tensor<5x?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @indexed_generic_op_constant_fusion
//       CHECK:   %[[CST:.*]] = constant {{.*}} : f32
//       CHECK:   linalg.indexed_generic
//  CHECK-SAME:     args_in = 1 : i64
//  CHECK-SAME:     args_out = 1 : i64
//       CHECK:   ^{{[a-zA-Z0-9_]*}}
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG4:.*]]: f32)
//       CHECK:     mulf %[[CST]], %[[ARG4]]

// -----

#map0 = affine_map<(d0, d1, d2) -> ()>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_op_zero_dim_constant_fusion(%arg0 : tensor<5x?x?xf32>)
  -> tensor<5x?x?xf32>
{
  %0 = constant dense<42.0> : tensor<f32>
  %1 = linalg.generic
       {args_in = 2 : i64, args_out = 1 : i64,
         indexing_maps = [#map0, #map1, #map1],
         iterator_types = ["parallel", "parallel", "parallel"]}
       %0, %arg0 {
       ^bb0(%arg1: f32, %arg2: f32):
         %2 = mulf %arg1, %arg2 : f32
         linalg.yield %2 : f32
       }: tensor<f32>, tensor<5x?x?xf32> -> tensor<5x?x?xf32>
  return %1 : tensor<5x?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @generic_op_zero_dim_constant_fusion
//       CHECK:   %[[CST:.*]] = constant {{.*}} : f32
//       CHECK:   linalg.generic
//  CHECK-SAME:     args_in = 1 : i64
//  CHECK-SAME:     args_out = 1 : i64
//       CHECK:   ^{{.*}}(%[[ARG1:.*]]: f32)
//       CHECK:     mulf %[[CST]], %[[ARG1]]

// -----

#map0 = affine_map<(d0, d1, d2) -> ()>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @indexed_generic_op_zero_dim_constant_fusion
  (%arg0 : tensor<5x?x?xf32>) -> tensor<5x?x?xf32>
{
  %0 = constant dense<42.0> : tensor<f32>
  %1 = linalg.indexed_generic
       {args_in = 2 : i64, args_out = 1 : i64,
         indexing_maps = [#map0, #map1, #map1],
         iterator_types = ["parallel", "parallel", "parallel"]}
       %0, %arg0 {
       ^bb0(%arg1 : index, %arg2 : index, %arg3 : index, %arg4: f32, %arg5: f32):
         %2 = mulf %arg4, %arg5 : f32
         linalg.yield %2 : f32
       }: tensor<f32>, tensor<5x?x?xf32> -> tensor<5x?x?xf32>
  return %1 : tensor<5x?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @indexed_generic_op_zero_dim_constant_fusion
//       CHECK:   %[[CST:.*]] = constant {{.*}} : f32
//       CHECK:   linalg.indexed_generic
//  CHECK-SAME:     args_in = 1 : i64
//  CHECK-SAME:     args_out = 1 : i64
//       CHECK:   ^{{[a-zA-Z0-9_]*}}
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG4:.*]]: f32)
//       CHECK:     mulf %[[CST]], %[[ARG4]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func @generic_op_indexed_generic_op_fusion(%arg0: tensor<?x?xi32>,
                                           %arg1: tensor<?x?xi32>) {
    %0 = linalg.generic {
      args_in = 2 : i64,
      args_out = 1 : i64,
      indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"] } %arg0, %arg1 {
    ^bb0(%arg2: i32, %arg3: i32):       // no predecessors
      %10 = addi %arg2, %arg3 : i32
      linalg.yield %10 : i32
    } : tensor<?x?xi32>, tensor<?x?xi32> -> tensor<?x?xi32>
    %1 = linalg.indexed_generic {
      args_in = 1 : i64,
      args_out = 1 : i64,
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel", "parallel"] } %0 {
    ^bb0(%arg2: index, %arg3: index, %arg4: i32):       // no predecessors
      %2 = index_cast %arg2 : index to i32
      %3 = index_cast %arg3 : index to i32
      %4 = addi %arg4, %2 : i32
      %5 = subi %4, %3 : i32
      linalg.yield %5 : i32
    }: tensor<?x?xi32> -> tensor<?x?xi32>
  return
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @generic_op_indexed_generic_op_fusion
//   CHECK-NOT: linalg.generic
//       CHECK: linalg.indexed_generic
// CHECK-SAME:    args_in = 2
// CHECK-SAME:    args_out = 1
// CHECK-SAME:    indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP0]]]
//      CHECK: ^{{[a-zA-Z0-9_]*}}
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: index
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: index
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: i32
// CHECK-SAME: %[[ARG3:[a-zA-Z0-9_]*]]: i32
//      CHECK:   %[[VAL1:.+]] = addi %[[ARG2]], %[[ARG3]] : i32
//      CHECK:   %[[ADD_OPERAND:.+]] = index_cast %[[ARG0]] : index to i32
//      CHECK:   %[[SUB_OPERAND:.+]] = index_cast %[[ARG1]] : index to i32
//      CHECK:   %[[VAL2:.+]] = addi %[[VAL1]], %[[ADD_OPERAND]] : i32
//      CHECK:   %[[VAL3:.+]] = subi %[[VAL2]], %[[SUB_OPERAND]] : i32
//      CHECK:   linalg.yield %[[VAL3]] : i32

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func @indexed_generic_op_generic_op_fusion(%arg0: tensor<?x?xi32>,
                                           %arg1: tensor<?x?xi32>) {
  %0 = linalg.indexed_generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel"] } %arg0 {
  ^bb0(%arg2: index, %arg3: index, %arg4: i32):       // no predecessors
    %2 = index_cast %arg2 : index to i32
    %3 = index_cast %arg3 : index to i32
    %4 = addi %arg4, %2 : i32
    %5 = subi %4, %3 : i32
    linalg.yield %5 : i32
  }: tensor<?x?xi32> -> tensor<?x?xi32>
  %1 = linalg.generic {
    args_in = 2 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0, #map0],
    iterator_types = ["parallel", "parallel"] } %0, %arg1 {
  ^bb0(%arg2: i32, %arg3: i32):       // no predecessors
    %10 = addi %arg2, %arg3 : i32
    linalg.yield %10 : i32
  } : tensor<?x?xi32>, tensor<?x?xi32> -> tensor<?x?xi32>
  return
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @indexed_generic_op_generic_op_fusion
//       CHECK: linalg.indexed_generic
// CHECK-SAME:    args_in = 2
// CHECK-SAME:    args_out = 1
// CHECK-SAME:    indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP0]]]
//      CHECK: ^{{[a-zA-Z0-9_]*}}
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: index
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: index
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: i32
// CHECK-SAME: %[[ARG3:[a-zA-Z0-9_]*]]: i32
//      CHECK:   %[[ADD_OPERAND:.+]] = index_cast %[[ARG0]] : index to i32
//      CHECK:   %[[SUB_OPERAND:.+]] = index_cast %[[ARG1]] : index to i32
//      CHECK:   %[[VAL1:.+]] = addi %[[ARG2]], %[[ADD_OPERAND]] : i32
//      CHECK:   %[[VAL2:.+]] = subi %[[VAL1]], %[[SUB_OPERAND]] : i32
//      CHECK:   %[[VAL3:.+]] = addi %[[VAL2]], %[[ARG3]] : i32
//      CHECK:   linalg.yield %[[VAL3]] : i32
//   CHECK-NOT: linalg.generic

// -----

// The indices of the first indexed_generic op are swapped after fusion.
#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func @indexed_generic_op_fusion(%arg0: tensor<?x?xi32>) {
    %0 = linalg.indexed_generic {
      args_in = 1 : i64,
      args_out = 1 : i64,
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel", "parallel"] } %arg0 {
    ^bb0(%arg2: index, %arg3: index, %arg4: i32):       // no predecessors
      %2 = index_cast %arg2 : index to i32
      %3 = index_cast %arg3 : index to i32
      %4 = addi %arg4, %2 : i32
      %5 = subi %4, %3 : i32
      linalg.yield %5 : i32
    }: tensor<?x?xi32> -> tensor<?x?xi32>
    %1 = linalg.indexed_generic {
      args_in = 1 : i64,
      args_out = 1 : i64,
      indexing_maps = [#map1, #map1],
      iterator_types = ["parallel", "parallel"] } %0 {
    ^bb0(%arg2: index, %arg3: index, %arg4: i32):       // no predecessors
      %2 = index_cast %arg2 : index to i32
      %3 = index_cast %arg3 : index to i32
      %4 = addi %arg4, %2 : i32
      %5 = subi %4, %3 : i32
      linalg.yield %5 : i32
    }: tensor<?x?xi32> -> tensor<?x?xi32>
  return
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @indexed_generic_op_fusion
//       CHECK: linalg.indexed_generic
// CHECK-SAME:    args_in = 1
// CHECK-SAME:    args_out = 1
// CHECK-SAME:    indexing_maps = [#[[$MAP0]], #[[$MAP0]]]
//      CHECK: ^{{[a-zA-Z0-9_]*}}
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: index
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: index
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: i32
//      CHECK:   %[[ADD_OPERAND1:.+]] = index_cast %[[ARG1]] : index to i32
//      CHECK:   %[[SUB_OPERAND1:.+]] = index_cast %[[ARG0]] : index to i32
//      CHECK:   %[[VAL1:.+]] = addi %[[ARG2]], %[[ADD_OPERAND1]] : i32
//      CHECK:   %[[VAL2:.+]] = subi %[[VAL1]], %[[SUB_OPERAND1]] : i32
//      CHECK:   %[[ADD_OPERAND2:.+]] = index_cast %[[ARG0]] : index to i32
//      CHECK:   %[[SUB_OPERAND2:.+]] = index_cast %[[ARG1]] : index to i32
//      CHECK:   %[[VAL3:.+]] = addi %[[VAL2]], %[[ADD_OPERAND2]] : i32
//      CHECK:   %[[VAL4:.+]] = subi %[[VAL3]], %[[SUB_OPERAND2]] : i32
//      CHECK:   linalg.yield %[[VAL4]] : i32
//   CHECK-NOT: linalg.indexed_generic

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 4 + d2, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @indexed_generic_op_reshape_producer_fusion(%arg0 : tensor<?x?x?xi32>)
  -> tensor<?x?x4x?xi32> {
  %0 = linalg.tensor_reshape %arg0 [affine_map<(i, j, k, l) -> (i)>,
                                    affine_map<(i, j, k, l) -> (j, k)>,
                                    affine_map<(i, j, k, l) -> (l)>] :
    tensor<?x?x?xi32> into tensor<?x?x4x?xi32>
  %1 = linalg.indexed_generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"] } %0 {
  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: i32):       // no predecessors
    %2 = index_cast %arg2 : index to i32
    %3 = addi %arg6, %2 : i32
    linalg.yield %3 : i32
  }: tensor<?x?x4x?xi32> -> tensor<?x?x4x?xi32>
  return %1 : tensor<?x?x4x?xi32>
}

// CHECK-LABEL: func @indexed_generic_op_reshape_producer_fusion
//   CHECK-NOT: linalg.tensor_reshape
//       CHECK: linalg.indexed_generic
//  CHECK-SAME:   args_in = 1
//  CHECK-SAME:   args_out = 1
//  CHECK-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
//   CHECK-NOT: linalg.tensor_reshape

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 20 + d2 * 5 + d3)>

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @indexed_generic_op_reshape_consumer_fusion(%arg0 : tensor<?x?x4x5xi32>)
  -> tensor<?x?xi32> {
  %0 = linalg.indexed_generic {
    args_in = 1 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"] } %arg0 {
  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: i32):       // no predecessors
    %2 = index_cast %arg2 : index to i32
    %3 = addi %arg6, %2 : i32
    linalg.yield %3 : i32
  }: tensor<?x?x4x5xi32> -> tensor<?x?x4x5xi32>
  %1 = linalg.tensor_reshape %0 [affine_map<(i, j, k, l) -> (i)>,
                                 affine_map<(i, j, k, l) -> (j, k, l)>] :
    tensor<?x?x4x5xi32> into tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: func @indexed_generic_op_reshape_consumer_fusion
//   CHECK-NOT: linalg.tensor_reshape
//       CHECK: linalg.indexed_generic
//  CHECK-SAME:   args_in = 1
//  CHECK-SAME:   args_out = 1
//  CHECK-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
//   CHECK-NOT: linalg.tensor_reshape
