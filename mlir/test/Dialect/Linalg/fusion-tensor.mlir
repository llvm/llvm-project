// RUN: mlir-opt %s -linalg-fusion-for-tensor-ops -split-input-file | FileCheck %s

// CHECK-DAG: [[$MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @add_mul_fusion
func @add_mul_fusion(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = memref.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = memref.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       // no predecessors
      %4 = addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP0]], [[$MAP0]], [[$MAP0]], [[$MAP0]]{{\]}}
  %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%3, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       // no predecessors
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = addf [[ARG0]], [[ARG1]]
      // CHECK-NOT: linalg.yield
      // CHECK: mulf [[T1]], [[ARG2]]
      // CHECK: linalg.yield
      %5 = mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}

// -----

// CHECK-DAG: [[$MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[$MAP1:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d1, d0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @transpose_add_mul_fusion
func @transpose_add_mul_fusion(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = memref.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = memref.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       // no predecessors
      %4 = addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP0]], [[$MAP1]], [[$MAP0]], [[$MAP0]]{{\]}}
  %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%3, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       // no predecessors
      %5 = mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}

// -----

// CHECK-DAG: [[$MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[$MAP1:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d1, d0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @add_transpose_mul_fusion
func @add_transpose_mul_fusion(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = memref.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = memref.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       // no predecessors
      %4 = addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP1]], [[$MAP0]], [[$MAP0]], [[$MAP0]]{{\]}}
  %4 = linalg.generic {indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%3, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>){
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       // no predecessors
      %5= mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
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
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = memref.dim %arg0, %c0 : tensor<?xf32>
  %1 = linalg.init_tensor [%0] : tensor<?xf32>
  %2 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]}
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
      outs(%1 : tensor<?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       // no predecessors
      %3 = addf %arg3, %arg4 : f32
      linalg.yield %3 : f32
  } -> tensor<?xf32>
  // CHECK: linalg.generic {
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP1]], [[$MAP1]], [[$MAP0]], [[$MAP0]]
  %3 = memref.dim %arg2, %c1 : tensor<?x?xf32>
  %4 = linalg.init_tensor [%0, %3] : tensor<?x?xf32>
  %5 = linalg.generic {indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%2, %arg2 : tensor<?xf32>, tensor<?x?xf32>)
      outs(%4 : tensor<?x?xf32>){
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       // no predecessors
      %6 = mulf %arg5, %arg6 : f32
      linalg.yield %6 : f32
    } -> tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<() -> ()>
#map0 = affine_map<() -> ()>

// CHECK-LABEL: @add_mul_scalar_fusion
func @add_mul_scalar_fusion(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32>
{
  %0 = linalg.init_tensor [] : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []}
      ins(%arg0, %arg1 : tensor<f32>, tensor<f32>)
      outs(%0 : tensor<f32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       // no predecessors
      %2 = addf %arg3, %arg4 : f32
      linalg.yield %2 : f32
  } -> tensor<f32>
  // CHECK: linalg.generic {
  // CHECK: addf
  // CHECK: mulf
  %2 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []}
      ins(%1, %arg2 : tensor<f32>, tensor<f32>)
      outs(%0 : tensor<f32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       // no predecessors
      %3 = mulf %arg3, %arg4 : f32
      linalg.yield %3 : f32
  } -> tensor<f32>

  return %2 : tensor<f32>
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_op_constant_fusion(%arg0 : tensor<5x?x?xf32>) -> tensor<5x?x?xf32>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %cst = constant dense<42.0> : tensor<5xf32>
  %0 = memref.dim %arg0, %c1 : tensor<5x?x?xf32>
  %1 = memref.dim %arg0, %c2 : tensor<5x?x?xf32>
  %2 = linalg.init_tensor [5, %0, %1] : tensor<5x?x?xf32>
  %3 = linalg.generic {
    indexing_maps = [#map0, #map1, #map1],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%cst, %arg0 : tensor<5xf32>, tensor<5x?x?xf32>)
    outs(%2 : tensor<5x?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %4 = mulf %arg1, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<5x?x?xf32>
  return %3 : tensor<5x?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @generic_op_constant_fusion
//       CHECK:   %[[CST:.*]] = constant {{.*}} : f32
//       CHECK:   linalg.generic
//       CHECK:   ^{{.+}}(%[[ARG1:[a-zA-Z0-9_]+]]: f32, %{{.+}}: f32):
//       CHECK:     mulf %[[CST]], %[[ARG1]]

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @indexed_generic_op_constant_fusion(%arg0 : tensor<5x?x?xf32>)
                                         -> tensor<5x?x?xf32>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %cst = constant dense<42.0> : tensor<5xf32>
  %0 = memref.dim %arg0, %c1 : tensor<5x?x?xf32>
  %1 = memref.dim %arg0, %c2 : tensor<5x?x?xf32>
  %2 = linalg.init_tensor [5, %0, %1] : tensor<5x?x?xf32>
  %3 = linalg.indexed_generic {
    indexing_maps = [#map0, #map1, #map1],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%cst, %arg0 : tensor<5xf32>, tensor<5x?x?xf32>)
    outs(%2 : tensor<5x?x?xf32>) {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: f32, %arg5 : f32, %arg6 : f32):
      %4 = mulf %arg4, %arg5 : f32
      linalg.yield %4 : f32
    } -> tensor<5x?x?xf32>
  return %3 : tensor<5x?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @indexed_generic_op_constant_fusion
//       CHECK:   %[[CST:.*]] = constant {{.*}} : f32
//       CHECK:   linalg.indexed_generic
//       CHECK:   ^{{[a-zA-Z0-9_]*}}
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]*]]: f32, %{{.*}}: f32)
//       CHECK:     mulf %[[CST]], %[[ARG4]]

// -----

#map0 = affine_map<(d0, d1, d2) -> ()>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_op_zero_dim_constant_fusion(%arg0 : tensor<5x?x?xf32>)
  -> tensor<5x?x?xf32>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %cst = constant dense<42.0> : tensor<f32>
  %0 = memref.dim %arg0, %c1 : tensor<5x?x?xf32>
  %1 = memref.dim %arg0, %c2 : tensor<5x?x?xf32>
  %2 = linalg.init_tensor [5, %0, %1] : tensor<5x?x?xf32>
  %3 = linalg.generic {
    indexing_maps = [#map0, #map1, #map1],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%cst, %arg0 : tensor<f32>, tensor<5x?x?xf32>)
    outs(%2 : tensor<5x?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %4 = mulf %arg1, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<5x?x?xf32>
  return %3 : tensor<5x?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @generic_op_zero_dim_constant_fusion
//       CHECK:   %[[CST:.*]] = constant {{.*}} : f32
//       CHECK:   linalg.generic
//       CHECK:   ^{{.*}}(%[[ARG1:[a-zA-Z0-9_]*]]: f32, %{{.*}}: f32)
//       CHECK:     mulf %[[CST]], %[[ARG1]]

// -----

#map0 = affine_map<(d0, d1, d2) -> ()>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @indexed_generic_op_zero_dim_constant_fusion
  (%arg0 : tensor<5x?x?xf32>) -> tensor<5x?x?xf32>
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %cst = constant dense<42.0> : tensor<f32>
  %0 = memref.dim %arg0, %c1 : tensor<5x?x?xf32>
  %1 = memref.dim %arg0, %c2 : tensor<5x?x?xf32>
  %2 = linalg.init_tensor [5, %0, %1] : tensor<5x?x?xf32>
  %3 = linalg.indexed_generic {
    indexing_maps = [#map0, #map1, #map1],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%cst, %arg0 : tensor<f32>, tensor<5x?x?xf32>)
    outs(%2 : tensor<5x?x?xf32>) {
    ^bb0(%arg1 : index, %arg2 : index, %arg3 : index, %arg4: f32, %arg5: f32, %arg6: f32):
      %4 = mulf %arg4, %arg5 : f32
      linalg.yield %4 : f32
    } -> tensor<5x?x?xf32>
  return %3 : tensor<5x?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @indexed_generic_op_zero_dim_constant_fusion
//       CHECK:   %[[CST:.*]] = constant {{.*}} : f32
//       CHECK:   linalg.indexed_generic
//       CHECK:   ^{{[a-zA-Z0-9_]*}}
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9_]*]]: f32, %{{.*}}: f32)
//       CHECK:     mulf %[[CST]], %[[ARG4]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func @generic_op_indexed_generic_op_fusion(%arg0: tensor<?x?xi32>,
                                           %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = memref.dim %arg0, %c0 : tensor<?x?xi32>
  %1 = memref.dim %arg0, %c1 : tensor<?x?xi32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xi32>
  %3 = linalg.generic {
    indexing_maps = [#map0, #map0, #map0],
    iterator_types = ["parallel", "parallel"] }
    ins(%arg0, %arg1  : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):       // no predecessors
      %10 = addi %arg2, %arg3 : i32
      linalg.yield %10 : i32
    } -> tensor<?x?xi32>
  %4 = linalg.indexed_generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel"] }
    ins(%3 : tensor<?x?xi32>)
    outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: index, %arg3: index, %arg4: i32, %arg5: i32):       // no predecessors
      %5 = index_cast %arg2 : index to i32
      %6 = index_cast %arg3 : index to i32
      %7 = addi %arg4, %5 : i32
      %8 = subi %7, %6 : i32
      linalg.yield %8 : i32
    } -> tensor<?x?xi32>
  return %4 : tensor<?x?xi32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @generic_op_indexed_generic_op_fusion
//   CHECK-NOT: linalg.generic
//       CHECK: linalg.indexed_generic
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
                                           %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = memref.dim %arg0, %c0 : tensor<?x?xi32>
  %1 = memref.dim %arg0, %c1 : tensor<?x?xi32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xi32>
  %3 = linalg.indexed_generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel"] }
    ins(%arg0 : tensor<?x?xi32>)
    outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: index, %arg3: index, %arg4: i32, %arg5: i32):       // no predecessors
      %4 = index_cast %arg2 : index to i32
      %5 = index_cast %arg3 : index to i32
      %6 = addi %arg4, %4 : i32
      %7 = subi %6, %5 : i32
      linalg.yield %7 : i32
    } -> tensor<?x?xi32>
  %4 = linalg.generic {
    indexing_maps = [#map0, #map0, #map0],
    iterator_types = ["parallel", "parallel"] }
    ins(%3, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):       // no predecessors
      %10 = addi %arg2, %arg3 : i32
      linalg.yield %10 : i32
    } -> tensor<?x?xi32>
  return %4 : tensor<?x?xi32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @indexed_generic_op_generic_op_fusion
//       CHECK: linalg.indexed_generic
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
func @indexed_generic_op_fusion(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = memref.dim %arg0, %c0 : tensor<?x?xi32>
  %1 = memref.dim %arg0, %c1 : tensor<?x?xi32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xi32>
  %3 = linalg.indexed_generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel"] }
    ins(%arg0 : tensor<?x?xi32>)
    outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: index, %arg3: index, %arg4: i32, %arg5: i32):       // no predecessors
      %4 = index_cast %arg2 : index to i32
      %5 = index_cast %arg3 : index to i32
      %6 = addi %arg4, %4 : i32
      %7 = subi %5, %6 : i32
      linalg.yield %7 : i32
    } -> tensor<?x?xi32>
  %4= linalg.indexed_generic {
    indexing_maps = [#map1, #map1],
    iterator_types = ["parallel", "parallel"] }
    ins(%3 : tensor<?x?xi32>)
    outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: index, %arg3: index, %arg4: i32, %arg5: i32):       // no predecessors
      %5 = index_cast %arg2 : index to i32
      %6 = index_cast %arg3 : index to i32
      %7 = addi %arg4, %5 : i32
      %8 = subi %7, %6 : i32
      linalg.yield %8 : i32
    } -> tensor<?x?xi32>
  return %4 : tensor<?x?xi32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @indexed_generic_op_fusion
//       CHECK: linalg.indexed_generic
// CHECK-SAME:    indexing_maps = [#[[$MAP0]], #[[$MAP0]]]
//      CHECK: ^{{[a-zA-Z0-9_]*}}
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: index
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: index
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: i32
//      CHECK:   %[[ADD_OPERAND1:.+]] = index_cast %[[ARG1]] : index to i32
//      CHECK:   %[[SUB_OPERAND1:.+]] = index_cast %[[ARG0]] : index to i32
//      CHECK:   %[[VAL1:.+]] = addi %[[ARG2]], %[[ADD_OPERAND1]] : i32
//      CHECK:   %[[VAL2:.+]] = subi %[[SUB_OPERAND1]], %[[VAL1]] : i32
//      CHECK:   %[[ADD_OPERAND2:.+]] = index_cast %[[ARG0]] : index to i32
//      CHECK:   %[[SUB_OPERAND2:.+]] = index_cast %[[ARG1]] : index to i32
//      CHECK:   %[[VAL3:.+]] = addi %[[VAL2]], %[[ADD_OPERAND2]] : i32
//      CHECK:   %[[VAL4:.+]] = subi %[[VAL3]], %[[SUB_OPERAND2]] : i32
//      CHECK:   linalg.yield %[[VAL4]] : i32
//   CHECK-NOT: linalg.indexed_generic

// -----

func @scalar_indexed_generic_fusion
  (%arg0: tensor<5x1x1xf32>, %arg1 : tensor<i32>) -> tensor<10xf32>
{
  %c0 = constant 0 : index
  %cst = constant dense<1.000000e+00> : tensor<10xf32>
  %0 = linalg.init_tensor [] : tensor<f32>
  %1 = linalg.indexed_generic
    {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>],
     iterator_types = []}
    ins(%arg1 : tensor<i32>) outs(%0 : tensor<f32>) {
    ^bb0(%arg2: i32, %arg3: f32):  // no predecessors
      %3 = index_cast %arg2 : i32 to index
      %4 = tensor.extract %arg0[%3, %c0, %c0] : tensor<5x1x1xf32>
      linalg.yield %4 : f32
    } -> tensor<f32>
  %2 = linalg.init_tensor [10] : tensor<10xf32>
  %3 = linalg.generic
   {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%1, %cst : tensor<f32>, tensor<10xf32>) outs(%2 : tensor<10xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %4 = mulf %arg2, %arg3 : f32
      linalg.yield %4 : f32
    } -> tensor<10xf32>
  return %3 : tensor<10xf32>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0) -> ()>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (d0)>
//       CHECK: func @scalar_indexed_generic_fusion
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<5x1x1xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<i32>
//       CHECK:   %[[T0:.+]] = linalg.indexed_generic
//  CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]]]
//  CHECK-SAME:     iterator_types = ["parallel"]
//  CHECK-SAME:     ins(%[[ARG1]] : tensor<i32>)
//       CHECK:     tensor.extract %[[ARG0]]
//       CHECK:     linalg.yield
//       CHECK   return %[[T0]]

// -----

func @constant_fusion(%arg0 : tensor<4xf32>) -> (tensor<4xf32>) {
  %cst = constant dense<1.0> : tensor<4xf32>
  %1 = linalg.init_tensor [4] : tensor<4xf32>
  %2 = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                      affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]}
    ins (%arg0, %cst : tensor<4xf32>, tensor<4xf32>)
    outs (%1 : tensor<4xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %3 = addf %arg1, %arg2 : f32
      linalg.yield %3 : f32
    } -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
//      CHECK: func @constant_fusion(%[[ARG0:.+]]: tensor<4xf32>)
//  CHECK-DAG:   %[[CST:.+]] = constant 1.000000e+00 : f32
//  CHECK-DAG:   %[[T0:.+]] = linalg.init_tensor [4] : tensor<4xf32>
//      CHECK:   %[[T1:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME:     ins(%[[ARG0]] : tensor<4xf32>)
// CHECK-SAME:     outs(%[[T0]] : tensor<4xf32>)
//      CHECK:   ^{{.+}}(
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: f32, %[[ARG2:[a-zA-Z0-9_]+]]: f32)
//      CHECK:     %[[T2:.+]] = addf %[[ARG1]], %[[CST]]
//      CHECK:     linalg.yield %[[T2]]
//      CHECK:   return %[[T1]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (0, d0)>
#map2 = affine_map<(d0) -> (0)>
func @consumer_with_reduction(%arg0: tensor<1x10xf32>,
                              %arg1: tensor<1x10xf32>,
                              %arg2: tensor<1xf32>) -> tensor<1xf32> {
  %init = linalg.init_tensor [1, 10] : tensor<1x10xf32>
  %0 = linalg.generic
    {indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<1x10xf32>, tensor<1x10xf32>)
    outs(%init : tensor<1x10xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %2 = addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<1x10xf32>
  %1 = linalg.generic
    {indexing_maps = [#map1, #map2],
     iterator_types = ["reduction"]}
    ins(%0 : tensor<1x10xf32>)
    outs(%arg2 : tensor<1xf32>)  {
  ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
    %2 = addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<1xf32>
  return %1 : tensor<1xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0) -> (0, d0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (0)>
//      CHECK: func @consumer_with_reduction(%[[ARG0:.+]]: tensor<1x10xf32>, %[[ARG1:.+]]: tensor<1x10xf32>, %[[ARG2:.+]]: tensor<1xf32>)
//      CHECK:   %[[RES:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP1]]]
// CHECK-SAME:     iterator_types = ["reduction"]
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<1x10xf32>, tensor<1x10xf32>)
//      CHECK:   ^{{.+}}(%[[T0:.+]]: f32, %[[T1:.+]]: f32, %[[T2:.+]]: f32)
//      CHECK:     %[[T3:.+]] = addf %[[T0]], %[[T1]] : f32
//      CHECK:     %[[T4:.+]] = addf %[[T3]], %[[T2]] : f32
//      CHECK:     linalg.yield %[[T4]]
//      CHECK:   return %[[RES]]

// -----

// CHECK-LABEL: func @sigmoid_dynamic_dim(
//       CHECK:   %[[RES:.*]] = linalg.generic
//   CHECK-NOT:   linalg.generic
//       CHECK:   return %[[RES]]
func @sigmoid_dynamic_dim(%0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  %cp5 = constant 5.000000e-01 : f32
  %c0 = constant 0 : index
  %shape = shape.shape_of %0 : tensor<?x1xf32> -> tensor<?xindex>
  %extend = shape.to_extent_tensor %shape : tensor<?xindex> -> tensor<2xindex>
  %extracted = tensor.extract %extend[%c0] : tensor<2xindex>
  %init0 = linalg.init_tensor [%extracted, 1] : tensor<?x1xf32>
  %1 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  }
     outs(%init0 : tensor<?x1xf32>) {
    ^bb0(%a: f32):  // no predecessors
      linalg.yield %cp5 : f32
  } -> tensor<?x1xf32>
  %d0 = memref.dim %0, %c0 : tensor<?x1xf32>
  %init1 = linalg.init_tensor [%d0, 1] : tensor<?x1xf32>
  %2 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  }
      ins(%0, %1 : tensor<?x1xf32>, tensor<?x1xf32>)
     outs(%init1 : tensor<?x1xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):  // no predecessors
      %m = mulf %a, %b : f32
      linalg.yield %m : f32
  } -> tensor<?x1xf32>
  return %2 : tensor<?x1xf32>
}

// -----

func private @compute1(%a: f64) -> f64
func private @compute2(%a: f64, %b: i32) -> i32

// CHECK-LABEL: func @generic_index_op2(
func @generic_index_op2(%arg0: tensor<1x8xf64>, %arg1: tensor<1x8xi32>) -> tensor<1x8xi32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]}
  outs(%arg0 : tensor<1x8xf64>) {
  ^bb0(%a: f64):
    %r = call @compute1(%a) : (f64) -> f64
    linalg.yield %r : f64
  } -> tensor<1x8xf64>

  // CHECK-NEXT:   %[[R:.*]] = linalg.generic
  //      CHECK:     bb0(%[[BBA:[0-9a-z]*]]: f64, %[[BBB:[0-9a-z]*]]: i32):
  // CHECK-NEXT:       %[[A:.*]] = call @compute1(%[[BBA]]) : (f64) -> f64
  // CHECK-NEXT:       %[[B:.*]] = call @compute2(%[[A]], %[[BBB]]) : (f64, i32) -> i32
  // CHECK-NEXT:       linalg.yield %[[B]] : i32
  // CHECK-NEXT:   } -> tensor<1x8xi32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]}
  ins(%0 : tensor<1x8xf64>)
  outs(%arg1 : tensor<1x8xi32>) {
  ^bb0(%a: f64, %b: i32):
    %r = call @compute2(%a, %b) : (f64, i32) -> i32
    linalg.yield %r : i32
  } -> tensor<1x8xi32>

  // CHECK-NEXT:   return %[[R]] : tensor<1x8xi32>
  return %1 : tensor<1x8xi32>
}
