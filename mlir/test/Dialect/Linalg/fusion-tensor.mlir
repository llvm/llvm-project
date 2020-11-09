// RUN: mlir-opt %s -linalg-fusion-for-tensor-ops -split-input-file | FileCheck %s

// CHECK-DAG: [[$MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @add_mul_fusion
func @add_mul_fusion(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %0 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = addf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP0]], [[$MAP0]], [[$MAP0]], [[$MAP0]]{{\]}}
  %2 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) {
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
    } -> tensor<?x?xf32>
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
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = addf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP0]], [[$MAP1]], [[$MAP0]], [[$MAP0]]{{\]}}
  %2 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%arg5: f32, %arg6: f32):       // no predecessors
      %3 = mulf %arg5, %arg6 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
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
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = addf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP1]], [[$MAP0]], [[$MAP0]], [[$MAP0]]{{\]}}
  %2 = linalg.generic {indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%arg5: f32, %arg6: f32):       // no predecessors
      %3 = mulf %arg5, %arg6 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
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
  %0 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]}
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = addf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?xf32>
  // CHECK: linalg.generic {
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP1]], [[$MAP1]], [[$MAP0]], [[$MAP0]]
  %2 = linalg.generic {indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg2 : tensor<?xf32>, tensor<?x?xf32>) {
    ^bb0(%arg5: f32, %arg6: f32):       // no predecessors
      %3 = mulf %arg5, %arg6 : f32
      linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<() -> ()>
#map0 = affine_map<() -> ()>

// CHECK-LABEL: @add_mul_scalar_fusion
func @add_mul_scalar_fusion(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32>
{
  %0 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []}
      ins(%arg0, %arg1 : tensor<f32>, tensor<f32>) {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = addf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<f32>
  // CHECK: linalg.generic {
  // CHECK: addf
  // CHECK: mulf
  %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []}
      ins(%0, %arg2 : tensor<f32>, tensor<f32>) {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<f32>

  return %1 : tensor<f32>
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_op_constant_fusion(%arg0 : tensor<5x?x?xf32>) -> tensor<5x?x?xf32>
{
  %0 = constant dense<42.0> : tensor<5xf32>
  %1 = linalg.generic {
         indexing_maps = [#map0, #map1, #map1],
         iterator_types = ["parallel", "parallel", "parallel"]}
         ins(%0, %arg0 : tensor<5xf32>, tensor<5x?x?xf32>) {
       ^bb0(%arg1: f32, %arg2: f32):
         %2 = mulf %arg1, %arg2 : f32
         linalg.yield %2 : f32
       } -> tensor<5x?x?xf32>
  return %1 : tensor<5x?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @generic_op_constant_fusion
//       CHECK:   %[[CST:.*]] = constant {{.*}} : f32
//       CHECK:   linalg.generic
//       CHECK:   ^{{.*}}(%[[ARG1:.*]]: f32)
//       CHECK:     mulf %[[CST]], %[[ARG1]]

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @indexed_generic_op_constant_fusion(%arg0 : tensor<5x?x?xf32>)
                                         -> tensor<5x?x?xf32>
{
  %0 = constant dense<42.0> : tensor<5xf32>
  %1 = linalg.indexed_generic {
         indexing_maps = [#map0, #map1, #map1],
         iterator_types = ["parallel", "parallel", "parallel"]}
         ins(%0, %arg0 : tensor<5xf32>, tensor<5x?x?xf32>) {
       ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: f32, %arg5 : f32):
         %2 = mulf %arg4, %arg5 : f32
         linalg.yield %2 : f32
       } -> tensor<5x?x?xf32>
  return %1 : tensor<5x?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @indexed_generic_op_constant_fusion
//       CHECK:   %[[CST:.*]] = constant {{.*}} : f32
//       CHECK:   linalg.indexed_generic
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
  %1 = linalg.generic {
         indexing_maps = [#map0, #map1, #map1],
         iterator_types = ["parallel", "parallel", "parallel"]}
         ins(%0, %arg0 : tensor<f32>, tensor<5x?x?xf32>) {
       ^bb0(%arg1: f32, %arg2: f32):
         %2 = mulf %arg1, %arg2 : f32
         linalg.yield %2 : f32
       } -> tensor<5x?x?xf32>
  return %1 : tensor<5x?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @generic_op_zero_dim_constant_fusion
//       CHECK:   %[[CST:.*]] = constant {{.*}} : f32
//       CHECK:   linalg.generic
//       CHECK:   ^{{.*}}(%[[ARG1:.*]]: f32)
//       CHECK:     mulf %[[CST]], %[[ARG1]]

// -----

#map0 = affine_map<(d0, d1, d2) -> ()>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @indexed_generic_op_zero_dim_constant_fusion
  (%arg0 : tensor<5x?x?xf32>) -> tensor<5x?x?xf32>
{
  %0 = constant dense<42.0> : tensor<f32>
  %1 = linalg.indexed_generic {
         indexing_maps = [#map0, #map1, #map1],
         iterator_types = ["parallel", "parallel", "parallel"]}
         ins(%0, %arg0 : tensor<f32>, tensor<5x?x?xf32>) {
       ^bb0(%arg1 : index, %arg2 : index, %arg3 : index, %arg4: f32, %arg5: f32):
         %2 = mulf %arg4, %arg5 : f32
         linalg.yield %2 : f32
       } -> tensor<5x?x?xf32>
  return %1 : tensor<5x?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @indexed_generic_op_zero_dim_constant_fusion
//       CHECK:   %[[CST:.*]] = constant {{.*}} : f32
//       CHECK:   linalg.indexed_generic
//       CHECK:   ^{{[a-zA-Z0-9_]*}}
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]*]]: index
//  CHECK-SAME:     %[[ARG4:.*]]: f32)
//       CHECK:     mulf %[[CST]], %[[ARG4]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func @generic_op_indexed_generic_op_fusion(%arg0: tensor<?x?xi32>,
                                           %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %0 = linalg.generic {
      indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"] }
      ins(%arg0, %arg1  : tensor<?x?xi32>, tensor<?x?xi32>) {
    ^bb0(%arg2: i32, %arg3: i32):       // no predecessors
      %10 = addi %arg2, %arg3 : i32
      linalg.yield %10 : i32
    } -> tensor<?x?xi32>
    %1 = linalg.indexed_generic {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel", "parallel"] }
      ins(%0 : tensor<?x?xi32>) {
    ^bb0(%arg2: index, %arg3: index, %arg4: i32):       // no predecessors
      %2 = index_cast %arg2 : index to i32
      %3 = index_cast %arg3 : index to i32
      %4 = addi %arg4, %2 : i32
      %5 = subi %4, %3 : i32
      linalg.yield %5 : i32
    } -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
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
  %0 = linalg.indexed_generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel"] }
    ins(%arg0 : tensor<?x?xi32>) {
  ^bb0(%arg2: index, %arg3: index, %arg4: i32):       // no predecessors
    %2 = index_cast %arg2 : index to i32
    %3 = index_cast %arg3 : index to i32
    %4 = addi %arg4, %2 : i32
    %5 = subi %4, %3 : i32
    linalg.yield %5 : i32
  } -> tensor<?x?xi32>
  %1 = linalg.generic {
    indexing_maps = [#map0, #map0, #map0],
    iterator_types = ["parallel", "parallel"] }
    ins(%0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) {
  ^bb0(%arg2: i32, %arg3: i32):       // no predecessors
    %10 = addi %arg2, %arg3 : i32
    linalg.yield %10 : i32
  } -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
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
    %0 = linalg.indexed_generic {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel", "parallel"] }
      ins(%arg0 : tensor<?x?xi32>) {
    ^bb0(%arg2: index, %arg3: index, %arg4: i32):       // no predecessors
      %2 = index_cast %arg2 : index to i32
      %3 = index_cast %arg3 : index to i32
      %4 = addi %arg4, %2 : i32
      %5 = subi %4, %3 : i32
      linalg.yield %5 : i32
    } -> tensor<?x?xi32>
    %1 = linalg.indexed_generic {
      indexing_maps = [#map1, #map1],
      iterator_types = ["parallel", "parallel"] }
      ins(%0 : tensor<?x?xi32>) {
    ^bb0(%arg2: index, %arg3: index, %arg4: i32):       // no predecessors
      %2 = index_cast %arg2 : index to i32
      %3 = index_cast %arg3 : index to i32
      %4 = addi %arg4, %2 : i32
      %5 = subi %4, %3 : i32
      linalg.yield %5 : i32
    } -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
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
//      CHECK:   %[[VAL2:.+]] = subi %[[VAL1]], %[[SUB_OPERAND1]] : i32
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
  %0 = linalg.indexed_generic
    {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>],
     iterator_types = []}
    ins(%arg1 : tensor<i32>) {
    ^bb0(%arg2: i32):  // no predecessors
      %3 = index_cast %arg2 : i32 to index
      %4 = extract_element %arg0[%3, %c0, %c0] : tensor<5x1x1xf32>
      linalg.yield %4 : f32
    } -> tensor<f32>
  %1 = linalg.generic
   {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%0, %cst : tensor<f32>, tensor<10xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      %3 = mulf %arg2, %arg3 : f32
      linalg.yield %3 : f32
    } -> tensor<10xf32>
  return %1 : tensor<10xf32>
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
//       CHECK:     extract_element %[[ARG0]]
//       CHECK:     linalg.yield
//       CHECK   return %[[T0]]
