// RUN: mlir-opt -split-input-file -transform-interpreter %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.tensor.fold_tensor_empty
    } : !transform.op<"func.func">
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 28)>
// CHECK: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0 * 28)>

func.func @empty_reshape_expansion(%arg0 : index, %sz0: index) -> tensor<2x3x5x4x?x7xf32> {
  %0 = tensor.empty(%arg0) : tensor<6x5x?xf32>
  %1 = tensor.expand_shape %0 [[0, 1], [2], [3, 4, 5]] output_shape [2, 3, 5, 4, %sz0, 7] : tensor<6x5x?xf32> into tensor<2x3x5x4x?x7xf32>
  return %1 : tensor<2x3x5x4x?x7xf32>
}
// CHECK-LABEL: func @empty_reshape_expansion
// CHECK-SAME:     %[[ARG0:.+]]: index
// CHECK:        %[[OLD_INIT:.+]] = tensor.empty(%{{.*}}) : tensor<6x5x?xf32>
// CHECK-NEXT:   %[[DIM:.*]] = tensor.dim %[[OLD_INIT]]
// CHECK-NEXT:   %[[D:.+]] = affine.apply #[[$MAP]]()[%[[DIM]]]
// CHECK-NEXT:   %[[INIT:.+]] = tensor.empty(%[[D]])
// CHECK-NEXT:   return %[[INIT]]

func.func @empty_reshape_collapse(%arg0 : index) -> tensor<6x5x?xf32> {
  %0 = tensor.empty(%arg0) : tensor<2x3x5x4x?x7xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2], [3, 4, 5]]
      : tensor<2x3x5x4x?x7xf32> into tensor<6x5x?xf32>
  return %1 : tensor<6x5x?xf32>
}
// CHECK-LABEL: func @empty_reshape_collapse
// CHECK-SAME:     %[[ARG0:.+]]: index
// CHECK:        %[[OLD_INIT:.+]] = tensor.empty(%{{.*}}) : tensor<2x3x5x4x?x7xf32>
// CHECK-NEXT:   %[[DIM:.*]] = tensor.dim %[[OLD_INIT]]
// CHECK-NEXT:   %[[D:.+]] = affine.apply #[[$MAP2]]()[%[[DIM]]]
// CHECK-NEXT:   %[[INIT:.+]] = tensor.empty(%[[D]])
// CHECK-NEXT:   return %[[INIT]]

func.func @fold_empty_tensor_with_slice
  (%arg0 : index, %arg1 : index) -> tensor<5x?x20xf32>
{
  %0 = tensor.empty(%arg0) : tensor<?x10x40xf32>
  %1 = tensor.extract_slice %0[0, 0, 0] [5, %arg1, 20] [1, 1, 1]
    : tensor<?x10x40xf32> to tensor<5x?x20xf32>
  return %1 : tensor<5x?x20xf32>
}
// CHECK-LABEL: func @fold_empty_tensor_with_slice
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//      CHECK:   %[[T0:.+]] = tensor.empty(%[[ARG1]])
//      CHECK:   return %[[T0]]

// CHECK-LABEL: func @rank_reducing_empty_tensor_extract
func.func @rank_reducing_empty_tensor_extract(%sz : index, %idx : index) -> tensor<2xf32> {
  // CHECK: tensor.empty() : tensor<2xf32>
  %a = tensor.empty(%sz) : tensor<?x2xf32>

  // CHECK-NOT: extract
  %r = tensor.extract_slice %a[%idx, 0] [1, 2] [1, 1] : tensor<?x2xf32> to tensor<2xf32>
  return %r: tensor<2xf32>
}

func.func @pack_empty(%arg0: tensor<8x8x32x32xf32>) -> tensor<8x8x32x32xf32> {
  %empty_unpacked = tensor.empty() : tensor<256x256xf32>
  %packed = tensor.pack %empty_unpacked
    inner_dims_pos = [0, 1] inner_tiles = [32, 32]
    into %arg0 : tensor<256x256xf32> -> tensor<8x8x32x32xf32>
  return %packed : tensor<8x8x32x32xf32>
}

// CHECK-LABEL: func.func @pack_empty(
// CHECK-SAME:   %[[T:.+]]: tensor<8x8x32x32xf32>
// CHECK-NOT:    tensor.pack
// CHECK:        return %[[T]] : tensor<8x8x32x32xf32>

func.func @pack_empty_dynamic(%arg0: tensor<?x?x?x?xf32>, %dim0: index, %dim1: index) -> tensor<?x?x?x?xf32> {
  %empty_unpacked = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %packed = tensor.pack %empty_unpacked
    inner_dims_pos = [0, 1] inner_tiles = [32, 32]
    into %arg0 : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  return %packed : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func.func @pack_empty_dynamic(
// CHECK-SAME:   %[[T:.+]]: tensor<?x?x?x?xf32>,
// CHECK-SAME:   %[[DIM0:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:   %[[DIM1:[a-zA-Z0-9_]+]]: index
// CHECK-NOT:    tensor.pack
// CHECK:        return %[[T]] : tensor<?x?x?x?xf32>

func.func @unpack_empty(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %empty_packed = tensor.empty() : tensor<8x8x32x32xf32>
  %unpacked = tensor.unpack %empty_packed
    inner_dims_pos = [0, 1] inner_tiles = [32, 32]
    into %arg0 : tensor<8x8x32x32xf32> -> tensor<256x256xf32>
  return %unpacked : tensor<256x256xf32>
}

// CHECK-LABEL: func.func @unpack_empty(
// CHECK-SAME:   %[[T:.+]]: tensor<256x256xf32>
// CHECK-NOT:    tensor.unpack
// CHECK:        return %[[T]] : tensor<256x256xf32>

func.func @unpack_empty_dynamic(%arg0: tensor<?x?xf32>, %dim0: index, %dim1: index, %dim2: index, %dim3: index) -> tensor<?x?xf32> {
  %empty_packed = tensor.empty(%dim0, %dim1, %dim2, %dim3) : tensor<?x?x?x?xf32>
  %unpacked = tensor.unpack %empty_packed
    inner_dims_pos = [0, 1] inner_tiles = [32, 32]
    into %arg0 : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  return %unpacked : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @unpack_empty_dynamic(
// CHECK-SAME:   %[[T:.+]]: tensor<?x?xf32>,
// CHECK-SAME:   %[[DIM0:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:   %[[DIM1:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:   %[[DIM2:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:   %[[DIM3:[a-zA-Z0-9_]+]]: index
// CHECK-NOT:    tensor.unpack
// CHECK:        return %[[T]] : tensor<?x?xf32>

func.func @pack_padded_empty(%arg0: tensor<8x8x32x32xf32>) -> tensor<8x8x32x32xf32> {
  %pad = arith.constant 1.0 : f32
  %empty_unpacked = tensor.empty() : tensor<256x256xf32>
  %packed = tensor.pack %empty_unpacked
    padding_value(%pad : f32)
    inner_dims_pos = [0, 1] inner_tiles = [32, 32]
    into %arg0 : tensor<256x256xf32> -> tensor<8x8x32x32xf32>
  return %packed : tensor<8x8x32x32xf32>
}

// CHECK-LABEL: func.func @pack_padded_empty(
// CHECK-SAME:   %[[T:.+]]: tensor<8x8x32x32xf32>
// CHECK:        %[[PACK:.+]] = tensor.pack
// CHECK:        return %[[PACK]] : tensor<8x8x32x32xf32>

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.tensor.fold_tensor_empty
          {fold_single_use_only = true}
    } : !transform.op<"func.func">
    transform.yield
  }
}

func.func @double_use_of_tensor_empty(%arg0: index, %arg1: index)
    -> (tensor<5x?x20xf32>, tensor<5x?x20xf32>)
{
  %0 = tensor.empty(%arg0) : tensor<?x10x40xf32>
  %1 = tensor.extract_slice %0[0, 0, 0] [5, %arg1, 20] [1, 1, 1]
    : tensor<?x10x40xf32> to tensor<5x?x20xf32>
  %2 = tensor.extract_slice %0[1, 1, 1] [5, %arg1, 20] [1, 1, 1]
    : tensor<?x10x40xf32> to tensor<5x?x20xf32>
  return %1, %2 : tensor<5x?x20xf32>, tensor<5x?x20xf32>
}
// CHECK-LABEL: func @double_use_of_tensor_empty(
//       CHECK:   tensor.empty{{.*}} : tensor<?x10x40xf32>
//       CHECK:   tensor.extract_slice
//       CHECK:   tensor.extract_slice

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.tensor.fold_tensor_empty
    } : !transform.op<"func.func">
    transform.yield
  }
}

func.func @concats_of_empty(
    %arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index)
    -> tensor<5x?x?xf32>
{
  %0 = tensor.empty(%arg0, %arg1) : tensor<5x?x?xf32>
  %1 = tensor.empty(%arg2, %arg3) : tensor<5x?x?xf32>
  %2 = tensor.concat dim(1) %0, %1 : (tensor<5x?x?xf32>, tensor<5x?x?xf32>) -> tensor<5x?x?xf32>
  return %2 : tensor<5x?x?xf32>
}
//       CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func @concats_of_empty(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index)
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[EMPTY0:.+]] = tensor.empty(%[[ARG0]], %[[ARG1]])
//   CHECK-DAG:   %[[EMPTY1:.+]] = tensor.empty(%[[ARG2]], %[[ARG3]])
//       CHECK:   %[[D2:.+]] = tensor.dim %[[EMPTY0]], %[[C2]]
//   CHECK-DAG:   %[[D0_1:.+]] = tensor.dim %[[EMPTY0]], %[[C1]]
//   CHECK-DAG:   %[[D1_1:.+]] = tensor.dim %[[EMPTY1]], %[[C1]]
//   CHECK-DAG:   %[[SUM:.+]] = affine.apply #[[MAP]]()[%[[D0_1]], %[[D1_1]]]
//       CHECK:   %[[NEW_EMPTY:.+]] = tensor.empty(%[[SUM]], %[[D2]])
//       CHECK:   return %[[NEW_EMPTY]]
