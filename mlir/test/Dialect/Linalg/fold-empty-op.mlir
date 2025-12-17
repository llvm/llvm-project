// RUN: mlir-opt -split-input-file -transform-interpreter %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.linalg.fold_pack_unpack_into_empty
    } : !transform.op<"func.func">
    transform.yield
  }
}

func.func @pack_empty(%arg0: tensor<8x8x32x32xf32>) -> tensor<8x8x32x32xf32> {
  %empty_unpacked = tensor.empty() : tensor<256x256xf32>
  %packed = linalg.pack %empty_unpacked
    inner_dims_pos = [0, 1] inner_tiles = [32, 32]
    into %arg0 : tensor<256x256xf32> -> tensor<8x8x32x32xf32>
  return %packed : tensor<8x8x32x32xf32>
}

// CHECK-LABEL: func.func @pack_empty(
// CHECK-SAME:   %[[T:.+]]: tensor<8x8x32x32xf32>
// CHECK-NOT:    linalg.pack
// CHECK:        return %[[T]] : tensor<8x8x32x32xf32>

func.func @pack_empty_dynamic(%arg0: tensor<?x?x32x32xf32>, %dim0: index, %dim1: index) -> tensor<?x?x32x32xf32> {
  %empty_unpacked = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %packed = linalg.pack %empty_unpacked
    inner_dims_pos = [0, 1] inner_tiles = [32, 32]
    into %arg0 : tensor<?x?xf32> -> tensor<?x?x32x32xf32>
  return %packed : tensor<?x?x32x32xf32>
}

// CHECK-LABEL: func.func @pack_empty_dynamic(
// CHECK-SAME:   %[[T:.+]]: tensor<?x?x32x32xf32>,
// CHECK-SAME:   %[[DIM0:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:   %[[DIM1:[a-zA-Z0-9_]+]]: index
// CHECK-NOT:    linalg.pack
// CHECK:        return %[[T]] : tensor<?x?x32x32xf32>

func.func @unpack_empty(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %empty_packed = tensor.empty() : tensor<8x8x32x32xf32>
  %unpacked = linalg.unpack %empty_packed
    inner_dims_pos = [0, 1] inner_tiles = [32, 32]
    into %arg0 : tensor<8x8x32x32xf32> -> tensor<256x256xf32>
  return %unpacked : tensor<256x256xf32>
}

// CHECK-LABEL: func.func @unpack_empty(
// CHECK-SAME:   %[[T:.+]]: tensor<256x256xf32>
// CHECK-NOT:    linalg.unpack
// CHECK:        return %[[T]] : tensor<256x256xf32>

func.func @unpack_empty_dynamic(%arg0: tensor<?x?xf32>, %dim0: index, %dim1: index) -> tensor<?x?xf32> {
  %empty_packed = tensor.empty(%dim0, %dim1) : tensor<?x?x32x32xf32>
  %unpacked = linalg.unpack %empty_packed
    inner_dims_pos = [0, 1] inner_tiles = [32, 32]
    into %arg0 : tensor<?x?x32x32xf32> -> tensor<?x?xf32>
  return %unpacked : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @unpack_empty_dynamic(
// CHECK-SAME:   %[[T:.+]]: tensor<?x?xf32>,
// CHECK-SAME:   %[[DIM0:[a-zA-Z0-9_]+]]: index,
// CHECK-SAME:   %[[DIM1:[a-zA-Z0-9_]+]]: index
// CHECK-NOT:    linalg.unpack
// CHECK:        return %[[T]] : tensor<?x?xf32>

func.func @pack_padded_empty(%arg0: tensor<8x8x32x32xf32>) -> tensor<8x8x32x32xf32> {
  %pad = arith.constant 1.0 : f32
  %empty_unpacked = tensor.empty() : tensor<256x256xf32>
  %packed = linalg.pack %empty_unpacked
    padding_value(%pad : f32)
    inner_dims_pos = [0, 1] inner_tiles = [32, 32]
    into %arg0 : tensor<256x256xf32> -> tensor<8x8x32x32xf32>
  return %packed : tensor<8x8x32x32xf32>
}

// CHECK-LABEL: func.func @pack_padded_empty(
// CHECK-SAME:   %[[T:.+]]: tensor<8x8x32x32xf32>
// CHECK:        %[[PACK:.+]] = linalg.pack
// CHECK:        return %[[PACK]] : tensor<8x8x32x32xf32>
