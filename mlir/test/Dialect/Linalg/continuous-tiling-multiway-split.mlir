// RUN: mlir-opt --transform-interpreter --canonicalize --split-input-file %s | FileCheck %s

// This tests the results of continuous_tile_sizes on multiway splitOp.
// continuous_tile_sizes returns a list of tile-sizes and a list of split points.
// The list of split points is consumed by splitOp to split the linalg.matmul op
// along dimension 1 to produce as many split-up linalg.matmul ops.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiles, %splits = transform.structured.continuous_tile_sizes %0 { dimension = 1, target_size = 9} : (!transform.any_op) -> !transform.any_op
    %low, %high = transform.structured.split %0 after %splits { dimension = 1, multiway } : !transform.any_op, !transform.any_op
    transform.yield
  }
}

func.func @continuous_tile_linalg_matmul(
  %arg0: tensor<25x34xf32>, %arg1: tensor<34x25xf32>, %arg2: tensor<25x25xf32>)
    -> tensor<25x25xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<25x34xf32>, tensor<34x25xf32>)
                     outs(%arg2: tensor<25x25xf32>)
    -> tensor<25x25xf32>

  return %0 : tensor<25x25xf32>
}

// CHECK-LABEL: @continuous_tile_linalg_matmul
// CHECK-SAME: %[[IN1:.+]]: tensor<25x34xf32>, %[[IN2:.+]]: tensor<34x25xf32>, %[[OUT:.+]]: tensor<25x25xf32>
// CHECK:      %[[SLICE:.+]] = tensor.extract_slice %[[IN2]][0, 0] [34, 18] [1, 1] : tensor<34x25xf32> to tensor<34x18xf32>
// CHECK       %[[SLICE0:.+]] = tensor.extract_slice %[[OUT]][0, 0] [25, 18] [1, 1] : tensor<25x25xf32> to tensor<25x18xf32>
// CHECK       %[[MM0:.+]] = linalg.matmul ins(%[[IN1]], %[[SLICE]] : tensor<25x34xf32>, tensor<34x18xf32>) outs(%[[SLICE0]] : tensor<25x18xf32>) -> tensor<25x18xf32>
// CHECK       %[[INSLICE:.+]] = tensor.insert_slice %[[MM0]] into %[[OUT]][0, 0] [25, 18] [1, 1] : tensor<25x18xf32> into tensor<25x25xf32>
// CHECK       %[[SLICE1]] = tensor.extract_slice %[[IN2]][0, 18] [34, 7] [1, 1] : tensor<34x25xf32> to tensor<34x7xf32>
// CHECK       %[[SLICE2]] = tensor.extract_slice %[[INSLICE]][0, 18] [25, 7] [1, 1] : tensor<25x25xf32> to tensor<25x7xf32>
// CHECK       %[[SLICE3]] = tensor.extract_slice %[[SLICE1]][0, 0] [34, 4] [1, 1] : tensor<34x7xf32> to tensor<34x4xf32>
// CHECK       %[[SLICE4]] = tensor.extract_slice %[[SLICE2]][0, 0] [25, 4] [1, 1] : tensor<25x7xf32> to tensor<25x4xf32>
// CHECK       %[[MM1:.+]] = linalg.matmul ins(%[[IN1]], %[[SLICE3]] : tensor<25x34xf32>, tensor<34x4xf32>) outs(%[[SLICE4]] : tensor<25x4xf32>) -> tensor<25x4xf32>
// CHECK       %[[INSLICE0:.+]] = tensor.insert_slice %[[MM1]] into %[[SLICE2]][0, 0] [25, 4] [1, 1] : tensor<25x4xf32> into tensor<25x7xf32>
// CHECK       %[[SLICE5]] = tensor.extract_slice %[[SLICE1]][0, 4] [34, 3] [1, 1] : tensor<34x7xf32> to tensor<34x3xf32>
// CHECK       %[[SLICE6]] = tensor.extract_slice %[[INSLICE0]][0, 4] [25, 3] [1, 1] : tensor<25x7xf32> to tensor<25x3xf32>
// CHECK       %[[SLICE7]] = tensor.extract_slice %[[SLICE5]][0, 0] [34, 2] [1, 1] : tensor<34x3xf32> to tensor<34x2xf32>
// CHECK       %[[SLICE8]] = tensor.extract_slice %[[SLICE6]][0, 0] [25, 2] [1, 1] : tensor<25x3xf32> to tensor<25x2xf32>
// CHECK       %[[MM2:.+]] = linalg.matmul ins(%[[IN1]], %[[SLICE7]] : tensor<25x34xf32>, tensor<34x2xf32>) outs(%[[SLICE8]] : tensor<25x2xf32>) -> tensor<25x2xf32>
// CHECK       %[[INSLICE1:.+]] = tensor.insert_slice %[[MM2]] into %[[SLICE6]][0, 0] [25, 2] [1, 1] : tensor<25x2xf32> into tensor<25x3xf32>
// CHECK       %[[SLICE9]] = tensor.extract_slice %[[SLICE5]][0, 2] [34, 1] [1, 1] : tensor<34x3xf32> to tensor<34x1xf32>
// CHECK       %[[SLICE10]] = tensor.extract_slice %[[INSLICE1]][0, 2] [25, 1] [1, 1] : tensor<25x3xf32> to tensor<25x1xf32>
// CHECK       %[[MM3:.+]] = linalg.matmul ins(%[[IN1]], %[[SLICE9]] : tensor<25x34xf32>, tensor<34x1xf32>) outs(%[[SLICE10]] : tensor<25x1xf32>) -> tensor<25x1xf32>
// CHECK       %[[INSLICE2]] = tensor.insert_slice %[[MM3]] into %[[INSLICE1]][0, 2] [25, 1] [1, 1] : tensor<25x1xf32> into tensor<25x3xf32>
// CHECK       %[[INSLICE3]] = tensor.insert_slice %[[INSLICE2]] into %[[INSLICE0]][0, 4] [25, 3] [1, 1] : tensor<25x3xf32> into tensor<25x7xf32>
// CHECK       %[[INSLICE4]] = tensor.insert_slice %[[INSLICE3]] into %[[INSLICE]][0, 18] [25, 7] [1, 1] : tensor<25x7xf32> into tensor<25x25xf32>
// CHECK       return %[[INSLICE4]] : tensor<25x25xf32>

// -----

// Tests the same as above except that the !transform.param<i64> output type in
// continuous_tile_sizes op triggers tile sizes and split points to be computed
// statically and not dynamically.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiles, %splits = transform.structured.continuous_tile_sizes %0 { dimension = 1, target_size = 9} : (!transform.any_op) -> !transform.param<i64>
    %low, %high = transform.structured.split %0 after %splits { dimension = 1, multiway } : !transform.any_op, !transform.param<i64>
    transform.yield
  }
}

func.func @continuous_tile_static_linalg_matmul(
  %arg0: tensor<25x34xf32>, %arg1: tensor<34x25xf32>, %arg2: tensor<25x25xf32>)
    -> tensor<25x25xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<25x34xf32>, tensor<34x25xf32>)
                     outs(%arg2: tensor<25x25xf32>)
    -> tensor<25x25xf32>

  return %0 : tensor<25x25xf32>
}

// CHECK-LABEL: @continuous_tile_static_linalg_matmul
// CHECK-SAME: %[[IN1:.+]]: tensor<25x34xf32>, %[[IN2:.+]]: tensor<34x25xf32>, %[[OUT:.+]]: tensor<25x25xf32>
// CHECK:      %[[SLICE:.+]] = tensor.extract_slice %[[IN2]][0, 0] [34, 18] [1, 1] : tensor<34x25xf32> to tensor<34x18xf32>
// CHECK       %[[SLICE0:.+]] = tensor.extract_slice %[[OUT]][0, 0] [25, 18] [1, 1] : tensor<25x25xf32> to tensor<25x18xf32>
// CHECK       %[[MM0:.+]] = linalg.matmul ins(%[[IN1]], %[[SLICE]] : tensor<25x34xf32>, tensor<34x18xf32>) outs(%[[SLICE0]] : tensor<25x18xf32>) -> tensor<25x18xf32>
// CHECK       %[[INSLICE:.+]] = tensor.insert_slice %[[MM0]] into %[[OUT]][0, 0] [25, 18] [1, 1] : tensor<25x18xf32> into tensor<25x25xf32>
// CHECK       %[[SLICE1]] = tensor.extract_slice %[[IN2]][0, 18] [34, 7] [1, 1] : tensor<34x25xf32> to tensor<34x7xf32>
// CHECK       %[[SLICE2]] = tensor.extract_slice %[[INSLICE]][0, 18] [25, 7] [1, 1] : tensor<25x25xf32> to tensor<25x7xf32>
// CHECK       %[[SLICE3]] = tensor.extract_slice %[[SLICE1]][0, 0] [34, 4] [1, 1] : tensor<34x7xf32> to tensor<34x4xf32>
// CHECK       %[[SLICE4]] = tensor.extract_slice %[[SLICE2]][0, 0] [25, 4] [1, 1] : tensor<25x7xf32> to tensor<25x4xf32>
// CHECK       %[[MM1:.+]] = linalg.matmul ins(%[[IN1]], %[[SLICE3]] : tensor<25x34xf32>, tensor<34x4xf32>) outs(%[[SLICE4]] : tensor<25x4xf32>) -> tensor<25x4xf32>
// CHECK       %[[INSLICE0:.+]] = tensor.insert_slice %[[MM1]] into %[[SLICE2]][0, 0] [25, 4] [1, 1] : tensor<25x4xf32> into tensor<25x7xf32>
// CHECK       %[[SLICE5]] = tensor.extract_slice %[[SLICE1]][0, 4] [34, 3] [1, 1] : tensor<34x7xf32> to tensor<34x3xf32>
// CHECK       %[[SLICE6]] = tensor.extract_slice %[[INSLICE0]][0, 4] [25, 3] [1, 1] : tensor<25x7xf32> to tensor<25x3xf32>
// CHECK       %[[SLICE7]] = tensor.extract_slice %[[SLICE5]][0, 0] [34, 2] [1, 1] : tensor<34x3xf32> to tensor<34x2xf32>
// CHECK       %[[SLICE8]] = tensor.extract_slice %[[SLICE6]][0, 0] [25, 2] [1, 1] : tensor<25x3xf32> to tensor<25x2xf32>
// CHECK       %[[MM2:.+]] = linalg.matmul ins(%[[IN1]], %[[SLICE7]] : tensor<25x34xf32>, tensor<34x2xf32>) outs(%[[SLICE8]] : tensor<25x2xf32>) -> tensor<25x2xf32>
// CHECK       %[[INSLICE1:.+]] = tensor.insert_slice %[[MM2]] into %[[SLICE6]][0, 0] [25, 2] [1, 1] : tensor<25x2xf32> into tensor<25x3xf32>
// CHECK       %[[SLICE9]] = tensor.extract_slice %[[SLICE5]][0, 2] [34, 1] [1, 1] : tensor<34x3xf32> to tensor<34x1xf32>
// CHECK       %[[SLICE10]] = tensor.extract_slice %[[INSLICE1]][0, 2] [25, 1] [1, 1] : tensor<25x3xf32> to tensor<25x1xf32>
// CHECK       %[[MM3:.+]] = linalg.matmul ins(%[[IN1]], %[[SLICE9]] : tensor<25x34xf32>, tensor<34x1xf32>) outs(%[[SLICE10]] : tensor<25x1xf32>) -> tensor<25x1xf32>
// CHECK       %[[INSLICE2]] = tensor.insert_slice %[[MM3]] into %[[INSLICE1]][0, 2] [25, 1] [1, 1] : tensor<25x1xf32> into tensor<25x3xf32>
// CHECK       %[[INSLICE3]] = tensor.insert_slice %[[INSLICE2]] into %[[INSLICE0]][0, 4] [25, 3] [1, 1] : tensor<25x3xf32> into tensor<25x7xf32>
// CHECK       %[[INSLICE4]] = tensor.insert_slice %[[INSLICE3]] into %[[INSLICE]][0, 18] [25, 7] [1, 1] : tensor<25x7xf32> into tensor<25x25xf32>
// CHECK       return %[[INSLICE4]] : tensor<25x25xf32>
