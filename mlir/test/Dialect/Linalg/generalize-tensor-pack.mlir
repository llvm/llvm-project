// RUN: mlir-opt -split-input-file --test-linalg-transform-patterns="test-generalize-tensor-pack"  %s | FileCheck %s

func.func @simple_KCRS_to_KCRSsr(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<1x1x1x1x8x32xf32>) -> tensor<1x1x1x1x8x32xf32> {
  %0 = tensor.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x32x8xf32> -> tensor<1x1x1x1x8x32xf32>
  return %0 : tensor<1x1x1x1x8x32xf32>
}
// CHECK-LABEL: func.func @simple_KCRS_to_KCRSsr
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<8x32xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[TILE]] : tensor<32x8xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<8x32xf32>)
// CHECK-SAME:      permutation = [1, 0]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

func.func @simple_pad_and_pack(%input: tensor<5x1xf32>, %output: tensor<1x1x8x2xf32>, %pad: f32) -> tensor<1x1x8x2xf32> {
  %0 = tensor.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<5x1xf32> -> tensor<1x1x8x2xf32>
  return %0 : tensor<1x1x8x2xf32>
}
// CHECK-LABEL: func.func @simple_pad_and_pack
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[PAD_VAL:[a-zA-Z0-9]+]]
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         %[[PAD:.+]] = tensor.pad %[[SRC]] low[%[[C0]], %[[C0]]] high[%[[C3]], %[[C1]]]
// CHECK:           tensor.yield %[[PAD_VAL]]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<8x2xf32>
// CHECK:         %[[TRANSP:.+]] = linalg.transpose
// CHECK-SAME:      ins(%[[PAD]] : tensor<8x2xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<8x2xf32>)
// CHECK-SAME:      permutation = [0, 1]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

func.func @simple_NC_to_CNnc(%arg0: tensor<32x8xf32>, %arg1: tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32>{
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<32x8xf32> -> tensor<1x1x32x8xf32>
  return %0 : tensor<1x1x32x8xf32>
}
// CHECK-LABEL: func.func @simple_NC_to_CNnc
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[SRC]] : tensor<32x8xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<32x8xf32>)
// CHECK-SAME:      permutation = [0, 1]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:         return %[[INSERT]]
