// RUN: mlir-opt %s -split-input-file | mlir-opt | FileCheck %s

// CHECK-LABEL: func private @sparse_1d_tensor(
// CHECK-SAME: tensor<32xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>>)
func private @sparse_1d_tensor(tensor<32xf64, #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>>)

// -----

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL: func private @sparse_2d_tensor(
// CHECK-SAME: tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d0, d1)>, pointerBitWidth = 64, indexBitWidth = 64 }>>)
func private @sparse_2d_tensor(tensor<?x?xf32, #CSR>)
