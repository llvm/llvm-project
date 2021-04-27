// RUN: mlir-opt <%s | mlir-opt | FileCheck %s

// CHECK: func private @sparse_1d_tensor(tensor<32xf64, #tensor.sparse<{sparseDimLevelType = ["compressed"]}>>)
func private @sparse_1d_tensor(tensor<32xf64, #tensor.sparse<{sparseDimLevelType = ["compressed"]}>>)

#CSR = #tensor.sparse<{
  sparseDimLevelType = [ "dense", "compressed" ],
  sparseDimOrdering = affine_map<(i,j) -> (i,j)>,
  sparseIndexBitWidth = 64,
  sparsePointerBitWidth = 64
}>

// CHECK: func private @sparse_2d_tensor(tensor<?x?xf32, #tensor.sparse<{sparseDimLevelType = ["dense", "compressed"], sparseDimOrdering = affine_map<(d0, d1) -> (d0, d1)>, sparseIndexBitWidth = 64 : i64, sparsePointerBitWidth = 64 : i64}>>)
func private @sparse_2d_tensor(tensor<?x?xf32, #CSR>)
