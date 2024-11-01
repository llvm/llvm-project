// RUN: mlir-opt %s -one-shot-bufferize -verify-diagnostics

#SparseVector = #sparse_tensor.encoding<{
  lvlTypes = ["compressed"]
}>

func.func @sparse_tensor_op(%arg0: tensor<64xf32, #SparseVector>) -> tensor<64xf32, #SparseVector> {
  // expected-error @below{{sparse_tensor ops must be bufferized with the sparse compiler}}
  // expected-error @below{{failed to bufferize op}}
  %0 = sparse_tensor.convert %arg0 : tensor<64xf32, #SparseVector> to tensor<64xf32, #SparseVector>
  return %0 : tensor<64xf32, #SparseVector>
}
