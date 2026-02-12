// RUN: mlir-opt %s -sparse-tensor-codegen -verify-diagnostics

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

module {
  func.func @main() -> tensor<8xf32, #SparseVector> {
    %dense = arith.constant dense<[1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0]>
      : tensor<8xf32>

    // expected-error@+1 {{failed to legalize operation 'sparse_tensor.convert' that was explicitly marked illegal}}
    %sparse = sparse_tensor.convert %dense
      : tensor<8xf32> to tensor<8xf32, #SparseVector>

    return %sparse : tensor<8xf32, #SparseVector>
  }
}

// -----
