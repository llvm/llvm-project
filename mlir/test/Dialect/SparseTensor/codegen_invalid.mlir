// RUN: mlir-opt %s -sparse-tensor-codegen -verify-diagnostics

// NOTE: This test has valid IR, however we are testing whether
// the legalization failure occurs when important passes are
// missing. Notably, using --lower-sparse-ops-to-foreach
// followed by --lower-sparse-foreach-to-scf prior to
// sparse codegen will convert the dense tensor correctly.

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
