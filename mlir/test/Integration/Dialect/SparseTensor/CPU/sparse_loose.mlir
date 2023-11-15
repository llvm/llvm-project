//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparsifier_opts} = enable-runtime-library=true
// DEFINE: %{sparsifier_opts_sve} = enable-arm-sve=true %{sparsifier_opts}
// DEFINE: %{compile} = mlir-opt %s --sparsifier="%{sparsifier_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparsifier="%{sparsifier_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_opts} = -e entry -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | env %{env} %{run} | FileCheck %s

#CSR_hi = #sparse_tensor.encoding<{
  map = (i, j) -> ( i : dense, j : loose_compressed)
}>

module {
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f64
    %d = arith.constant dense<[[ 1.0,  2.0,  3.0,  4.0 ],
                               [ 5.0,  6.0,  7.0,  8.0 ],
                               [ 0.0,  0.0,  5.5,  0.0 ],
                               [ 9.0, 10.0, 11.0, 12.0 ],
                               [13.0, 14.0, 15.0, 16.0 ]]> : tensor<5x4xf64>
    %s = sparse_tensor.convert %d : tensor<5x4xf64> to tensor<5x4xf64, #CSR_hi>

    //
    // CHECK:      ( 0, 4, 4, 8, 8, 9, 9, 13 )
    // CHECK-NEXT: ( 0, 1, 2, 3, 0, 1, 2, 3, 2, 0, 1, 2, 3, 0, 1, 2, 3 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 5.5, 9, 10, 11, 12, 13, 14, 15, 16 )
    //
    %pos = sparse_tensor.positions %s {level = 1 : index } : tensor<5x4xf64, #CSR_hi> to memref<?xindex>
    %vecp = vector.transfer_read %pos[%c0], %c0 : memref<?xindex>, vector<8xindex>
    vector.print %vecp : vector<8xindex>
    %crd = sparse_tensor.coordinates %s {level = 1 : index } : tensor<5x4xf64, #CSR_hi> to memref<?xindex>
    %vecc = vector.transfer_read %crd[%c0], %c0 : memref<?xindex>, vector<17xindex>
    vector.print %vecc : vector<17xindex>
    %val = sparse_tensor.values %s : tensor<5x4xf64, #CSR_hi> to memref<?xf64>
    %vecv = vector.transfer_read %val[%c0], %f0 : memref<?xf64>, vector<17xf64>
    vector.print %vecv : vector<17xf64>

    // Release the resources.
    bufferization.dealloc_tensor %s: tensor<5x4xf64, #CSR_hi>

    return
  }
}
