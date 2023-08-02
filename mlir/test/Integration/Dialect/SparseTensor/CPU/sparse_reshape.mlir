// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = enable-runtime-library=false
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

#SparseVector = #sparse_tensor.encoding<{
  lvlTypes = ["compressed"]
}>

#SparseMatrix = #sparse_tensor.encoding<{
  lvlTypes = ["compressed", "compressed"]
}>

#Sparse3dTensor = #sparse_tensor.encoding<{
  lvlTypes = ["compressed", "compressed", "compressed"]
}>

module {

  func.func @reshape0(%arg0: tensor<3x4xf64, #SparseMatrix>) -> tensor<2x6xf64, #SparseMatrix> {
    %shape = arith.constant dense <[ 2, 6 ]> : tensor<2xi32>
    %0 = tensor.reshape %arg0(%shape) : (tensor<3x4xf64, #SparseMatrix>, tensor<2xi32>) -> tensor<2x6xf64, #SparseMatrix>
    return %0 : tensor<2x6xf64, #SparseMatrix>
  }

  func.func @reshape1(%arg0: tensor<3x4xf64, #SparseMatrix>) -> tensor<12xf64, #SparseVector> {
    %shape = arith.constant dense <[ 12 ]> : tensor<1xi32>
    %0 = tensor.reshape %arg0(%shape) : (tensor<3x4xf64, #SparseMatrix>, tensor<1xi32>) -> tensor<12xf64, #SparseVector>
    return %0 : tensor<12xf64, #SparseVector>
  }

  func.func @reshape2(%arg0: tensor<3x4xf64, #SparseMatrix>) -> tensor<2x3x2xf64, #Sparse3dTensor> {
    %shape = arith.constant dense <[ 2, 3, 2 ]> : tensor<3xi32>
    %0 = tensor.reshape %arg0(%shape) : (tensor<3x4xf64, #SparseMatrix>, tensor<3xi32>) -> tensor<2x3x2xf64, #Sparse3dTensor>
    return %0 : tensor<2x3x2xf64, #Sparse3dTensor>
  }


  func.func @entry() {
    %m = arith.constant dense <[ [ 1.1,  0.0,  1.3,  0.0 ],
                                 [ 2.1,  0.0,  2.3,  0.0 ],
                                 [ 3.1,  0.0,  3.3,  0.0 ]]> : tensor<3x4xf64>
    %sm = sparse_tensor.convert %m : tensor<3x4xf64> to tensor<3x4xf64, #SparseMatrix>

    %reshaped0 = call @reshape0(%sm) : (tensor<3x4xf64, #SparseMatrix>) -> tensor<2x6xf64, #SparseMatrix>
    %reshaped1 = call @reshape1(%sm) : (tensor<3x4xf64, #SparseMatrix>) -> tensor<12xf64, #SparseVector>
    %reshaped2 = call @reshape2(%sm) : (tensor<3x4xf64, #SparseMatrix>) -> tensor<2x3x2xf64, #Sparse3dTensor>

    %c0 = arith.constant 0 : index
    %df = arith.constant -1.0 : f64

    // CHECK: ( 1.1, 1.3, 2.1, 2.3, 3.1, 3.3
    %b0 = sparse_tensor.values %reshaped0: tensor<2x6xf64, #SparseMatrix> to memref<?xf64>
    %v0 = vector.transfer_read %b0[%c0], %df: memref<?xf64>, vector<12xf64>
    vector.print %v0 : vector<12xf64>

    // CHECK: ( 1.1, 1.3, 2.1, 2.3, 3.1, 3.3
    %b1 = sparse_tensor.values %reshaped1: tensor<12xf64, #SparseVector> to memref<?xf64>
    %v1 = vector.transfer_read %b1[%c0], %df: memref<?xf64>, vector<12xf64>
    vector.print %v1 : vector<12xf64>

    // CHECK: ( 1.1, 1.3, 2.1, 2.3, 3.1, 3.3
    %b2 = sparse_tensor.values %reshaped2: tensor<2x3x2xf64, #Sparse3dTensor> to memref<?xf64>
    %v2 = vector.transfer_read %b2[%c0], %df: memref<?xf64>, vector<12xf64>
    vector.print %v2: vector<12xf64>

    bufferization.dealloc_tensor %sm : tensor<3x4xf64, #SparseMatrix>
    bufferization.dealloc_tensor %reshaped0 : tensor<2x6xf64, #SparseMatrix>
    bufferization.dealloc_tensor %reshaped1 : tensor<12xf64, #SparseVector>
    bufferization.dealloc_tensor %reshaped2 : tensor<2x3x2xf64, #Sparse3dTensor>

    return
  }

}
