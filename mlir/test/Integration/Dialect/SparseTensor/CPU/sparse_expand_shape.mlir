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
// DEFINE: %{run_opts} = -e main -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

#SparseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#Sparse3dTensor = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : compressed)
}>

#Sparse4dTensor = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (d0 : compressed, d1 : compressed, d2 : compressed, d3 : compressed)
}>

//
// Test with various forms of the two most elementary reshape
// operations: expand
//
module {

  func.func @expand_dense(%arg0: tensor<12xf64>) -> tensor<3x4xf64> {
    %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<12xf64> into tensor<3x4xf64>
    return %0 : tensor<3x4xf64>
  }

  func.func @expand_from_sparse(%arg0: tensor<12xf64, #SparseVector>) -> tensor<3x4xf64> {
    %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<12xf64, #SparseVector> into tensor<3x4xf64>
    return %0 : tensor<3x4xf64>
  }

  func.func @expand_to_sparse(%arg0: tensor<12xf64>) -> tensor<3x4xf64, #SparseMatrix> {
    %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<12xf64> into tensor<3x4xf64, #SparseMatrix>
    return %0 : tensor<3x4xf64, #SparseMatrix>
  }

  func.func @expand_sparse2sparse(%arg0: tensor<12xf64, #SparseVector>) -> tensor<3x4xf64, #SparseMatrix> {
    %0 = tensor.expand_shape %arg0 [[0, 1]] : tensor<12xf64, #SparseVector> into tensor<3x4xf64, #SparseMatrix>
    return %0 : tensor<3x4xf64, #SparseMatrix>
  }

  func.func @expand_dense_3x2x2(%arg0: tensor<3x4xf64>) -> tensor<3x2x2xf64> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<3x4xf64> into tensor<3x2x2xf64>
    return %0 : tensor<3x2x2xf64>
  }

  func.func @expand_from_sparse_3x2x2(%arg0: tensor<3x4xf64, #SparseMatrix>) -> tensor<3x2x2xf64> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<3x4xf64, #SparseMatrix> into tensor<3x2x2xf64>
    return %0 : tensor<3x2x2xf64>
  }

  func.func @expand_to_sparse_3x2x2(%arg0: tensor<3x4xf64>) -> tensor<3x2x2xf64, #Sparse3dTensor> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<3x4xf64> into tensor<3x2x2xf64, #Sparse3dTensor>
    return %0 : tensor<3x2x2xf64, #Sparse3dTensor>
  }

  func.func @expand_sparse2sparse_3x2x2(%arg0: tensor<3x4xf64, #SparseMatrix>) -> tensor<3x2x2xf64, #Sparse3dTensor> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<3x4xf64, #SparseMatrix> into tensor<3x2x2xf64, #Sparse3dTensor>
    return %0 : tensor<3x2x2xf64, #Sparse3dTensor>
  }

  func.func @expand_dense_dyn(%arg0: tensor<?x?xf64>) -> tensor<?x2x?xf64> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<?x?xf64> into tensor<?x2x?xf64>
    return %0 : tensor<?x2x?xf64>
  }

  func.func @expand_from_sparse_dyn(%arg0: tensor<?x?xf64, #SparseMatrix>) -> tensor<?x2x?xf64> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<?x?xf64, #SparseMatrix> into tensor<?x2x?xf64>
    return %0 : tensor<?x2x?xf64>
  }

  func.func @expand_to_sparse_dyn(%arg0: tensor<?x?xf64>) -> tensor<?x2x?xf64, #Sparse3dTensor> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<?x?xf64> into tensor<?x2x?xf64, #Sparse3dTensor>
    return %0 : tensor<?x2x?xf64, #Sparse3dTensor>
  }

  func.func @expand_sparse2sparse_dyn(%arg0: tensor<?x?xf64, #SparseMatrix>) -> tensor<?x2x?xf64, #Sparse3dTensor> {
    %0 = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<?x?xf64, #SparseMatrix> into tensor<?x2x?xf64, #Sparse3dTensor>
    return %0 : tensor<?x2x?xf64, #Sparse3dTensor>
  }

  //
  // Main driver.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %df = arith.constant -1.0 : f64

    // Setup test vectors and matrices..
    %v = arith.constant dense <[ 1.0, 0.0, 3.0, 0.0,  5.0, 0.0,
                                 7.0, 0.0, 9.0, 0.0, 11.0, 0.0]> : tensor<12xf64>
    %m = arith.constant dense <[ [ 1.1,  1.2,  1.3,  1.4 ],
                                 [ 2.1,  2.2,  2.3,  2.4 ],
                                 [ 3.1,  3.2,  3.3,  3.4 ]]> : tensor<3x4xf64>

    %sv = sparse_tensor.convert %v : tensor<12xf64> to tensor<12xf64, #SparseVector>
    %sm = sparse_tensor.convert %m : tensor<3x4xf64> to tensor<3x4xf64, #SparseMatrix>

    %dm = tensor.cast %m : tensor<3x4xf64> to tensor<?x?xf64>
    %sdm = sparse_tensor.convert %dm : tensor<?x?xf64> to tensor<?x?xf64, #SparseMatrix>

    // Call the kernels.
    %expand0 = call @expand_dense(%v) : (tensor<12xf64>) -> tensor<3x4xf64>
    %expand1 = call @expand_from_sparse(%sv) : (tensor<12xf64, #SparseVector>) -> tensor<3x4xf64>
    %expand2 = call @expand_to_sparse(%v) : (tensor<12xf64>) -> tensor<3x4xf64, #SparseMatrix>
    %expand3 = call @expand_sparse2sparse(%sv) : (tensor<12xf64, #SparseVector>) -> tensor<3x4xf64, #SparseMatrix>
    %expand4 = call @expand_dense_3x2x2(%m) : (tensor<3x4xf64>) -> tensor<3x2x2xf64>
    %expand5 = call @expand_from_sparse_3x2x2(%sm) : (tensor<3x4xf64, #SparseMatrix>) -> tensor<3x2x2xf64>
    %expand6 = call @expand_to_sparse_3x2x2(%m) : (tensor<3x4xf64>) -> tensor<3x2x2xf64, #Sparse3dTensor>
    %expand7 = call @expand_sparse2sparse_3x2x2(%sm) : (tensor<3x4xf64, #SparseMatrix>) -> tensor<3x2x2xf64, #Sparse3dTensor>
    %expand8 = call @expand_dense_dyn(%dm) : (tensor<?x?xf64>) -> tensor<?x2x?xf64>
    %expand9 = call @expand_from_sparse_dyn(%sdm) : (tensor<?x?xf64, #SparseMatrix>) -> tensor<?x2x?xf64>
    %expand10 = call @expand_to_sparse_dyn(%dm) : (tensor<?x?xf64>) -> tensor<?x2x?xf64, #Sparse3dTensor>
    %expand11 = call @expand_sparse2sparse_dyn(%sdm) : (tensor<?x?xf64, #SparseMatrix>) -> tensor<?x2x?xf64, #Sparse3dTensor>

    //
    // Verify results of expand with dense output.
    //
    // CHECK:      ( ( 1, 0, 3, 0 ), ( 5, 0, 7, 0 ), ( 9, 0, 11, 0 ) )
    // CHECK-NEXT: ( ( 1, 0, 3, 0 ), ( 5, 0, 7, 0 ), ( 9, 0, 11, 0 ) )
    // CHECK-NEXT: ( ( ( 1.1, 1.2 ), ( 1.3, 1.4 ) ), ( ( 2.1, 2.2 ), ( 2.3, 2.4 ) ), ( ( 3.1, 3.2 ), ( 3.3, 3.4 ) ) )
    // CHECK-NEXT: ( ( ( 1.1, 1.2 ), ( 1.3, 1.4 ) ), ( ( 2.1, 2.2 ), ( 2.3, 2.4 ) ), ( ( 3.1, 3.2 ), ( 3.3, 3.4 ) ) )
    // CHECK-NEXT: ( ( ( 1.1, 1.2 ), ( 1.3, 1.4 ) ), ( ( 2.1, 2.2 ), ( 2.3, 2.4 ) ), ( ( 3.1, 3.2 ), ( 3.3, 3.4 ) ) )
    // CHECK-NEXT: ( ( ( 1.1, 1.2 ), ( 1.3, 1.4 ) ), ( ( 2.1, 2.2 ), ( 2.3, 2.4 ) ), ( ( 3.1, 3.2 ), ( 3.3, 3.4 ) ) )
    //
    %m0 = vector.transfer_read %expand0[%c0, %c0], %df: tensor<3x4xf64>, vector<3x4xf64>
    vector.print %m0 : vector<3x4xf64>
    %m1 = vector.transfer_read %expand1[%c0, %c0], %df: tensor<3x4xf64>, vector<3x4xf64>
    vector.print %m1 : vector<3x4xf64>
    %m4 = vector.transfer_read %expand4[%c0, %c0, %c0], %df: tensor<3x2x2xf64>, vector<3x2x2xf64>
    vector.print %m4 : vector<3x2x2xf64>
    %m5 = vector.transfer_read %expand5[%c0, %c0, %c0], %df: tensor<3x2x2xf64>, vector<3x2x2xf64>
    vector.print %m5 : vector<3x2x2xf64>
    %m8 = vector.transfer_read %expand8[%c0, %c0, %c0], %df: tensor<?x2x?xf64>, vector<3x2x2xf64>
    vector.print %m8 : vector<3x2x2xf64>
    %m9 = vector.transfer_read %expand9[%c0, %c0, %c0], %df: tensor<?x2x?xf64>, vector<3x2x2xf64>
    vector.print %m9 : vector<3x2x2xf64>

    //
    // Verify results of expand with sparse output.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 6
    // CHECK-NEXT: dim = ( 3, 4 )
    // CHECK-NEXT: lvl = ( 3, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 2
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 6
    // CHECK-NEXT: crd[1] : ( 0, 2, 0, 2, 0, 2
    // CHECK-NEXT: values : ( 1, 3, 5, 7, 9, 11
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 6
    // CHECK-NEXT: dim = ( 3, 4 )
    // CHECK-NEXT: lvl = ( 3, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 2
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 6
    // CHECK-NEXT: crd[1] : ( 0, 2, 0, 2, 0, 2
    // CHECK-NEXT: values : ( 1, 3, 5, 7, 9, 11
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 3, 2, 2 )
    // CHECK-NEXT: lvl = ( 3, 2, 2 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 2
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 6
    // CHECK-NEXT: crd[1] : ( 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: pos[2] : ( 0, 2, 4, 6, 8, 10, 12
    // CHECK-NEXT: crd[2] : ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: values : ( 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 3, 2, 2 )
    // CHECK-NEXT: lvl = ( 3, 2, 2 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 2
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 6
    // CHECK-NEXT: crd[1] : ( 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: pos[2] : ( 0, 2, 4, 6, 8, 10, 12
    // CHECK-NEXT: crd[2] : ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: values : ( 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 3, 2, 2 )
    // CHECK-NEXT: lvl = ( 3, 2, 2 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 2
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 6
    // CHECK-NEXT: crd[1] : ( 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: pos[2] : ( 0, 2, 4, 6, 8, 10, 12
    // CHECK-NEXT: crd[2] : ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: values : ( 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 3, 2, 2 )
    // CHECK-NEXT: lvl = ( 3, 2, 2 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 2
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 6
    // CHECK-NEXT: crd[1] : ( 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: pos[2] : ( 0, 2, 4, 6, 8, 10, 12
    // CHECK-NEXT: crd[2] : ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: values : ( 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %expand2 : tensor<3x4xf64, #SparseMatrix>
    sparse_tensor.print %expand3 : tensor<3x4xf64, #SparseMatrix>
    sparse_tensor.print %expand6 : tensor<3x2x2xf64, #Sparse3dTensor>
    sparse_tensor.print %expand7 : tensor<3x2x2xf64, #Sparse3dTensor>
    sparse_tensor.print %expand10 : tensor<?x2x?xf64, #Sparse3dTensor>
    sparse_tensor.print %expand11 : tensor<?x2x?xf64, #Sparse3dTensor>


    // Release sparse resources.
    bufferization.dealloc_tensor %sv : tensor<12xf64, #SparseVector>
    bufferization.dealloc_tensor %sm : tensor<3x4xf64, #SparseMatrix>
    bufferization.dealloc_tensor %sdm : tensor<?x?xf64, #SparseMatrix>
    bufferization.dealloc_tensor %expand2 : tensor<3x4xf64, #SparseMatrix>
    bufferization.dealloc_tensor %expand3 : tensor<3x4xf64, #SparseMatrix>
    bufferization.dealloc_tensor %expand6 : tensor<3x2x2xf64, #Sparse3dTensor>
    bufferization.dealloc_tensor %expand7 : tensor<3x2x2xf64, #Sparse3dTensor>
    bufferization.dealloc_tensor %expand10 : tensor<?x2x?xf64, #Sparse3dTensor>
    bufferization.dealloc_tensor %expand11 : tensor<?x2x?xf64, #Sparse3dTensor>

    // Release dense resources.
    bufferization.dealloc_tensor %expand1 : tensor<3x4xf64>
    bufferization.dealloc_tensor %expand5 : tensor<3x2x2xf64>
    bufferization.dealloc_tensor %expand9 : tensor<?x2x?xf64>

    return
  }
}
