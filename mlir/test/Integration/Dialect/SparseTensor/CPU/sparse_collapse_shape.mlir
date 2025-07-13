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
// DEFINE: %{run_libs_sve} = -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils
// DEFINE: %{run_opts} = -e main -entry-point-result=void
// DEFINE: %{run} = mlir-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs_sve}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true
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
// operations: collapse.
//
module {

  func.func @collapse_dense(%arg0: tensor<3x4xf64>) -> tensor<12xf64> {
    %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<3x4xf64> into tensor<12xf64>
    return %0 : tensor<12xf64>
  }

  func.func @collapse_from_sparse(%arg0: tensor<3x4xf64, #SparseMatrix>) -> tensor<12xf64> {
    %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<3x4xf64, #SparseMatrix> into tensor<12xf64>
    return %0 : tensor<12xf64>
  }

  func.func @collapse_to_sparse(%arg0: tensor<3x4xf64>) -> tensor<12xf64, #SparseVector> {
    %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<3x4xf64> into tensor<12xf64, #SparseVector>
    return %0 : tensor<12xf64, #SparseVector>
  }

  func.func @collapse_sparse2sparse(%arg0: tensor<3x4xf64, #SparseMatrix>) -> tensor<12xf64, #SparseVector> {
    %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<3x4xf64, #SparseMatrix> into tensor<12xf64, #SparseVector>
    return %0 : tensor<12xf64, #SparseVector>
  }

  func.func @collapse_dense_6x10(%arg0: tensor<2x3x5x2xf64>) -> tensor<6x10xf64> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<2x3x5x2xf64> into tensor<6x10xf64>
    return %0 : tensor<6x10xf64>
  }

  func.func @collapse_from_sparse_6x10(%arg0: tensor<2x3x5x2xf64, #Sparse4dTensor>) -> tensor<6x10xf64> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<2x3x5x2xf64, #Sparse4dTensor> into tensor<6x10xf64>
    return %0 : tensor<6x10xf64>
  }

  func.func @collapse_to_sparse_6x10(%arg0: tensor<2x3x5x2xf64>) -> tensor<6x10xf64, #SparseMatrix> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<2x3x5x2xf64> into tensor<6x10xf64, #SparseMatrix>
    return %0 : tensor<6x10xf64, #SparseMatrix>
  }

  func.func @collapse_sparse2sparse_6x10(%arg0: tensor<2x3x5x2xf64, #Sparse4dTensor>) -> tensor<6x10xf64, #SparseMatrix> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<2x3x5x2xf64, #Sparse4dTensor> into tensor<6x10xf64, #SparseMatrix>
    return %0 : tensor<6x10xf64, #SparseMatrix>
  }

  func.func @collapse_dense_dyn(%arg0: tensor<?x?x?x?xf64>) -> tensor<?x?xf64> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<?x?x?x?xf64> into tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  func.func @collapse_from_sparse_dyn(%arg0: tensor<?x?x?x?xf64, #Sparse4dTensor>) -> tensor<?x?xf64> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<?x?x?x?xf64, #Sparse4dTensor> into tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  func.func @collapse_to_sparse_dyn(%arg0: tensor<?x?x?x?xf64>) -> tensor<?x?xf64, #SparseMatrix> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<?x?x?x?xf64> into tensor<?x?xf64, #SparseMatrix>
    return %0 : tensor<?x?xf64, #SparseMatrix>
  }

  func.func @collapse_sparse2sparse_dyn(%arg0: tensor<?x?x?x?xf64, #Sparse4dTensor>) -> tensor<?x?xf64, #SparseMatrix> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<?x?x?x?xf64, #Sparse4dTensor> into tensor<?x?xf64, #SparseMatrix>
    return %0 : tensor<?x?xf64, #SparseMatrix>
  }

  //
  // Main driver.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %df = arith.constant -1.0 : f64

    // Setup test vectors and matrices..
    %m = arith.constant dense <[ [ 1.1,  0.0,  1.3,  0.0 ],
                                 [ 2.1,  0.0,  2.3,  0.0 ],
                                 [ 3.1,  0.0,  3.3,  0.0 ]]> : tensor<3x4xf64>
    %n = arith.constant dense <[
      [ [[ 1.0, 0.0], [ 3.0, 0.0], [ 5.0, 0.0], [ 7.0, 0.0], [ 9.0, 0.0]],
        [[ 0.0, 0.0], [ 0.0, 0.0], [ 0.0, 0.0], [ 0.0, 0.0], [ 0.0, 0.0]],
        [[21.0, 0.0], [23.0, 0.0], [25.0, 0.0], [27.0, 0.0], [29.0, 0.0]] ],
      [ [[ 0.0, 0.0], [ 0.0, 0.0], [ 0.0, 0.0], [ 0.0, 0.0], [ 0.0, 0.0]],
        [[41.0, 0.0], [43.0, 0.0], [45.0, 0.0], [47.0, 0.0], [49.0, 0.0]],
        [[ 0.0, 0.0], [ 0.0, 0.0], [ 0.0, 0.0], [ 0.0, 0.0], [ 0.0, 0.0]] ] ]> : tensor<2x3x5x2xf64>
    %sm = sparse_tensor.convert %m : tensor<3x4xf64> to tensor<3x4xf64, #SparseMatrix>
    %sn = sparse_tensor.convert %n : tensor<2x3x5x2xf64> to tensor<2x3x5x2xf64, #Sparse4dTensor>

    %dm = tensor.cast %m : tensor<3x4xf64> to tensor<?x?xf64>

    %dn = tensor.cast %n : tensor<2x3x5x2xf64> to tensor<?x?x?x?xf64>
    %sdn = sparse_tensor.convert %dn : tensor<?x?x?x?xf64> to tensor<?x?x?x?xf64, #Sparse4dTensor>

    // Call the kernels.
    %collapse0 = call @collapse_dense(%m) : (tensor<3x4xf64>) -> tensor<12xf64>
    %collapse1 = call @collapse_from_sparse(%sm) : (tensor<3x4xf64, #SparseMatrix>) -> tensor<12xf64>
    %collapse2 = call @collapse_to_sparse(%m) : (tensor<3x4xf64>) -> tensor<12xf64, #SparseVector>
    %collapse3 = call @collapse_sparse2sparse(%sm) : (tensor<3x4xf64, #SparseMatrix>) -> tensor<12xf64, #SparseVector>
    %collapse4 = call @collapse_dense_6x10(%n) : (tensor<2x3x5x2xf64>) -> tensor<6x10xf64>
    %collapse5 = call @collapse_from_sparse_6x10(%sn) : (tensor<2x3x5x2xf64, #Sparse4dTensor>) -> tensor<6x10xf64>
    %collapse6 = call @collapse_to_sparse_6x10(%n) : (tensor<2x3x5x2xf64>) -> tensor<6x10xf64, #SparseMatrix>
    %collapse7 = call @collapse_sparse2sparse_6x10(%sn) : (tensor<2x3x5x2xf64, #Sparse4dTensor>) -> tensor<6x10xf64, #SparseMatrix>
    %collapse8 = call @collapse_dense_dyn(%dn) : (tensor<?x?x?x?xf64>) -> tensor<?x?xf64>
    %collapse9 = call @collapse_from_sparse_dyn(%sdn) : (tensor<?x?x?x?xf64, #Sparse4dTensor>) -> tensor<?x?xf64>
    %collapse10 = call @collapse_to_sparse_dyn(%dn) : (tensor<?x?x?x?xf64>) -> tensor<?x?xf64, #SparseMatrix>
    %collapse11 = call @collapse_sparse2sparse_dyn(%sdn) : (tensor<?x?x?x?xf64, #Sparse4dTensor>) -> tensor<?x?xf64, #SparseMatrix>

    //
    // Verify results of collapse
    //
    // CHECK:      ( 1.1, 0, 1.3, 0, 2.1, 0, 2.3, 0, 3.1, 0, 3.3, 0 )
    // CHECK-NEXT: ( 1.1, 0, 1.3, 0, 2.1, 0, 2.3, 0, 3.1, 0, 3.3, 0 )
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 6
    // CHECK-NEXT: dim = ( 12 )
    // CHECK-NEXT: lvl = ( 12 )
    // CHECK-NEXT: pos[0] : ( 0, 6 )
    // CHECK-NEXT: crd[0] : ( 0, 2, 4, 6, 8, 10 )
    // CHECK-NEXT: values : ( 1.1, 1.3, 2.1, 2.3, 3.1, 3.3 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 6
    // CHECK-NEXT: dim = ( 12 )
    // CHECK-NEXT: lvl = ( 12 )
    // CHECK-NEXT: pos[0] : ( 0, 6 )
    // CHECK-NEXT: crd[0] : ( 0, 2, 4, 6, 8, 10 )
    // CHECK-NEXT: values : ( 1.1, 1.3, 2.1, 2.3, 3.1, 3.3 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ( ( 1, 0, 3, 0, 5, 0, 7, 0, 9, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ), ( 21, 0, 23, 0, 25, 0, 27, 0, 29, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ), ( 41, 0, 43, 0, 45, 0, 47, 0, 49, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ) )
    // CHECK-NEXT: ( ( 1, 0, 3, 0, 5, 0, 7, 0, 9, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ), ( 21, 0, 23, 0, 25, 0, 27, 0, 29, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ), ( 41, 0, 43, 0, 45, 0, 47, 0, 49, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ) )
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 15
    // CHECK-NEXT: dim = ( 6, 10 )
    // CHECK-NEXT: lvl = ( 6, 10 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 0, 2, 4 )
    // CHECK-NEXT: pos[1] : ( 0, 5, 10, 15 )
    // CHECK-NEXT: crd[1] : ( 0, 2, 4, 6, 8, 0, 2, 4, 6, 8, 0, 2, 4, 6, 8 )
    // CHECK-NEXT: values : ( 1, 3, 5, 7, 9, 21, 23, 25, 27, 29, 41, 43, 45, 47, 49 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 15
    // CHECK-NEXT: dim = ( 6, 10 )
    // CHECK-NEXT: lvl = ( 6, 10 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 0, 2, 4 )
    // CHECK-NEXT: pos[1] : ( 0, 5, 10, 15 )
    // CHECK-NEXT: crd[1] : ( 0, 2, 4, 6, 8, 0, 2, 4, 6, 8, 0, 2, 4, 6, 8 )
    // CHECK-NEXT: values : ( 1, 3, 5, 7, 9, 21, 23, 25, 27, 29, 41, 43, 45, 47, 49 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ( ( 1, 0, 3, 0, 5, 0, 7, 0, 9, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ), ( 21, 0, 23, 0, 25, 0, 27, 0, 29, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ), ( 41, 0, 43, 0, 45, 0, 47, 0, 49, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ) )
    // CHECK-NEXT: ( ( 1, 0, 3, 0, 5, 0, 7, 0, 9, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ), ( 21, 0, 23, 0, 25, 0, 27, 0, 29, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ), ( 41, 0, 43, 0, 45, 0, 47, 0, 49, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ) )
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 15
    // CHECK-NEXT: dim = ( 6, 10 )
    // CHECK-NEXT: lvl = ( 6, 10 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 0, 2, 4 )
    // CHECK-NEXT: pos[1] : ( 0, 5, 10, 15 )
    // CHECK-NEXT: crd[1] : ( 0, 2, 4, 6, 8, 0, 2, 4, 6, 8, 0, 2, 4, 6, 8 )
    // CHECK-NEXT: values : ( 1, 3, 5, 7, 9, 21, 23, 25, 27, 29, 41, 43, 45, 47, 49 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 15
    // CHECK-NEXT: dim = ( 6, 10 )
    // CHECK-NEXT: lvl = ( 6, 10 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 0, 2, 4 )
    // CHECK-NEXT: pos[1] : ( 0, 5, 10, 15 )
    // CHECK-NEXT: crd[1] : ( 0, 2, 4, 6, 8, 0, 2, 4, 6, 8, 0, 2, 4, 6, 8 )
    // CHECK-NEXT: values : ( 1, 3, 5, 7, 9, 21, 23, 25, 27, 29, 41, 43, 45, 47, 49 )
    // CHECK-NEXT: ----
    //
    %v0 = vector.transfer_read %collapse0[%c0], %df: tensor<12xf64>, vector<12xf64>
    vector.print %v0 : vector<12xf64>
    %v1 = vector.transfer_read %collapse1[%c0], %df: tensor<12xf64>, vector<12xf64>
    vector.print %v1 : vector<12xf64>
    sparse_tensor.print %collapse2 : tensor<12xf64, #SparseVector>
    sparse_tensor.print %collapse3 : tensor<12xf64, #SparseVector>

    %v4 = vector.transfer_read %collapse4[%c0, %c0], %df: tensor<6x10xf64>, vector<6x10xf64>
    vector.print %v4 : vector<6x10xf64>
    %v5 = vector.transfer_read %collapse5[%c0, %c0], %df: tensor<6x10xf64>, vector<6x10xf64>
    vector.print %v5 : vector<6x10xf64>
    sparse_tensor.print %collapse6 : tensor<6x10xf64, #SparseMatrix>
    sparse_tensor.print %collapse7 : tensor<6x10xf64, #SparseMatrix>

    %v8 = vector.transfer_read %collapse8[%c0, %c0], %df: tensor<?x?xf64>, vector<6x10xf64>
    vector.print %v8 : vector<6x10xf64>
    %v9 = vector.transfer_read %collapse9[%c0, %c0], %df: tensor<?x?xf64>, vector<6x10xf64>
    vector.print %v9 : vector<6x10xf64>
    sparse_tensor.print %collapse10 : tensor<?x?xf64, #SparseMatrix>
    sparse_tensor.print %collapse11 : tensor<?x?xf64, #SparseMatrix>

    // Release sparse resources.
    bufferization.dealloc_tensor %sm : tensor<3x4xf64, #SparseMatrix>
    bufferization.dealloc_tensor %sn : tensor<2x3x5x2xf64, #Sparse4dTensor>
    bufferization.dealloc_tensor %sdn : tensor<?x?x?x?xf64, #Sparse4dTensor>
    bufferization.dealloc_tensor %collapse2 : tensor<12xf64, #SparseVector>
    bufferization.dealloc_tensor %collapse3 : tensor<12xf64, #SparseVector>
    bufferization.dealloc_tensor %collapse6 : tensor<6x10xf64, #SparseMatrix>
    bufferization.dealloc_tensor %collapse7 : tensor<6x10xf64, #SparseMatrix>
    bufferization.dealloc_tensor %collapse10 : tensor<?x?xf64, #SparseMatrix>
    bufferization.dealloc_tensor %collapse11 : tensor<?x?xf64, #SparseMatrix>

    // Release dense resources.
    bufferization.dealloc_tensor %collapse1 : tensor<12xf64>
    bufferization.dealloc_tensor %collapse5 : tensor<6x10xf64>
    bufferization.dealloc_tensor %collapse9: tensor<?x?xf64>

    return
  }
}
