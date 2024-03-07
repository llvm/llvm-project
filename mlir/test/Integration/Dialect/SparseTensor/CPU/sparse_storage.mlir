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
// Do the same run, but now with vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

//
// Several common sparse storage schemes.
//

#Dense  = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : dense)
}>

#CSR  = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#DCSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

#DCSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : compressed, d0 : compressed)
}>

#BlockRow = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : dense)
}>

#BlockCol = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : compressed, d0 : dense)
}>

//
// Integration test that looks "under the hood" of sparse storage schemes.
//
module {
  //
  // Main driver that initializes a sparse tensor and inspects the sparse
  // storage schemes in detail. Note that users of the MLIR sparsifier
  // are typically not concerned with such details, but the test ensures
  // everything is working "under the hood".
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = arith.constant 0.0 : f64

    //
    // Initialize a dense tensor.
    //
    %t = arith.constant dense<[
       [ 1.0,  0.0,  2.0,  0.0,  0.0,  0.0,  0.0,  3.0],
       [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
       [ 0.0,  0.0,  4.0,  0.0,  0.0,  0.0,  0.0,  0.0],
       [ 0.0,  0.0,  0.0,  5.0,  0.0,  0.0,  0.0,  0.0],
       [ 0.0,  0.0,  0.0,  0.0,  6.0,  0.0,  0.0,  0.0],
       [ 0.0,  7.0,  8.0,  0.0,  0.0,  0.0,  0.0,  9.0],
       [ 0.0,  0.0, 10.0,  0.0,  0.0,  0.0, 11.0, 12.0],
       [ 0.0, 13.0, 14.0,  0.0,  0.0,  0.0, 15.0, 16.0],
       [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
       [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 17.0,  0.0]
    ]> : tensor<10x8xf64>

    //
    // Convert dense tensor to various sparse tensors.
    //
    %0 = sparse_tensor.convert %t : tensor<10x8xf64> to tensor<10x8xf64, #Dense>
    %1 = sparse_tensor.convert %t : tensor<10x8xf64> to tensor<10x8xf64, #CSR>
    %2 = sparse_tensor.convert %t : tensor<10x8xf64> to tensor<10x8xf64, #DCSR>
    %3 = sparse_tensor.convert %t : tensor<10x8xf64> to tensor<10x8xf64, #CSC>
    %4 = sparse_tensor.convert %t : tensor<10x8xf64> to tensor<10x8xf64, #DCSC>
    %x = sparse_tensor.convert %t : tensor<10x8xf64> to tensor<10x8xf64, #BlockRow>
    %y = sparse_tensor.convert %t : tensor<10x8xf64> to tensor<10x8xf64, #BlockCol>

    //
    // Inspect storage scheme of Dense.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 80
    // CHECK-NEXT: dim = ( 10, 8 )
    // CHECK-NEXT: lvl = ( 10, 8 )
    // CHECK-NEXT: values : ( 1, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, 9, 0, 0, 10, 0, 0, 0, 11, 12, 0, 13, 14, 0, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %0 : tensor<10x8xf64, #Dense>

    //
    // Inspect storage scheme of CSR.
    //
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 17
    // CHECK-NEXT: dim = ( 10, 8 )
    // CHECK-NEXT: lvl = ( 10, 8 )
    // CHECK-NEXT: pos[1] : ( 0, 3, 3, 4, 5, 6, 9, 12, 16, 16, 17
    // CHECK-NEXT: crd[1] : ( 0, 2, 7, 2, 3, 4, 1, 2, 7, 2, 6, 7, 1, 2, 6, 7, 6
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %1 : tensor<10x8xf64, #CSR>

    //
    // Inspect storage scheme of DCSR.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 17
    // CHECK-NEXT: dim = ( 10, 8 )
    // CHECK-NEXT: lvl = ( 10, 8 )
    // CHECK-NEXT: pos[0] : ( 0, 8
    // CHECK-NEXT: crd[0] : ( 0, 2, 3, 4, 5, 6, 7, 9
    // CHECK-NEXT: pos[1] : ( 0, 3, 4, 5, 6, 9, 12, 16, 17
    // CHECK-NEXT: crd[1] : ( 0, 2, 7, 2, 3, 4, 1, 2, 7, 2, 6, 7, 1, 2, 6, 7, 6
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %2 : tensor<10x8xf64, #DCSR>

    //
    // Inspect storage scheme of CSC.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 17
    // CHECK-NEXT: dim = ( 10, 8 )
    // CHECK-NEXT: lvl = ( 8, 10 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 3, 8, 9, 10, 10, 13, 17
    // CHECK-NEXT: crd[1] : ( 0, 5, 7, 0, 2, 5, 6, 7, 3, 4, 6, 7, 9, 0, 5, 6, 7
    // CHECK-NEXT: values : ( 1, 7, 13, 2, 4, 8, 10, 14, 5, 6, 11, 15, 17, 3, 9, 12, 16
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %3 : tensor<10x8xf64, #CSC>

    //
    // Inspect storage scheme of DCSC.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 17
    // CHECK-NEXT: dim = ( 10, 8 )
    // CHECK-NEXT: lvl = ( 8, 10 )
    // CHECK-NEXT: pos[0] : ( 0, 7
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3, 4, 6, 7
    // CHECK-NEXT: pos[1] : ( 0, 1, 3, 8, 9, 10, 13, 17
    // CHECK-NEXT: crd[1] : ( 0, 5, 7, 0, 2, 5, 6, 7, 3, 4, 6, 7, 9, 0, 5, 6, 7
    // CHECK-NEXT: values : ( 1, 7, 13, 2, 4, 8, 10, 14, 5, 6, 11, 15, 17, 3, 9, 12, 16
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %4 : tensor<10x8xf64, #DCSC>

    //
    // Inspect storage scheme of BlockRow.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 64
    // CHECK-NEXT: dim = ( 10, 8 )
    // CHECK-NEXT: lvl = ( 10, 8 )
    // CHECK-NEXT: pos[0] : ( 0, 8
    // CHECK-NEXT: crd[0] : ( 0, 2, 3, 4, 5, 6, 7, 9
    // CHECK-NEXT: values : ( 1, 0, 2, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, 9, 0, 0, 10, 0, 0, 0, 11, 12, 0, 13, 14, 0, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 17, 0
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %x : tensor<10x8xf64, #BlockRow>

    //
    // Inspect storage scheme of BlockCol.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 70
    // CHECK-NEXT: dim = ( 10, 8 )
    // CHECK-NEXT: lvl = ( 8, 10 )
    // CHECK-NEXT: pos[0] : ( 0, 7
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3, 4, 6, 7
    // CHECK-NEXT: values : ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 13, 0, 0, 2, 0, 4, 0, 0, 8, 10, 14, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 15, 0, 17, 3, 0, 0, 0, 0, 9, 12, 16, 0, 0
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %y : tensor<10x8xf64, #BlockCol>

    // Release the resources.
    bufferization.dealloc_tensor %0 : tensor<10x8xf64, #Dense>
    bufferization.dealloc_tensor %1 : tensor<10x8xf64, #CSR>
    bufferization.dealloc_tensor %2 : tensor<10x8xf64, #DCSR>
    bufferization.dealloc_tensor %3 : tensor<10x8xf64, #CSC>
    bufferization.dealloc_tensor %4 : tensor<10x8xf64, #DCSC>
    bufferization.dealloc_tensor %x : tensor<10x8xf64, #BlockRow>
    bufferization.dealloc_tensor %y : tensor<10x8xf64, #BlockCol>

    return
  }
}
