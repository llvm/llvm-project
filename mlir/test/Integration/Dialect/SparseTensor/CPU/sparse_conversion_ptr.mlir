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
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#DCSR  = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed),
  posWidth = 8,
  crdWidth = 8
}>

#DCSC  = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : compressed, d0 : compressed),
  posWidth = 64,
  crdWidth = 64
}>

#CSC  = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed),
  posWidth = 16,
  crdWidth = 32
}>

//
// Integration test that tests conversions between sparse tensors,
// where the position and index sizes in the overhead storage change
// in addition to layout.
//
module {
  func.func @main() {
    %c1 = arith.constant 1 : index
    %t1 = arith.constant sparse<
      [ [0,0], [0,1], [0,63], [1,0], [1,1], [31,0], [31,63] ],
       [ 1.0,   2.0,   3.0,    4.0,   5.0,   6.0,    7.0 ]> : tensor<32x64xf64>
    %t2 = tensor.cast %t1 : tensor<32x64xf64> to tensor<?x?xf64>

    // Dense to sparse.
    %1 = sparse_tensor.convert %t1 : tensor<32x64xf64> to tensor<32x64xf64, #DCSR>
    %2 = sparse_tensor.convert %t1 : tensor<32x64xf64> to tensor<32x64xf64, #DCSC>
    %3 = sparse_tensor.convert %t1 : tensor<32x64xf64> to tensor<32x64xf64, #CSC>

    // Sparse to sparse.
    %4 = sparse_tensor.convert %1 : tensor<32x64xf64, #DCSR> to tensor<32x64xf64, #DCSC>
    %5 = sparse_tensor.convert %2 : tensor<32x64xf64, #DCSC> to tensor<32x64xf64, #DCSR>
    %6 = sparse_tensor.convert %3 : tensor<32x64xf64, #CSC>  to tensor<32x64xf64, #DCSR>

    //
    // Verify the outputs.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 7
    // CHECK-NEXT: dim = ( 32, 64 )
    // CHECK-NEXT: lvl = ( 32, 64 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 31
    // CHECK-NEXT: pos[1] : ( 0, 3, 5, 7
    // CHECK-NEXT: crd[1] : ( 0, 1, 63, 0, 1, 0, 63
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 7
    // CHECK-NEXT: dim = ( 32, 64 )
    // CHECK-NEXT: lvl = ( 64, 32 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 63
    // CHECK-NEXT: pos[1] : ( 0, 3, 5, 7
    // CHECK-NEXT: crd[1] : ( 0, 1, 31, 0, 1, 0, 31
    // CHECK-NEXT: values : ( 1, 4, 6, 2, 5, 3, 7
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 7
    // CHECK-NEXT: dim = ( 32, 64 )
    // CHECK-NEXT: lvl = ( 64, 32 )
    // CHECK-NEXT: pos[1] : ( 0, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7
    // CHECK-NEXT: crd[1] : ( 0, 1, 31, 0, 1, 0, 31
    // CHECK-NEXT: values : ( 1, 4, 6, 2, 5, 3, 7
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 7
    // CHECK-NEXT: dim = ( 32, 64 )
    // CHECK-NEXT: lvl = ( 64, 32 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 63
    // CHECK-NEXT: pos[1] : ( 0, 3, 5, 7
    // CHECK-NEXT: crd[1] : ( 0, 1, 31, 0, 1, 0, 31
    // CHECK-NEXT: values : ( 1, 4, 6, 2, 5, 3, 7
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 7
    // CHECK-NEXT: dim = ( 32, 64 )
    // CHECK-NEXT: lvl = ( 32, 64 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 31
    // CHECK-NEXT: pos[1] : ( 0, 3, 5, 7
    // CHECK-NEXT: crd[1] : ( 0, 1, 63, 0, 1, 0, 63
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 7
    // CHECK-NEXT: dim = ( 32, 64 )
    // CHECK-NEXT: lvl = ( 32, 64 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 31
    // CHECK-NEXT: pos[1] : ( 0, 3, 5, 7
    // CHECK-NEXT: crd[1] : ( 0, 1, 63, 0, 1, 0, 63
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %1 : tensor<32x64xf64, #DCSR>
    sparse_tensor.print %2 : tensor<32x64xf64, #DCSC>
    sparse_tensor.print %3 : tensor<32x64xf64, #CSC>
    sparse_tensor.print %4 : tensor<32x64xf64, #DCSC>
    sparse_tensor.print %5 : tensor<32x64xf64, #DCSR>
    sparse_tensor.print %6 : tensor<32x64xf64, #DCSR>

    // Release the resources.
    bufferization.dealloc_tensor %1 : tensor<32x64xf64, #DCSR>
    bufferization.dealloc_tensor %2 : tensor<32x64xf64, #DCSC>
    bufferization.dealloc_tensor %3 : tensor<32x64xf64, #CSC>
    bufferization.dealloc_tensor %4 : tensor<32x64xf64, #DCSC>
    bufferization.dealloc_tensor %5 : tensor<32x64xf64, #DCSR>
    bufferization.dealloc_tensor %6 : tensor<32x64xf64, #DCSR>

    return
  }
}
