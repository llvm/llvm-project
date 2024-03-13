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

#Tensor1  = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : compressed)
}>

#Tensor2  = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d1 : compressed, d2 : compressed, d0 : compressed)
}>

#Tensor3  = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : compressed, d0 : compressed, d1 : compressed)
}>

//
// Integration test that tests conversions between sparse tensors.
//
module {
  //
  // Main driver.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    //
    // Initialize a 3-dim dense tensor.
    //
    %t = arith.constant dense<[
       [  [  1.0,  2.0,  3.0,  4.0 ],
          [  5.0,  6.0,  7.0,  8.0 ],
          [  9.0, 10.0, 11.0, 12.0 ] ],
       [  [ 13.0, 14.0, 15.0, 16.0 ],
          [ 17.0, 18.0, 19.0, 20.0 ],
          [ 21.0, 22.0, 23.0, 24.0 ] ]
    ]> : tensor<2x3x4xf64>

    //
    // Convert dense tensor directly to various sparse tensors.
    //    tensor1: stored as 2x3x4
    //    tensor2: stored as 3x4x2
    //    tensor3: stored as 4x2x3
    //
    %1 = sparse_tensor.convert %t : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor1>
    %2 = sparse_tensor.convert %t : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor2>
    %3 = sparse_tensor.convert %t : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor3>

    //
    // Convert sparse tensor to various sparse tensors. Note that the result
    // should always correspond to the direct conversion, since the sparse
    // tensor formats have the ability to restore into the original ordering.
    //
    %a = sparse_tensor.convert %1 : tensor<2x3x4xf64, #Tensor1> to tensor<2x3x4xf64, #Tensor1>
    %b = sparse_tensor.convert %2 : tensor<2x3x4xf64, #Tensor2> to tensor<2x3x4xf64, #Tensor1>
    %c = sparse_tensor.convert %3 : tensor<2x3x4xf64, #Tensor3> to tensor<2x3x4xf64, #Tensor1>
    %d = sparse_tensor.convert %1 : tensor<2x3x4xf64, #Tensor1> to tensor<2x3x4xf64, #Tensor2>
    %e = sparse_tensor.convert %2 : tensor<2x3x4xf64, #Tensor2> to tensor<2x3x4xf64, #Tensor2>
    %f = sparse_tensor.convert %3 : tensor<2x3x4xf64, #Tensor3> to tensor<2x3x4xf64, #Tensor2>
    %g = sparse_tensor.convert %1 : tensor<2x3x4xf64, #Tensor1> to tensor<2x3x4xf64, #Tensor3>
    %h = sparse_tensor.convert %2 : tensor<2x3x4xf64, #Tensor2> to tensor<2x3x4xf64, #Tensor3>
    %i = sparse_tensor.convert %3 : tensor<2x3x4xf64, #Tensor3> to tensor<2x3x4xf64, #Tensor3>

    //
    // Verify the outputs.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 2, 3, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 2
    // CHECK-NEXT: crd[0] : ( 0, 1
    // CHECK-NEXT: pos[1] : ( 0, 3, 6
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 0, 1, 2
    // CHECK-NEXT: pos[2] : ( 0, 4, 8, 12, 16, 20, 24
    // CHECK-NEXT: crd[2] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 3, 4, 2 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 2
    // CHECK-NEXT: pos[1] : ( 0, 4, 8, 12
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    // CHECK-NEXT: pos[2] : ( 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24
    // CHECK-NEXT: crd[2] : ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: values : ( 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 4, 2, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 4
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 6, 8
    // CHECK-NEXT: crd[1] : ( 0, 1, 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: pos[2] : ( 0, 3, 6, 9, 12, 15, 18, 21, 24
    // CHECK-NEXT: crd[2] : ( 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2
    // CHECK-NEXT: values : ( 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 2, 3, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 2
    // CHECK-NEXT: crd[0] : ( 0, 1
    // CHECK-NEXT: pos[1] : ( 0, 3, 6
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 0, 1, 2
    // CHECK-NEXT: pos[2] : ( 0, 4, 8, 12, 16, 20, 24
    // CHECK-NEXT: crd[2] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 2, 3, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 2
    // CHECK-NEXT: crd[0] : ( 0, 1
    // CHECK-NEXT: pos[1] : ( 0, 3, 6
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 0, 1, 2
    // CHECK-NEXT: pos[2] : ( 0, 4, 8, 12, 16, 20, 24
    // CHECK-NEXT: crd[2] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 2, 3, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 2
    // CHECK-NEXT: crd[0] : ( 0, 1
    // CHECK-NEXT: pos[1] : ( 0, 3, 6
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 0, 1, 2
    // CHECK-NEXT: pos[2] : ( 0, 4, 8, 12, 16, 20, 24
    // CHECK-NEXT: crd[2] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 3, 4, 2 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 2
    // CHECK-NEXT: pos[1] : ( 0, 4, 8, 12
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    // CHECK-NEXT: pos[2] : ( 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24
    // CHECK-NEXT: crd[2] : ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: values : ( 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 3, 4, 2 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 2
    // CHECK-NEXT: pos[1] : ( 0, 4, 8, 12
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    // CHECK-NEXT: pos[2] : ( 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24
    // CHECK-NEXT: crd[2] : ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: values : ( 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 3, 4, 2 )
    // CHECK-NEXT: pos[0] : ( 0, 3
    // CHECK-NEXT: crd[0] : ( 0, 1, 2
    // CHECK-NEXT: pos[1] : ( 0, 4, 8, 12
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    // CHECK-NEXT: pos[2] : ( 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24
    // CHECK-NEXT: crd[2] : ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: values : ( 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 4, 2, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 4
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 6, 8
    // CHECK-NEXT: crd[1] : ( 0, 1, 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: pos[2] : ( 0, 3, 6, 9, 12, 15, 18, 21, 24
    // CHECK-NEXT: crd[2] : ( 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2
    // CHECK-NEXT: values : ( 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 4, 2, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 4
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 6, 8
    // CHECK-NEXT: crd[1] : ( 0, 1, 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: pos[2] : ( 0, 3, 6, 9, 12, 15, 18, 21, 24
    // CHECK-NEXT: crd[2] : ( 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2
    // CHECK-NEXT: values : ( 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 2, 3, 4 )
    // CHECK-NEXT: lvl = ( 4, 2, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 4
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 6, 8
    // CHECK-NEXT: crd[1] : ( 0, 1, 0, 1, 0, 1, 0, 1
    // CHECK-NEXT: pos[2] : ( 0, 3, 6, 9, 12, 15, 18, 21, 24
    // CHECK-NEXT: crd[2] : ( 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2
    // CHECK-NEXT: values : ( 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %1 : tensor<2x3x4xf64, #Tensor1>
    sparse_tensor.print %2 : tensor<2x3x4xf64, #Tensor2>
    sparse_tensor.print %3 : tensor<2x3x4xf64, #Tensor3>
    sparse_tensor.print %a : tensor<2x3x4xf64, #Tensor1>
    sparse_tensor.print %b : tensor<2x3x4xf64, #Tensor1>
    sparse_tensor.print %c : tensor<2x3x4xf64, #Tensor1>
    sparse_tensor.print %d : tensor<2x3x4xf64, #Tensor2>
    sparse_tensor.print %e : tensor<2x3x4xf64, #Tensor2>
    sparse_tensor.print %f : tensor<2x3x4xf64, #Tensor2>
    sparse_tensor.print %g : tensor<2x3x4xf64, #Tensor3>
    sparse_tensor.print %h : tensor<2x3x4xf64, #Tensor3>
    sparse_tensor.print %i : tensor<2x3x4xf64, #Tensor3>

    // Release the resources.
    bufferization.dealloc_tensor %1 : tensor<2x3x4xf64, #Tensor1>
    bufferization.dealloc_tensor %2 : tensor<2x3x4xf64, #Tensor2>
    bufferization.dealloc_tensor %3 : tensor<2x3x4xf64, #Tensor3>
    bufferization.dealloc_tensor %b : tensor<2x3x4xf64, #Tensor1>
    bufferization.dealloc_tensor %c : tensor<2x3x4xf64, #Tensor1>
    bufferization.dealloc_tensor %d : tensor<2x3x4xf64, #Tensor2>
    bufferization.dealloc_tensor %f : tensor<2x3x4xf64, #Tensor2>
    bufferization.dealloc_tensor %g : tensor<2x3x4xf64, #Tensor3>
    bufferization.dealloc_tensor %h : tensor<2x3x4xf64, #Tensor3>

    return
  }
}
