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

#AllDense = #sparse_tensor.encoding<{
  map = (i, j) -> (
    i : dense,
    j : dense
  )
}>

#AllDenseT = #sparse_tensor.encoding<{
  map = (i, j) -> (
    j : dense,
    i : dense
  )
}>

#CSR = #sparse_tensor.encoding<{
  map = (i, j) -> (
    i : dense,
    j : compressed
  )
}>

#DCSR = #sparse_tensor.encoding<{
  map = (i, j) -> (
    i : compressed,
    j : compressed
  )
}>

#CSC = #sparse_tensor.encoding<{
  map = (i, j) -> (
    j : dense,
    i : compressed
  )
}>

#DCSC = #sparse_tensor.encoding<{
  map = (i, j) -> (
    j : compressed,
    i : compressed
  )
}>

#BSR = #sparse_tensor.encoding<{
  map = (i, j) -> (
    i floordiv 2 : compressed,
    j floordiv 4 : compressed,
    i mod 2 : dense,
    j mod 4 : dense
  )
}>

#BSRC = #sparse_tensor.encoding<{
  map = (i, j) -> (
    i floordiv 2 : compressed,
    j floordiv 4 : compressed,
    j mod 4 : dense,
    i mod 2 : dense
  )
}>

#BSC = #sparse_tensor.encoding<{
  map = (i, j) -> (
    j floordiv 4 : compressed,
    i floordiv 2 : compressed,
    i mod 2 : dense,
    j mod 4 : dense
  )
}>

#BSCC = #sparse_tensor.encoding<{
  map = (i, j) -> (
    j floordiv 4 : compressed,
    i floordiv 2 : compressed,
    j mod 4 : dense,
    i mod 2 : dense
  )
}>

#BSR0 = #sparse_tensor.encoding<{
  map = (i, j) -> (
    i floordiv 2 : dense,
    j floordiv 4 : compressed,
    i mod 2 : dense,
    j mod 4 : dense
  )
}>

#BSC0 = #sparse_tensor.encoding<{
  map = (i, j) -> (
    j floordiv 4 : dense,
    i floordiv 2 : compressed,
    i mod 2 : dense,
    j mod 4 : dense
  )
}>

module {

  //
  // Main driver that tests sparse tensor storage.
  //
  func.func @main() {
    %x = arith.constant dense <[
         [ 1, 0, 2, 0, 0, 0, 0, 0 ],
         [ 0, 0, 0, 0, 0, 0, 0, 0 ],
         [ 0, 0, 0, 0, 0, 0, 0, 0 ],
         [ 0, 0, 3, 4, 0, 5, 0, 0 ] ]> : tensor<4x8xi32>

    %XO = sparse_tensor.convert %x : tensor<4x8xi32> to tensor<4x8xi32, #AllDense>
    %XT = sparse_tensor.convert %x : tensor<4x8xi32> to tensor<4x8xi32, #AllDenseT>

    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 32
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 4, 8 )
    // CHECK-NEXT: values : ( 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 5, 0, 0,
    // CHECK-NEXT: ----
    sparse_tensor.print %XO : tensor<4x8xi32, #AllDense>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 32
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 8, 4 )
    // CHECK-NEXT: values : ( 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0,
    // CHECK-NEXT: ----
    sparse_tensor.print %XT : tensor<4x8xi32, #AllDenseT>

    %a = sparse_tensor.convert %x : tensor<4x8xi32> to tensor<4x8xi32, #CSR>
    %b = sparse_tensor.convert %x : tensor<4x8xi32> to tensor<4x8xi32, #DCSR>
    %c = sparse_tensor.convert %x : tensor<4x8xi32> to tensor<4x8xi32, #CSC>
    %d = sparse_tensor.convert %x : tensor<4x8xi32> to tensor<4x8xi32, #DCSC>
    %e = sparse_tensor.convert %x : tensor<4x8xi32> to tensor<4x8xi32, #BSR>
    %f = sparse_tensor.convert %x : tensor<4x8xi32> to tensor<4x8xi32, #BSRC>
    %g = sparse_tensor.convert %x : tensor<4x8xi32> to tensor<4x8xi32, #BSC>
    %h = sparse_tensor.convert %x : tensor<4x8xi32> to tensor<4x8xi32, #BSCC>
    %i = sparse_tensor.convert %x : tensor<4x8xi32> to tensor<4x8xi32, #BSR0>
    %j = sparse_tensor.convert %x : tensor<4x8xi32> to tensor<4x8xi32, #BSC0>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 5
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 4, 8 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 2, 2, 5,
    // CHECK-NEXT: crd[1] : ( 0, 2, 2, 3, 5,
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5,
    // CHECK-NEXT: ----
    sparse_tensor.print %a : tensor<4x8xi32, #CSR>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 5
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 4, 8 )
    // CHECK-NEXT: pos[0] : ( 0, 2,
    // CHECK-NEXT: crd[0] : ( 0, 3,
    // CHECK-NEXT: pos[1] : ( 0, 2, 5,
    // CHECK-NEXT: crd[1] : ( 0, 2, 2, 3, 5,
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5,
    // CHECK-NEXT: ----
    sparse_tensor.print %b : tensor<4x8xi32, #DCSR>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 5
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 8, 4 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 1, 3, 4, 4, 5, 5, 5,
    // CHECK-NEXT: crd[1] : ( 0, 0, 3, 3, 3,
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5,
    // CHECK-NEXT: ----
    sparse_tensor.print %c : tensor<4x8xi32, #CSC>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 5
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 8, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 4,
    // CHECK-NEXT: crd[0] : ( 0, 2, 3, 5,
    // CHECK-NEXT: pos[1] : ( 0, 1, 3, 4, 5,
    // CHECK-NEXT: crd[1] : ( 0, 0, 3, 3, 3,
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5,
    // CHECK-NEXT: ----
    sparse_tensor.print %d : tensor<4x8xi32, #DCSC>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 2, 2, 2, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 2,
    // CHECK-NEXT: crd[0] : ( 0, 1,
    // CHECK-NEXT: pos[1] : ( 0, 1, 3,
    // CHECK-NEXT: crd[1] : ( 0, 0, 1,
    // CHECK-NEXT: values : ( 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 5, 0, 0,
    // CHECK-NEXT: ----
    sparse_tensor.print %e : tensor<4x8xi32, #BSR>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 2, 2, 4, 2 )
    // CHECK-NEXT: pos[0] : ( 0, 2,
    // CHECK-NEXT: crd[0] : ( 0, 1,
    // CHECK-NEXT: pos[1] : ( 0, 1, 3,
    // CHECK-NEXT: crd[1] : ( 0, 0, 1,
    // CHECK-NEXT: values : ( 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 5, 0, 0, 0, 0,
    // CHECK-NEXT: ----
    sparse_tensor.print %f : tensor<4x8xi32, #BSRC>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 2, 2, 2, 4 )
    // CHECK-NEXT: pos[0] : ( 0, 2,
    // CHECK-NEXT: crd[0] : ( 0, 1,
    // CHECK-NEXT: pos[1] : ( 0, 2, 3,
    // CHECK-NEXT: crd[1] : ( 0, 1, 1,
    // CHECK-NEXT: values : ( 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 5, 0, 0,
    // CHECK-NEXT: ----
    sparse_tensor.print %g : tensor<4x8xi32, #BSC>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 2, 2, 4, 2 )
    // CHECK-NEXT: pos[0] : ( 0, 2,
    // CHECK-NEXT: crd[0] : ( 0, 1,
    // CHECK-NEXT: pos[1] : ( 0, 2, 3,
    // CHECK-NEXT: crd[1] : ( 0, 1, 1,
    // CHECK-NEXT: values : ( 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 5, 0, 0, 0, 0,
    // CHECK-NEXT: ----
    sparse_tensor.print %h : tensor<4x8xi32, #BSCC>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 2, 2, 2, 4 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 3,
    // CHECK-NEXT: crd[1] : ( 0, 0, 1,
    // CHECK-NEXT: values : ( 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 5, 0, 0,
    // CHECK-NEXT: ----
    sparse_tensor.print %i : tensor<4x8xi32, #BSR0>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 4, 8 )
    // CHECK-NEXT: lvl = ( 2, 2, 2, 4 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3,
    // CHECK-NEXT: crd[1] : ( 0, 1, 1,
    // CHECK-NEXT: values : ( 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 5, 0, 0,
    // CHECK-NEXT: ----
    sparse_tensor.print %j : tensor<4x8xi32, #BSC0>

    // Release the resources.
    bufferization.dealloc_tensor %XO : tensor<4x8xi32, #AllDense>
    bufferization.dealloc_tensor %XT : tensor<4x8xi32, #AllDenseT>
    bufferization.dealloc_tensor %a : tensor<4x8xi32, #CSR>
    bufferization.dealloc_tensor %b : tensor<4x8xi32, #DCSR>
    bufferization.dealloc_tensor %c : tensor<4x8xi32, #CSC>
    bufferization.dealloc_tensor %d : tensor<4x8xi32, #DCSC>
    bufferization.dealloc_tensor %e : tensor<4x8xi32, #BSR>
    bufferization.dealloc_tensor %f : tensor<4x8xi32, #BSRC>
    bufferization.dealloc_tensor %g : tensor<4x8xi32, #BSC>
    bufferization.dealloc_tensor %h : tensor<4x8xi32, #BSCC>
    bufferization.dealloc_tensor %i : tensor<4x8xi32, #BSR0>
    bufferization.dealloc_tensor %j : tensor<4x8xi32, #BSC0>

    return
  }
}
