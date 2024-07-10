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

// REDEFINE: %{env} = TENSOR0="%mlir_src_dir/test/Integration/data/ds.mtx"
// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | env %{env} %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | env %{env} %{run} | FileCheck %s

!Filename = !llvm.ptr

#CSR = #sparse_tensor.encoding<{
  map = (i, j) -> ( i : dense, j : compressed)
}>

#CSR_hi = #sparse_tensor.encoding<{
  map = (i, j) -> ( i : dense, j : loose_compressed)
}>

#NV_24 = #sparse_tensor.encoding<{
  map = ( i, j ) -> ( i            : dense,
                      j floordiv 4 : dense,
                      j mod 4      : structured[2, 4]),
  crdWidth = 8
}>

#NV_58 = #sparse_tensor.encoding<{
  map = ( i, j ) -> ( i            : dense,
                      j floordiv 8 : dense,
                      j mod 8      : structured[5, 8]),
  crdWidth = 8
}>

module {

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // Input matrix:
  //
  //  [[0.0,  0.0,  1.0,  2.0,  0.0,  3.0,  0.0,  4.0],
  //   [0.0,  5.0,  6.0,  0.0,  7.0,  0.0,  0.0,  8.0],
  //   [9.0,  0.0, 10.0,  0.0, 11.0, 12.0,  0.0,  0.0]]
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    %A1 = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #CSR>
    %A2 = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #CSR_hi>
    %A3 = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #NV_24>
    %A4 = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #NV_58>

    //
    // CSR:
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 3, 8 )
    // CHECK-NEXT: lvl = ( 3, 8 )
    // CHECK-NEXT: pos[1] : ( 0, 4, 8, 12 )
    // CHECK-NEXT: crd[1] : ( 2, 3, 5, 7, 1, 2, 4, 7, 0, 2, 4, 5 )
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %A1 : tensor<?x?xf64, #CSR>

    //
    // CSR_hi:
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 3, 8 )
    // CHECK-NEXT: lvl = ( 3, 8 )
    // CHECK-NEXT: pos[1] : ( 0, 4, 4, 8, 8, 12, {{.*}} )
    // CHECK-NEXT: crd[1] : ( 2, 3, 5, 7, 1, 2, 4, 7, 0, 2, 4, 5 )
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %A2 : tensor<?x?xf64, #CSR_hi>

    //
    // NV_24:
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 3, 8 )
    // CHECK-NEXT: lvl = ( 3, 2, 4 )
    // CHECK-NEXT: crd[2] : ( 2, 3, 1, 3, 1, 2, 0, 3, 0, 2, 0, 1 )
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 )
    // CHECK-NEXT: ----
    // CHECK-NEXT: ---- Sparse Tensor ----
    //
    sparse_tensor.print %A3 : tensor<?x?xf64, #NV_24>

    //
    // NV_58:
    //
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 3, 8 )
    // CHECK-NEXT: lvl = ( 3, 1, 8 )
    // CHECK-NEXT: crd[2] : ( 2, 3, 5, 7, 1, 2, 4, 7, 0, 2, 4, 5 )
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %A4 : tensor<?x?xf64, #NV_58>

    // Release the resources.
    bufferization.dealloc_tensor %A1: tensor<?x?xf64, #CSR>
    bufferization.dealloc_tensor %A2: tensor<?x?xf64, #CSR_hi>
    bufferization.dealloc_tensor %A3: tensor<?x?xf64, #NV_24>
    bufferization.dealloc_tensor %A4: tensor<?x?xf64, #NV_58>

    return
  }
}
