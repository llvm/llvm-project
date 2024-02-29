//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
// do not use these LIT config files. Hence why this is kept inline.
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

// REDEFINE: %{env} = TENSOR0="%mlir_src_dir/test/Integration/data/block.mtx"
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

#BSR = #sparse_tensor.encoding<{
  map = (i, j) ->
    ( i floordiv 2 : dense
    , j floordiv 2 : compressed
    , i mod 2 : dense
    , j mod 2 : dense
    )
}>

#DSDD = #sparse_tensor.encoding<{
  map = (i, j, k, l) -> ( i  : dense, j  : compressed, k  : dense, l  : dense)
}>

#trait_scale_inplace = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"]
}

//
// Example 2x2 block storage:
//
//  +-----+-----+-----+    +-----+-----+-----+
//  | 1 2 | . . | 4 . |    | 1 2 |     | 4 0 |
//  | . 3 | . . | . 5 |    | 0 3 |     | 0 5 |
//  +-----+-----+-----+ => +-----+-----+-----+
//  | . . | 6 7 | . . |    |     | 6 7 |     |
//  | . . | 8 . | . . |    |     | 8 0 |     |
//  +-----+-----+-----+    +-----+-----+-----+
//
// Stored as:
//
//    positions[1]   : 0 2 3
//    coordinates[1] : 0 2 1
//    values         : 1.000000 2.000000 0.000000 3.000000 4.000000 0.000000 0.000000 5.000000 6.000000 7.000000 8.000000 0.000000
//
module {

  func.func private @getTensorFilename(index) -> (!Filename)

  func.func @scale(%arg0: tensor<?x?xf64, #BSR>) -> tensor<?x?xf64, #BSR> {
    %c = arith.constant 3.0 : f64
    %0 = linalg.generic #trait_scale_inplace
      outs(%arg0: tensor<?x?xf64, #BSR>) {
        ^bb(%x: f64):
          %1 = arith.mulf %x, %c : f64
          linalg.yield %1 : f64
      } -> tensor<?x?xf64, #BSR>
    return %0 : tensor<?x?xf64, #BSR>
  }

  func.func @main() {
    %c0 = arith.constant 0   : index
    %f0 = arith.constant 0.0 : f64

    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %A = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #BSR>

    // CHECK:   ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: pos[1] : ( 0, 2, 3,
    // CHECK-NEXT: crd[1] : ( 0, 2, 1,
    // CHECK-NEXT: values : ( 1, 2, 0, 3, 4, 0, 0, 5, 6, 7, 8, 0,
    // CHECK-NEXT: ----
    sparse_tensor.print %A : tensor<?x?xf64, #BSR>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: pos[1] : ( 0, 2, 3,
    // CHECK-NEXT: crd[1] : ( 0, 2, 1
    // CHECK-NEXT: values : ( 1, 2, 0, 3, 4, 0, 0, 5, 6, 7, 8, 0,
    // CHECK-NEXT: ----
    %t1 = sparse_tensor.reinterpret_map %A : tensor<?x?xf64, #BSR>
                                          to tensor<?x?x2x2xf64, #DSDD>
    sparse_tensor.print %t1 : tensor<?x?x2x2xf64, #DSDD>

    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: pos[1] : ( 0, 2, 3,
    // CHECK-NEXT: crd[1] : ( 0, 2, 1,
    // CHECK-NEXT: values : ( 3, 6, 0, 9, 12, 0, 0, 15, 18, 21, 24, 0,
    // CHECK-NEXT: ----
    %As = call @scale(%A) : (tensor<?x?xf64, #BSR>) -> (tensor<?x?xf64, #BSR>)
    sparse_tensor.print %As : tensor<?x?xf64, #BSR>

    // Release the resources.
    bufferization.dealloc_tensor %A: tensor<?x?xf64, #BSR>

    return
  }
}
