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
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#trait_mul = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>,  // A (in)
    affine_map<(i,j,k) -> (j,k)>,  // B (in, transposed)
    affine_map<(i,j,k) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) *= A(i,j) * B(j,i)"
}

#CSR = #sparse_tensor.encoding<{
  map = ( i, j ) -> (i : dense, j : compressed)
}>

#BSR = #sparse_tensor.encoding<{
  map = ( i, j ) ->
  ( i floordiv 2 : dense,
    j floordiv 2 : compressed,
    i mod 2      : dense,
    j mod 2      : dense
  )
}>

#NV_24 = #sparse_tensor.encoding<{
  map = ( i, j ) ->
  ( i            : dense,
    j floordiv 4 : dense,
    j mod 4      : structured[2, 4]
  ),
}>

module {

  func.func @mul(%arg0: tensor<4x8xf64>,
                 %arg1: tensor<4x8xf64, #BSR>) -> tensor<4x4xf64> {
    %out = arith.constant dense<0.0> : tensor<4x4xf64>
    %0 = linalg.generic #trait_mul
      ins(%arg0, %arg1: tensor<4x8xf64>, tensor<4x8xf64, #BSR>)
      outs(%out: tensor<4x4xf64>) {
        ^bb(%x: f64, %y : f64, %z : f64):
          %1 = arith.mulf %x, %y : f64
          %2 = arith.addf %1, %z : f64
          linalg.yield %2 : f64
    } -> tensor<4x4xf64>
    return %0 : tensor<4x4xf64>
  }

  func.func @mul_24(%arg0: tensor<4x8xf64>,
                    %arg1: tensor<4x8xf64, #NV_24>) -> tensor<4x4xf64> {
    %out = arith.constant dense<0.0> : tensor<4x4xf64>
    %0 = linalg.generic #trait_mul
      ins(%arg0, %arg1: tensor<4x8xf64>, tensor<4x8xf64, #NV_24>)
      outs(%out: tensor<4x4xf64>) {
        ^bb(%x: f64, %y : f64, %z : f64):
          %1 = arith.mulf %x, %y : f64
          %2 = arith.addf %1, %z : f64
          linalg.yield %2 : f64
    } -> tensor<4x4xf64>
    return %0 : tensor<4x4xf64>
  }

  func.func @mul_csr_bsr(%arg0: tensor<4x8xf64, #CSR>,
                         %arg1: tensor<4x8xf64, #BSR>) -> tensor<4x4xf64> {
    %out = arith.constant dense<0.0> : tensor<4x4xf64>
    %0 = linalg.generic #trait_mul
      ins(%arg0, %arg1: tensor<4x8xf64, #CSR>, tensor<4x8xf64, #BSR>)
      outs(%out: tensor<4x4xf64>) {
        ^bb(%x: f64, %y : f64, %z : f64):
          %1 = arith.mulf %x, %y : f64
          %2 = arith.addf %1, %z : f64
          linalg.yield %2 : f64
    } -> tensor<4x4xf64>
    return %0 : tensor<4x4xf64>
  }

  func.func @mul_dense(%arg0: tensor<4x8xf64>,
                       %arg1: tensor<4x8xf64>) -> tensor<4x4xf64> {
    %out = arith.constant dense<0.0> : tensor<4x4xf64>
    %0 = linalg.generic #trait_mul
      ins(%arg0, %arg1: tensor<4x8xf64>, tensor<4x8xf64>)
      outs(%out: tensor<4x4xf64>) {
        ^bb(%x: f64, %y : f64, %z : f64):
          %1 = arith.mulf %x, %y : f64
          %2 = arith.addf %1, %z : f64
          linalg.yield %2 : f64
    } -> tensor<4x4xf64>
    return %0 : tensor<4x4xf64>
  }

  //
  // Output utility.
  //
  func.func @dump_dense_f64(%arg0: tensor<4x4xf64>) {
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : f64
    %0 = vector.transfer_read %arg0[%c0, %c0], %d0: tensor<4x4xf64>, vector<4x4xf64>
    vector.print %0 : vector<4x4xf64>
    return
  }

  //
  // Main driver.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index

    %td = arith.constant dense<[[ 1.0, 2.0,  0.0,  0.0,  0.0,  0.0,  4.0,  5.0],
                                [ 6.0, 7.0,  0.0,  0.0,  0.0,  0.0, 10.0, 11.0],
                                [ 0.0, 0.0, 12.0, 13.0, 16.0, 17.0,  0.0,  0.0],
                                [ 0.0, 0.0, 18.0, 19.0, 22.0, 23.0,  0.0,  0.0]]> : tensor<4x8xf64>

    %a = sparse_tensor.convert %td : tensor<4x8xf64> to tensor<4x8xf64, #BSR>
    %b = sparse_tensor.convert %td : tensor<4x8xf64> to tensor<4x8xf64, #NV_24>
    %c = sparse_tensor.convert %td : tensor<4x8xf64> to tensor<4x8xf64, #CSR>

    %d = call @mul_dense(%td, %td)
         : (tensor<4x8xf64>, tensor<4x8xf64>) -> tensor<4x4xf64>
    %s = call @mul(%td, %a)
         : (tensor<4x8xf64>, tensor<4x8xf64, #BSR>) -> tensor<4x4xf64>
    %s24 = call @mul_24(%td, %b)
         : (tensor<4x8xf64>, tensor<4x8xf64, #NV_24>) -> tensor<4x4xf64>
    %scsr = call @mul_csr_bsr(%c, %a)
         : (tensor<4x8xf64, #CSR>, tensor<4x8xf64, #BSR>) -> tensor<4x4xf64>

    // CHECK-COUNT-4: ( ( 46, 115, 0, 0 ), ( 115, 306, 0, 0 ), ( 0, 0, 858, 1206 ), ( 0, 0, 1206, 1698 ) )
    call @dump_dense_f64(%d)    : (tensor<4x4xf64>) -> ()
    call @dump_dense_f64(%s)    : (tensor<4x4xf64>) -> ()
    call @dump_dense_f64(%s24)  : (tensor<4x4xf64>) -> ()
    call @dump_dense_f64(%scsr) : (tensor<4x4xf64>) -> ()

    bufferization.dealloc_tensor %a : tensor<4x8xf64, #BSR>
    bufferization.dealloc_tensor %b : tensor<4x8xf64, #NV_24>
    bufferization.dealloc_tensor %c : tensor<4x8xf64, #CSR>
    bufferization.dealloc_tensor %d : tensor<4x4xf64>
    bufferization.dealloc_tensor %s : tensor<4x4xf64>
    bufferization.dealloc_tensor %s24 : tensor<4x4xf64>
    bufferization.dealloc_tensor %scsr : tensor<4x4xf64>

    return
  }
}
