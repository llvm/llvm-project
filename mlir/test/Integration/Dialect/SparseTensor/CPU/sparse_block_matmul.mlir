//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparse_compiler_opts} = enable-runtime-library=true
// DEFINE: %{sparse_compiler_opts_sve} = enable-arm-sve=true %{sparse_compiler_opts}
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_opts} = -e entry -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false enable-index-reduction=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false enable-index-reduction=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
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


#BSR = #sparse_tensor.encoding<{
  map = ( i, j ) ->
  ( i floordiv 2 : dense,
    j floordiv 3 : compressed,
    i mod 2      : dense,
    j mod 3      : dense
  )
}>

module {

func.func @mul(%arg0: tensor<4x6xf64>,
               %arg1: tensor<4x6xf64, #BSR>) -> tensor<4x4xf64> {
  %out = tensor.empty() : tensor<4x4xf64>
  %0 = linalg.generic #trait_mul
    ins(%arg0, %arg1: tensor<4x6xf64>, tensor<4x6xf64, #BSR>)
    outs(%out: tensor<4x4xf64>) {
      ^bb(%x: f64, %y : f64, %z : f64):
        %1 = arith.mulf %x, %y : f64
        %2 = arith.addf %1, %z : f64
        linalg.yield %2 : f64
  } -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

func.func @mul_dense(%arg0: tensor<4x6xf64>,
                     %arg1: tensor<4x6xf64>) -> tensor<4x4xf64> {
  %out = tensor.empty() : tensor<4x4xf64>
  %0 = linalg.generic #trait_mul
    ins(%arg0, %arg1: tensor<4x6xf64>, tensor<4x6xf64>)
    outs(%out: tensor<4x4xf64>) {
      ^bb(%x: f64, %y : f64, %z : f64):
        %1 = arith.mulf %x, %y : f64
        %2 = arith.addf %1, %z : f64
        linalg.yield %2 : f64
  } -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}


  //
  // Output utilities.
  //
  func.func @dumpf64(%arg0: tensor<4x4xf64>) {
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : f64
    %0 = vector.transfer_read %arg0[%c0, %c0], %d0: tensor<4x4xf64>, vector<4x4xf64>
    vector.print %0 : vector<4x4xf64>
    return
  }

  //
  // Main driver.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index


    %td = arith.constant dense<[[ 0.0,  1.0,  2.0,  3.0,  4.0,  5.0],
                                [ 6.0,  7.0,  8.0,  9.0, 10.0, 11.0],
                                [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                                [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]]> : tensor<4x6xf64>


    %2 = sparse_tensor.convert %td : tensor<4x6xf64> to tensor<4x6xf64, #BSR>

    %d = call @mul_dense(%td, %td)
         : (tensor<4x6xf64>, tensor<4x6xf64>) -> tensor<4x4xf64>
    %s = call @mul(%td, %2)
         : (tensor<4x6xf64>, tensor<4x6xf64, #BSR>) -> tensor<4x4xf64>

    // CHECK-COUNT-2: ( ( 55, 145, 235, 325 ), ( 145, 451, 757, 1063 ), ( 235, 757, 1279, 1801 ), ( 325, 1063, 1801, 2539 ) )
    call @dumpf64(%d) : (tensor<4x4xf64>) -> ()
    call @dumpf64(%s) : (tensor<4x4xf64>) -> ()

    return
  }
}
