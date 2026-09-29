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
// Do the same run, but now with vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#DCSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#DCSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : compressed, d0 : compressed)
}>

#transpose_trait = {
  indexing_maps = [
    affine_map<(i,j) -> (j,i)>,  // A
    affine_map<(i,j) -> (i,j)>   // X
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(j,i)"
}

module {

  //
  // Transposing a sparse row-wise matrix into another sparse row-wise
  // matrix introduces a cycle in the iteration graph. This complication
  // can be avoided by manually inserting a conversion of the incoming
  // matrix into a sparse column-wise matrix first.
  //
  func.func @sparse_transpose(%arga: tensor<3x4xf64, #DCSR>)
                                  -> tensor<4x3xf64, #DCSR> {
    %t = sparse_tensor.convert %arga
      : tensor<3x4xf64, #DCSR> to tensor<3x4xf64, #DCSC>

    %i = tensor.empty() : tensor<4x3xf64, #DCSR>
    %0 = linalg.generic #transpose_trait
       ins(%t: tensor<3x4xf64, #DCSC>)
       outs(%i: tensor<4x3xf64, #DCSR>) {
       ^bb(%a: f64, %x: f64):
         linalg.yield %a : f64
    } -> tensor<4x3xf64, #DCSR>

    bufferization.dealloc_tensor %t : tensor<3x4xf64, #DCSC>

    return %0 : tensor<4x3xf64, #DCSR>
  }

  //
  // However, even better, the sparsifier is able to insert such a
  // conversion automatically to resolve a cycle in the iteration graph!
  //
  func.func @sparse_transpose_auto(%arga: tensor<3x4xf64, #DCSR>)
                                       -> tensor<4x3xf64, #DCSR> {
    %i = tensor.empty() : tensor<4x3xf64, #DCSR>
    %0 = linalg.generic #transpose_trait
       ins(%arga: tensor<3x4xf64, #DCSR>)
       outs(%i: tensor<4x3xf64, #DCSR>) {
       ^bb(%a: f64, %x: f64):
         linalg.yield %a : f64
    } -> tensor<4x3xf64, #DCSR>
    return %0 : tensor<4x3xf64, #DCSR>
  }

  //
  // Main driver.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %du = arith.constant 0.0 : f64

    // Setup input sparse matrix from compressed constant.
    %d = arith.constant dense <[
       [ 1.1,  1.2,  0.0,  1.4 ],
       [ 0.0,  0.0,  0.0,  0.0 ],
       [ 3.1,  0.0,  3.3,  3.4 ]
    ]> : tensor<3x4xf64>
    %a = sparse_tensor.convert %d : tensor<3x4xf64> to tensor<3x4xf64, #DCSR>

    // Call the kernels.
    %0 = call @sparse_transpose(%a)
      : (tensor<3x4xf64, #DCSR>) -> tensor<4x3xf64, #DCSR>
    %1 = call @sparse_transpose_auto(%a)
      : (tensor<3x4xf64, #DCSR>) -> tensor<4x3xf64, #DCSR>

    //
    // Verify result.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 6
    // CHECK-NEXT: dim = ( 4, 3 )
    // CHECK-NEXT: lvl = ( 4, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 4 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3, 4, 6 )
    // CHECK-NEXT: crd[1] : ( 0, 2, 0, 2, 0, 2 )
    // CHECK-NEXT: values : ( 1.1, 3.1, 1.2, 3.3, 1.4, 3.4 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 6
    // CHECK-NEXT: dim = ( 4, 3 )
    // CHECK-NEXT: lvl = ( 4, 3 )
    // CHECK-NEXT: pos[0] : ( 0, 4 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3, 4, 6 )
    // CHECK-NEXT: crd[1] : ( 0, 2, 0, 2, 0, 2 )
    // CHECK-NEXT: values : ( 1.1, 3.1, 1.2, 3.3, 1.4, 3.4 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %0 : tensor<4x3xf64, #DCSR>
    sparse_tensor.print %1 : tensor<4x3xf64, #DCSR>

    // Release resources.
    bufferization.dealloc_tensor %a : tensor<3x4xf64, #DCSR>
    bufferization.dealloc_tensor %0 : tensor<4x3xf64, #DCSR>
    bufferization.dealloc_tensor %1 : tensor<4x3xf64, #DCSR>

    return
  }
}
