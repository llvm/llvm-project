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

#DCSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#trait_mult_elt = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>,  // B
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * B(i,j)"
}

module {
  // Sparse kernel.
  func.func @sparse_mult_elt(
      %arga: tensor<32x16xf32, #DCSR>, %argb: tensor<32x16xf32, #DCSR>) -> tensor<32x16xf32, #DCSR> {
    %argx = tensor.empty() : tensor<32x16xf32, #DCSR>
    %0 = linalg.generic #trait_mult_elt
      ins(%arga, %argb: tensor<32x16xf32, #DCSR>, tensor<32x16xf32, #DCSR>)
      outs(%argx: tensor<32x16xf32, #DCSR>) {
        ^bb(%a: f32, %b: f32, %x: f32):
          %1 = arith.mulf %a, %b : f32
          linalg.yield %1 : f32
    } -> tensor<32x16xf32, #DCSR>
    return %0 : tensor<32x16xf32, #DCSR>
  }

  // Driver method to call and verify kernel.
  func.func @main() {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32

    // Setup very sparse matrices.
    %ta = arith.constant sparse<
       [ [2,2], [15,15], [31,0], [31,14] ], [ 2.0, 3.0, -2.0, 4.0 ]
    > : tensor<32x16xf32>
    %tb = arith.constant sparse<
       [ [1,1], [2,0], [2,2], [2,15], [31,0], [31,15] ], [ 5.0, 6.0, 7.0, 8.0, -10.0, 9.0 ]
    > : tensor<32x16xf32>
    %sta = sparse_tensor.convert %ta
      : tensor<32x16xf32> to tensor<32x16xf32, #DCSR>
    %stb = sparse_tensor.convert %tb
      : tensor<32x16xf32> to tensor<32x16xf32, #DCSR>

    // Call kernel.
    %0 = call @sparse_mult_elt(%sta, %stb)
      : (tensor<32x16xf32, #DCSR>,
         tensor<32x16xf32, #DCSR>) -> tensor<32x16xf32, #DCSR>

    //
    // Verify results. Only two entries stored in result!
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 2
    // CHECK-NEXT: dim = ( 32, 16 )
    // CHECK-NEXT: lvl = ( 32, 16 )
    // CHECK-NEXT: pos[0] : ( 0, 2
    // CHECK-NEXT: crd[0] : ( 2, 31
    // CHECK-NEXT: pos[1] : ( 0, 1, 2
    // CHECK-NEXT: crd[1] : ( 2, 0
    // CHECK-NEXT: values : ( 14, 20
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %0 : tensor<32x16xf32, #DCSR>

    // Release the resources.
    bufferization.dealloc_tensor %sta : tensor<32x16xf32, #DCSR>
    bufferization.dealloc_tensor %stb : tensor<32x16xf32, #DCSR>
    bufferization.dealloc_tensor %0   : tensor<32x16xf32, #DCSR>
    return
  }
}
