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
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
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
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=4 enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#DCSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#sel_trait = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // C (in)
    affine_map<(i,j) -> (i,j)>,  // L (in)
    affine_map<(i,j) -> (i,j)>,  // R (in)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"]
}

module {
  func.func @sparse_select(%cond: tensor<5x5xi1>,
                           %arga: tensor<5x5xf64, #DCSR>,
                           %argb: tensor<5x5xf64, #DCSR>) -> tensor<5x5xf64, #DCSR> {
    %xv = tensor.empty() : tensor<5x5xf64, #DCSR>
    %0 = linalg.generic #sel_trait
       ins(%cond, %arga, %argb: tensor<5x5xi1>, tensor<5x5xf64, #DCSR>, tensor<5x5xf64, #DCSR>)
        outs(%xv: tensor<5x5xf64, #DCSR>) {
        ^bb(%c: i1, %a: f64, %b: f64, %x: f64):
          %1 = arith.select %c, %a, %b : f64
          linalg.yield %1 : f64
    } -> tensor<5x5xf64, #DCSR>
    return %0 : tensor<5x5xf64, #DCSR>
  }

  // Driver method to call and verify vector kernels.
  func.func @main() {
    %c0 = arith.constant 0   : index
    %f0 = arith.constant 0.0 : f64

    %cond = arith.constant sparse<
        [ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4] ],
        [     1,      1,      1,      1,      1  ]
    > : tensor<5x5xi1>
    %lhs = arith.constant sparse<
        [ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4] ],
        [   0.1,    1.1,    2.1,    3.1,    4.1  ]
    > : tensor<5x5xf64>
    %rhs = arith.constant sparse<
        [ [0, 1], [1, 2], [2, 3], [3, 4], [4, 4]],
        [   1.1,    2.2,    3.3,    4.4 ,   5.5 ]
    > : tensor<5x5xf64>

    %sl = sparse_tensor.convert %lhs : tensor<5x5xf64> to tensor<5x5xf64, #DCSR>
    %sr = sparse_tensor.convert %rhs : tensor<5x5xf64> to tensor<5x5xf64, #DCSR>

    // Call sparse matrix kernels.
    %1 = call @sparse_select(%cond, %sl, %sr) : (tensor<5x5xi1>,
                                                 tensor<5x5xf64, #DCSR>,
                                                 tensor<5x5xf64, #DCSR>) -> tensor<5x5xf64, #DCSR>


    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 9
    // CHECK-NEXT: dim = ( 5, 5 )
    // CHECK-NEXT: lvl = ( 5, 5 )
    // CHECK-NEXT: pos[0] : ( 0, 5 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3, 4 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 6, 8, 9 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 1, 2, 2, 3, 3, 4, 4 )
    // CHECK-NEXT: values : ( 0.1, 1.1, 1.1, 2.2, 2.1, 3.3, 3.1, 4.4, 4.1 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %1 : tensor<5x5xf64, #DCSR>

    // Release the resources.
    bufferization.dealloc_tensor %sl: tensor<5x5xf64, #DCSR>
    bufferization.dealloc_tensor %sr: tensor<5x5xf64, #DCSR>
    bufferization.dealloc_tensor %1:  tensor<5x5xf64, #DCSR>

    return
  }
}
