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
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with vectorization.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false vl=4 enable-buffer-initialization=true
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
  func.func @entry() {
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


    // CHECK:     ( ( 0.1, 1.1, 0, 0, 0 ),
    // CHECK-SAME:  ( 0, 1.1, 2.2, 0, 0 ),
    // CHECK-SAME:  ( 0, 0, 2.1, 3.3, 0 ),
    // CHECK-SAME:  ( 0, 0, 0, 3.1, 4.4 ),
    // CHECK-SAME:  ( 0, 0, 0, 0, 4.1 ) )
    %r = sparse_tensor.convert %1 : tensor<5x5xf64, #DCSR> to tensor<5x5xf64>
    %v2 = vector.transfer_read %r[%c0, %c0], %f0 : tensor<5x5xf64>, vector<5x5xf64>
    vector.print %v2 : vector<5x5xf64>

    // Release the resources.
    bufferization.dealloc_tensor %sl: tensor<5x5xf64, #DCSR>
    bufferization.dealloc_tensor %sr: tensor<5x5xf64, #DCSR>
    bufferization.dealloc_tensor %1:  tensor<5x5xf64, #DCSR>

    return
  }
}
