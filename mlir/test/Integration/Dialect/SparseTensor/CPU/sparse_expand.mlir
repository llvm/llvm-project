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
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

module {
  func.func private @printMemrefF64(%ptr : tensor<*xf64>)

  //
  // Column-wise storage forces the ijk loop to permute into jki
  // so that access pattern expansion (workspace) needs to be
  // done along dimension with size 8.
  //
  func.func @matmul(%A: tensor<8x2xf64, #CSC>,
                    %B: tensor<2x4xf64, #CSC>) -> tensor<8x4xf64, #CSC> {
    %C = bufferization.alloc_tensor() : tensor<8x4xf64, #CSC>
    %D = linalg.matmul
      ins(%A, %B: tensor<8x2xf64, #CSC>, tensor<2x4xf64, #CSC>)
         outs(%C: tensor<8x4xf64, #CSC>) -> tensor<8x4xf64, #CSC>
    return %D: tensor<8x4xf64, #CSC>
  }

  //
  // Main driver.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant -1.0 : f64

    // Initialize various dense matrices for stress testing.
    %da = arith.constant dense<[
        [ 1.1, 2.1 ],
        [ 1.2, 2.2 ],
        [ 1.3, 2.3 ],
        [ 1.4, 2.4 ],
        [ 1.5, 2.5 ],
        [ 1.6, 2.6 ],
        [ 1.7, 2.7 ],
        [ 1.8, 2.8 ]
    ]> : tensor<8x2xf64>
    %db = arith.constant dense<[
        [ 10.1, 11.1, 12.1, 13.1 ],
        [ 10.2, 11.2, 12.2, 13.2 ]
    ]> : tensor<2x4xf64>

    // Convert all these matrices to sparse format.
    %x1 = sparse_tensor.convert %da : tensor<8x2xf64> to tensor<8x2xf64, #CSC>
    %x2 = sparse_tensor.convert %db : tensor<2x4xf64> to tensor<2x4xf64, #CSC>

    // Call kernels with dense.
    %x3 = call @matmul(%x1, %x2)
       : (tensor<8x2xf64, #CSC>,
          tensor<2x4xf64, #CSC>) -> tensor<8x4xf64, #CSC>

    // CHECK:      {{\[}}[32.53,   35.73,   38.93,   42.13],
    // CHECK-NEXT: [34.56,   37.96,   41.36,   44.76],
    // CHECK-NEXT: [36.59,   40.19,   43.79,   47.39],
    // CHECK-NEXT: [38.62,   42.42,   46.22,   50.02],
    // CHECK-NEXT: [40.65,   44.65,   48.65,   52.65],
    // CHECK-NEXT: [42.68,   46.88,   51.08,   55.28],
    // CHECK-NEXT: [44.71,   49.11,   53.51,   57.91],
    // CHECK-NEXT: [46.74,   51.34,   55.94,   60.54]]
    //
    %xc = sparse_tensor.convert %x3 : tensor<8x4xf64, #CSC> to tensor<8x4xf64>
    %xu = tensor.cast %xc : tensor<8x4xf64> to tensor<*xf64>
    call @printMemrefF64(%xu) : (tensor<*xf64>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %x1 : tensor<8x2xf64, #CSC>
    bufferization.dealloc_tensor %x2 : tensor<2x4xf64, #CSC>
    bufferization.dealloc_tensor %x3 : tensor<8x4xf64, #CSC>

    return
  }
}
