// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{command} = mlir-opt %s --sparse-compiler=%{option} | \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{command}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = enable-runtime-library=false
// RUN: %{command}

#CSC = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

module {
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

    //
    // CHECK:    ( ( 32.53, 35.73, 38.93, 42.13 ),
    // CHECK-SAME: ( 34.56, 37.96, 41.36, 44.76 ),
    // CHECK-SAME: ( 36.59, 40.19, 43.79, 47.39 ),
    // CHECK-SAME: ( 38.62, 42.42, 46.22, 50.02 ),
    // CHECK-SAME: ( 40.65, 44.65, 48.65, 52.65 ),
    // CHECK-SAME: ( 42.68, 46.88, 51.08, 55.28 ),
    // CHECK-SAME: ( 44.71, 49.11, 53.51, 57.91 ),
    // CHECK-SAME: ( 46.74, 51.34, 55.94, 60.54 ) )
    //
    %xc = sparse_tensor.convert %x3 : tensor<8x4xf64, #CSC> to tensor<8x4xf64>
    %xv = vector.transfer_read %xc[%c0, %c0], %d1 : tensor<8x4xf64>, vector<8x4xf64>
    vector.print %xv : vector<8x4xf64>

    // Release the resources.
    bufferization.dealloc_tensor %x1 : tensor<8x2xf64, #CSC>
    bufferization.dealloc_tensor %x2 : tensor<2x4xf64, #CSC>
    bufferization.dealloc_tensor %x3 : tensor<8x4xf64, #CSC>

    return
  }
}
