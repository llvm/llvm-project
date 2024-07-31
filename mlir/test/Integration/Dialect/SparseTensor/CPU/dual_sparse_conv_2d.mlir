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
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#DCSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>
#CSR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : dense, d1 : compressed)}>
#CDR = #sparse_tensor.encoding<{map = (d0, d1) -> (d0 : compressed, d1 : dense)}>
#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed)
}>

// An example of a 2D convolution with a sparse filter.
module {

  func.func @conv2d(%input:  tensor<8x8xi32>,
                    %filter: tensor<3x3xi32>,
                    %output: tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<8x8xi32>, tensor<3x3xi32>)
      outs (%output: tensor<6x6xi32>) -> tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
  }

  func.func @conv2d_all_sparse_DCSR(%input:  tensor<8x8xi32, #DCSR>,
                                    %filter: tensor<3x3xi32, #DCSR>) -> tensor<6x6xi32, #DCSR> {
    %s = tensor.empty() : tensor<6x6xi32, #DCSR>
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<8x8xi32, #DCSR>, tensor<3x3xi32, #DCSR>)
      outs (%s: tensor<6x6xi32, #DCSR>) -> tensor<6x6xi32, #DCSR>
    return %0 : tensor<6x6xi32, #DCSR>
  }

  func.func @conv2d_all_sparse_CSR(%input:  tensor<8x8xi32, #CSR>,
                                   %filter: tensor<3x3xi32, #CSR>) -> tensor<6x6xi32, #CSR> {
    %s = tensor.empty() : tensor<6x6xi32, #CSR>
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<8x8xi32, #CSR>, tensor<3x3xi32, #CSR>)
      outs (%s: tensor<6x6xi32, #CSR>) -> tensor<6x6xi32, #CSR>
    return %0 : tensor<6x6xi32, #CSR>
  }

  func.func @conv2d_all_sparse_CD(%input:  tensor<8x8xi32, #CDR>,
                                  %filter: tensor<3x3xi32, #CDR>) -> tensor<6x6xi32, #CDR> {
    %s = tensor.empty() : tensor<6x6xi32, #CDR>
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<8x8xi32, #CDR>, tensor<3x3xi32, #CDR>)
      outs (%s: tensor<6x6xi32, #CDR>) -> tensor<6x6xi32, #CDR>
    return %0 : tensor<6x6xi32, #CDR>
  }

  func.func @conv2d_all_sparse_CSC(%input:  tensor<8x8xi32, #CSC>,
                                   %filter: tensor<3x3xi32, #CSC>) -> tensor<6x6xi32, #CSC> {
    %s = tensor.empty() : tensor<6x6xi32, #CSC>
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<8x8xi32, #CSC>, tensor<3x3xi32, #CSC>)
      outs (%s: tensor<6x6xi32, #CSC>) -> tensor<6x6xi32, #CSC>
    return %0 : tensor<6x6xi32, #CSC>
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %i0 = arith.constant 0 : i32

    // A typical edge detection filter.
    %filter = arith.constant dense<[
      [  1,  0, -1 ],
      [  0,  0,  0 ],
      [ -1,  0,  1 ]
    ]> : tensor<3x3xi32>
    %sparse_filter_DCSR = sparse_tensor.convert %filter
      : tensor<3x3xi32> to tensor<3x3xi32, #DCSR>
    %sparse_filter_CSR = sparse_tensor.convert %filter
      : tensor<3x3xi32> to tensor<3x3xi32, #CSR>
    %sparse_filter_CD = sparse_tensor.convert %filter
      : tensor<3x3xi32> to tensor<3x3xi32, #CDR>
    %sparse_filter_CSC = sparse_tensor.convert %filter
      : tensor<3x3xi32> to tensor<3x3xi32, #CSC>

    %input = arith.constant dense<[
      [  1,  2,  3,  4,  0,  6,  7,  8 ],
      [  2,  2,  4,  4,  0,  0,  6,  8 ],
      [  2,  2,  4,  4,  0,  0,  6,  8 ],
      [  2,  2,  3,  4,  0,  0,  7,  8 ],
      [  1,  3,  3,  4,  0,  0,  6,  8 ],
      [  3,  2,  3,  4,  0,  0,  7,  8 ],
      [  1,  3,  3,  4,  3,  6,  6,  8 ],
      [  1,  3,  3,  4,  3,  0,  7,  8 ]
    ]> : tensor<8x8xi32>
    %sparse_input_DCSR = sparse_tensor.convert %input
      : tensor<8x8xi32> to tensor<8x8xi32, #DCSR>
    %sparse_input_CSR = sparse_tensor.convert %input
      : tensor<8x8xi32> to tensor<8x8xi32, #CSR>
    %sparse_input_CD = sparse_tensor.convert %input
      : tensor<8x8xi32> to tensor<8x8xi32, #CDR>
    %sparse_input_CSC = sparse_tensor.convert %input
      : tensor<8x8xi32> to tensor<8x8xi32, #CSC>

    // Call the kernel.
    %output = arith.constant dense<0> : tensor<6x6xi32>
    %0 = call @conv2d(%input, %filter, %output)
       : (tensor<8x8xi32>,
          tensor<3x3xi32>, tensor<6x6xi32>) -> tensor<6x6xi32>
    %2 = call @conv2d_all_sparse_DCSR(%sparse_input_DCSR, %sparse_filter_DCSR)
       : (tensor<8x8xi32, #DCSR>,
          tensor<3x3xi32, #DCSR>) -> tensor<6x6xi32, #DCSR>
    %3 = call @conv2d_all_sparse_CSR(%sparse_input_CSR, %sparse_filter_CSR)
       : (tensor<8x8xi32, #CSR>,
          tensor<3x3xi32, #CSR>) -> tensor<6x6xi32, #CSR>
    %4 = call @conv2d_all_sparse_CD(%sparse_input_CD, %sparse_filter_CD)
       : (tensor<8x8xi32, #CDR>,
          tensor<3x3xi32, #CDR>) -> tensor<6x6xi32, #CDR>
    %5 = call @conv2d_all_sparse_CSC(%sparse_input_CSC, %sparse_filter_CSC)
       : (tensor<8x8xi32, #CSC>,
          tensor<3x3xi32, #CSC>) -> tensor<6x6xi32, #CSC>

    // Verify the output.
    //
    // CHECK:    ( ( 0, 0, -1, -6, -1, 6 ),
    // CHECK-SAME: ( -1, 0, 1, 0, 1, 0 ),
    // CHECK-SAME: ( 0, -1, 1, 0, 0, 0 ),
    // CHECK-SAME: ( -1, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 3, 6, -3, -6 ),
    // CHECK-SAME: ( 2, -1, 3, 0, -3, 0 ) )
    //
    %v = vector.transfer_read %0[%c0, %c0], %i0
      : tensor<6x6xi32>, vector<6x6xi32>
    vector.print %v : vector<6x6xi32>

    //
    // Should be the same as dense output.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 36
    // CHECK-NEXT: dim = ( 6, 6 )
    // CHECK-NEXT: lvl = ( 6, 6 )
    // CHECK-NEXT: pos[0] : ( 0, 6 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3, 4, 5 )
    // CHECK-NEXT: pos[1] : ( 0, 6, 12, 18, 24, 30, 36 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5 )
    // CHECK-NEXT: values : ( 0, 0, -1, -6, -1, 6, -1, 0, 1, 0, 1, 0, 0, -1, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 3, 6, -3, -6, 2, -1, 3, 0, -3, 0 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %2 : tensor<6x6xi32, #DCSR>

    //
    // Should be the same as dense output.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 36
    // CHECK-NEXT: dim = ( 6, 6 )
    // CHECK-NEXT: lvl = ( 6, 6 )
    // CHECK-NEXT: pos[1] : ( 0, 6, 12, 18, 24, 30, 36 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5 )
    // CHECK-NEXT: values : ( 0, 0, -1, -6, -1, 6, -1, 0, 1, 0, 1, 0, 0, -1, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 3, 6, -3, -6, 2, -1, 3, 0, -3, 0 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %3 : tensor<6x6xi32, #CSR>

    //
    // Should be the same as dense output.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 36
    // CHECK-NEXT: dim = ( 6, 6 )
    // CHECK-NEXT: lvl = ( 6, 6 )
    // CHECK-NEXT: pos[0] : ( 0, 6 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3, 4, 5 )
    // CHECK-NEXT: values : ( 0, 0, -1, -6, -1, 6, -1, 0, 1, 0, 1, 0, 0, -1, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 3, 6, -3, -6, 2, -1, 3, 0, -3, 0 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %4 : tensor<6x6xi32, #CDR>

    //
    // Should be the same as dense output.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 36
    // CHECK-NEXT: dim = ( 6, 6 )
    // CHECK-NEXT: lvl = ( 6, 6 )
    // CHECK-NEXT: pos[1] : ( 0, 6, 12, 18, 24, 30, 36 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5 )
    // CHECK-NEXT: values : ( 0, -1, 0, -1, 0, 2, 0, 0, -1, 0, 0, -1, -1, 1, 1, 0, 3, 3, -6, 0, 0, 0, 6, 0, -1, 1, 0, 0, -3, -3, 6, 0, 0, 0, -6, 0 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %5 : tensor<6x6xi32, #CSC>

    // Release the resources.
    bufferization.dealloc_tensor %sparse_input_DCSR : tensor<8x8xi32, #DCSR>
    bufferization.dealloc_tensor %sparse_input_CSR : tensor<8x8xi32, #CSR>
    bufferization.dealloc_tensor %sparse_input_CSC : tensor<8x8xi32, #CSC>
    bufferization.dealloc_tensor %sparse_input_CD : tensor<8x8xi32, #CDR>
    bufferization.dealloc_tensor %sparse_filter_DCSR : tensor<3x3xi32, #DCSR>
    bufferization.dealloc_tensor %sparse_filter_CSR : tensor<3x3xi32, #CSR>
    bufferization.dealloc_tensor %sparse_filter_CD : tensor<3x3xi32, #CDR>
    bufferization.dealloc_tensor %sparse_filter_CSC : tensor<3x3xi32, #CSC>

    bufferization.dealloc_tensor %0 : tensor<6x6xi32>
    bufferization.dealloc_tensor %2 : tensor<6x6xi32, #DCSR>
    bufferization.dealloc_tensor %3 : tensor<6x6xi32, #CSR>
    bufferization.dealloc_tensor %4 : tensor<6x6xi32, #CDR>
    bufferization.dealloc_tensor %5 : tensor<6x6xi32, #CSC>
    return
  }
}
