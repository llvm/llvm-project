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
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense,
                     d1 : compressed)
}>

#DCSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed,
                     d1 : compressed)
}>

// An example of a 2D convolution with sparse data and filter.
module {
  func.func @conv2d(%input:  tensor<10x10xi32>,
                    %filter: tensor<5x5xi32>,
                    %output: tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<10x10xi32>, tensor<5x5xi32>)
      outs (%output: tensor<6x6xi32>) -> tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
  }

  func.func @conv2d_ss(%input:  tensor<10x10xi32, #CSR>,
                       %filter: tensor<5x5xi32, #CSR>,
                       %output: tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<10x10xi32, #CSR>, tensor<5x5xi32, #CSR>)
      outs (%output: tensor<6x6xi32>) -> tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
  }

  func.func @conv2d_bs(%input:  tensor<10x10xi32, #DCSR>,
                       %filter: tensor<5x5xi32, #CSR>,
                       %output: tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<10x10xi32, #DCSR>, tensor<5x5xi32, #CSR>)
      outs (%output: tensor<6x6xi32>) -> tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %i0 = arith.constant 0 : i32

    // Dense filter and input to "stress" test sparsity.

    %filter = arith.constant dense<[
      [  -1,  -2,  -3,  -4,  -5 ],
      [  -6,  -7,  -8,  -9, -10 ],
      [ -11, -12, -13, -14, -15 ],
      [ -16, -17, -18, -19, -20 ],
      [ -21, -22, -23, -24, -25 ]
    ]> : tensor<5x5xi32>

    %input = arith.constant dense<[
      [  0,  1,  2,  3,  4,  5,  6,  7,  8,  9 ],
      [ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 ],
      [ 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 ],
      [ 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 ],
      [ 40, 41, 42, 43, 44, 45, 46, 47, 48, 49 ],
      [ 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 ],
      [ 60, 61, 62, 63, 64, 65, 66, 67, 68, 69 ],
      [ 70, 71, 72, 73, 74, 75, 76, 77, 78, 79 ],
      [ 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 ],
      [ 90, 91, 92, 93, 94, 95, 96, 97, 98, 99 ]
    ]> : tensor<10x10xi32>

    // Sparse filter and input to test true sparsity.

    %sfilter = arith.constant dense<[
      [  0, -1,  0, -2,  0 ],
      [  0,  0,  0,  0,  0 ],
      [  0,  0,  8,  0,  0 ],
      [ -3,  0,  0, -4,  0 ],
      [  0,  0, -5,  0, -6 ]
    ]> : tensor<5x5xi32>

    %sinput = arith.constant dense<[
      [  0,  1,  2,  3,  0,  0,  0,  0,  0,  0 ],
      [  0,  4,  0,  0,  5,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  6,  0,  0,  7 ],
      [  0,  0,  0,  0,  0,  0,  0,  8,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  9,  0,  0,  0,  0,  0,  0,  0,  0 ],
      [  0,  0,  0,  0, 10,  0,  0,  0,  0,  0 ]
    ]> : tensor<10x10xi32>

    // Set up sparse tensors.

    %input_CSR = sparse_tensor.convert %input : tensor<10x10xi32> to tensor<10x10xi32, #CSR>
    %input_DCSR = sparse_tensor.convert %input : tensor<10x10xi32> to tensor<10x10xi32, #DCSR>
    %filter_CSR = sparse_tensor.convert %filter : tensor<5x5xi32> to tensor<5x5xi32, #CSR>

    %sinput_CSR = sparse_tensor.convert %sinput : tensor<10x10xi32> to tensor<10x10xi32, #CSR>
    %sinput_DCSR = sparse_tensor.convert %sinput : tensor<10x10xi32> to tensor<10x10xi32, #DCSR>
    %sfilter_CSR = sparse_tensor.convert %sfilter : tensor<5x5xi32> to tensor<5x5xi32, #CSR>

    // Call the kernels with stress input.
    %output0 = arith.constant dense<0> : tensor<6x6xi32>
    %0 = call @conv2d(%input, %filter, %output0)
       : (tensor<10x10xi32>, tensor<5x5xi32>, tensor<6x6xi32>) -> tensor<6x6xi32>
    %output1 = arith.constant dense<0> : tensor<6x6xi32>
    %1 = call @conv2d_ss(%input_CSR, %filter_CSR, %output1)
       : (tensor<10x10xi32, #CSR>, tensor<5x5xi32, #CSR>, tensor<6x6xi32>) -> tensor<6x6xi32>
    %output2 = arith.constant dense<0> : tensor<6x6xi32>
    %2 = call @conv2d_bs(%input_DCSR, %filter_CSR, %output2)
       : (tensor<10x10xi32, #DCSR>, tensor<5x5xi32, #CSR>, tensor<6x6xi32>) -> tensor<6x6xi32>

    // Call the kernels with sparse input.
    %output3 = arith.constant dense<0> : tensor<6x6xi32>
    %3 = call @conv2d(%sinput, %sfilter, %output3)
       : (tensor<10x10xi32>, tensor<5x5xi32>, tensor<6x6xi32>) -> tensor<6x6xi32>
    %output4 = arith.constant dense<0> : tensor<6x6xi32>
    %4 = call @conv2d_ss(%sinput_CSR, %sfilter_CSR, %output4)
       : (tensor<10x10xi32, #CSR>, tensor<5x5xi32, #CSR>, tensor<6x6xi32>) -> tensor<6x6xi32>
    %output5 = arith.constant dense<0> : tensor<6x6xi32>
    %5 = call @conv2d_bs(%sinput_DCSR, %sfilter_CSR, %output5)
       : (tensor<10x10xi32, #DCSR>, tensor<5x5xi32, #CSR>, tensor<6x6xi32>) -> tensor<6x6xi32>

    // Verify the output.
    //
    // CHECK:    ( ( -9700, -10025, -10350, -10675, -11000, -11325 ),
    // CHECK-SAME: ( -12950, -13275, -13600, -13925, -14250, -14575 ),
    // CHECK-SAME: ( -16200, -16525, -16850, -17175, -17500, -17825 ),
    // CHECK-SAME: ( -19450, -19775, -20100, -20425, -20750, -21075 ),
    // CHECK-SAME: ( -22700, -23025, -23350, -23675, -24000, -24325 ),
    // CHECK-SAME: ( -25950, -26275, -26600, -26925, -27250, -27575 ) )
    //
    // CHECK:    ( ( -9700, -10025, -10350, -10675, -11000, -11325 ),
    // CHECK-SAME: ( -12950, -13275, -13600, -13925, -14250, -14575 ),
    // CHECK-SAME: ( -16200, -16525, -16850, -17175, -17500, -17825 ),
    // CHECK-SAME: ( -19450, -19775, -20100, -20425, -20750, -21075 ),
    // CHECK-SAME: ( -22700, -23025, -23350, -23675, -24000, -24325 ),
    // CHECK-SAME: ( -25950, -26275, -26600, -26925, -27250, -27575 ) )
    //
    // CHECK:    ( ( -9700, -10025, -10350, -10675, -11000, -11325 ),
    // CHECK-SAME: ( -12950, -13275, -13600, -13925, -14250, -14575 ),
    // CHECK-SAME: ( -16200, -16525, -16850, -17175, -17500, -17825 ),
    // CHECK-SAME: ( -19450, -19775, -20100, -20425, -20750, -21075 ),
    // CHECK-SAME: ( -22700, -23025, -23350, -23675, -24000, -24325 ),
    // CHECK-SAME: ( -25950, -26275, -26600, -26925, -27250, -27575 ) )
    //
    // CHECK:    ( ( -7, -2, -39, 0, -30, -42 ),
    // CHECK-SAME: ( -4, -10, 0, -77, 0, -40 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 16, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 64 ),
    // CHECK-SAME: ( 0, 0, 0, -12, 0, -6 ),
    // CHECK-SAME: ( -60, -27, -50, 0, -16, 0 ) )
    //
    // CHECK:    ( ( -7, -2, -39, 0, -30, -42 ),
    // CHECK-SAME: ( -4, -10, 0, -77, 0, -40 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 16, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 64 ),
    // CHECK-SAME: ( 0, 0, 0, -12, 0, -6 ),
    // CHECK-SAME: ( -60, -27, -50, 0, -16, 0 ) )
    //
    // CHECK:    ( ( -7, -2, -39, 0, -30, -42 ),
    // CHECK-SAME: ( -4, -10, 0, -77, 0, -40 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 16, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 64 ),
    // CHECK-SAME: ( 0, 0, 0, -12, 0, -6 ),
    // CHECK-SAME: ( -60, -27, -50, 0, -16, 0 ) )
    //
    %v0 = vector.transfer_read %0[%c0, %c0], %i0 : tensor<6x6xi32>, vector<6x6xi32>
    vector.print %v0 : vector<6x6xi32>
    %v1 = vector.transfer_read %1[%c0, %c0], %i0 : tensor<6x6xi32>, vector<6x6xi32>
    vector.print %v1 : vector<6x6xi32>
    %v2 = vector.transfer_read %2[%c0, %c0], %i0 : tensor<6x6xi32>, vector<6x6xi32>
    vector.print %v2 : vector<6x6xi32>
    %v3 = vector.transfer_read %3[%c0, %c0], %i0 : tensor<6x6xi32>, vector<6x6xi32>
    vector.print %v3 : vector<6x6xi32>
    %v4 = vector.transfer_read %4[%c0, %c0], %i0 : tensor<6x6xi32>, vector<6x6xi32>
    vector.print %v4 : vector<6x6xi32>
    %v5 = vector.transfer_read %5[%c0, %c0], %i0 : tensor<6x6xi32>, vector<6x6xi32>
    vector.print %v5 : vector<6x6xi32>

    // Release resources.
    bufferization.dealloc_tensor %input_CSR : tensor<10x10xi32, #CSR>
    bufferization.dealloc_tensor %input_DCSR : tensor<10x10xi32, #DCSR>
    bufferization.dealloc_tensor %filter_CSR : tensor<5x5xi32, #CSR>
    bufferization.dealloc_tensor %sinput_CSR : tensor<10x10xi32, #CSR>
    bufferization.dealloc_tensor %sinput_DCSR : tensor<10x10xi32, #DCSR>
    bufferization.dealloc_tensor %sfilter_CSR : tensor<5x5xi32, #CSR>
    bufferization.dealloc_tensor %0 : tensor<6x6xi32>
    bufferization.dealloc_tensor %1 : tensor<6x6xi32>
    bufferization.dealloc_tensor %2 : tensor<6x6xi32>
    bufferization.dealloc_tensor %3 : tensor<6x6xi32>
    bufferization.dealloc_tensor %4 : tensor<6x6xi32>
    bufferization.dealloc_tensor %5 : tensor<6x6xi32>

    return
  }
}
