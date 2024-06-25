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

// An example of a quantized sparse matmul. With the zero offset for the
// sparse input, the sparsifier generates very efficient code for the
//      x(i,j) += (ext(a(i,k)) - 2) * ext(b(k,j))
// operation.
module {

  func.func @quantized_matmul(%input1: tensor<5x3xi8>,
                         %input2: tensor<3x6xi8, #DCSR>,
                         %output: tensor<5x6xi32>) -> tensor<5x6xi32> {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %0 = linalg.quantized_matmul
      ins(%input1, %input2, %c2, %c0 : tensor<5x3xi8>, tensor<3x6xi8, #DCSR>, i32, i32)
      outs(%output : tensor<5x6xi32>) -> tensor<5x6xi32>
    return %0: tensor<5x6xi32>
  }

  func.func @main() {
    %c0 = arith.constant 0 : index
    %i0 = arith.constant 0 : i32

    %input1 = arith.constant dense<[
      [  -128,   3,  127 ],
      [     0,   0,    0 ],
      [    11,   1,    0 ],
      [     0,   5,   -1 ],
      [    13,   0,    3 ]
    ]> : tensor<5x3xi8>

    %input2 = arith.constant dense<[
      [  127,   0, -128,    0,   0,   3 ],
      [    0,   0,    0,    0,   0,   0 ],
      [    0,   0,    0,  100,  10,   0 ]
    ]> : tensor<3x6xi8>

    %sparse_input2 = sparse_tensor.convert %input2 : tensor<3x6xi8> to tensor<3x6xi8, #DCSR>

    // Call the kernel.
    %output = arith.constant dense<0> : tensor<5x6xi32>
    %0 = call @quantized_matmul(%input1, %sparse_input2, %output)
       : (tensor<5x3xi8>,
          tensor<3x6xi8, #DCSR>,
          tensor<5x6xi32>) -> tensor<5x6xi32>

    //
    // Verify the output.
    //
    // CHECK:    ( ( -16510, 0, 16640, 12500, 1250, -390 ),
    // CHECK-SAME: ( -254, 0, 256, -200, -20, -6 ),
    // CHECK-SAME: ( 1143, 0, -1152, -200, -20, 27 ),
    // CHECK-SAME: ( -254, 0, 256, -300, -30, -6 ),
    // CHECK-SAME: ( 1397, 0, -1408, 100, 10, 33 ) )
    //
    %v = vector.transfer_read %0[%c0, %c0], %i0
      : tensor<5x6xi32>, vector<5x6xi32>
    vector.print %v : vector<5x6xi32>

    // Release the resources.
    bufferization.dealloc_tensor %sparse_input2 : tensor<3x6xi8, #DCSR>
    bufferization.dealloc_tensor %0 : tensor<5x6xi32>

    return
  }
}
