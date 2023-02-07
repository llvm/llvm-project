// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = enable-runtime-library=false
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#DCSR = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>

// An example of a quantized sparse matmul. With the zero offset for the
// sparse input, the sparse compiler generates very efficient code for the
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

  func.func @entry() {
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

    return
  }
}
