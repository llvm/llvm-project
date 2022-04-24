// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
// Do the same run, but now with SIMDization as well. This should not change the outcome.
//
// RUN: mlir-opt %s --sparse-compiler="vectorization-strategy=2 vl=2" | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

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
    %m = bufferization.to_memref %0 : memref<5x6xi32>
    %v = vector.transfer_read %m[%c0, %c0], %i0
      : memref<5x6xi32>, vector<5x6xi32>
    vector.print %v : vector<5x6xi32>

    // Release the resources.
    sparse_tensor.release %sparse_input2 : tensor<3x6xi8, #DCSR>
    memref.dealloc %m : memref<5x6xi32>

    return
  }
}
