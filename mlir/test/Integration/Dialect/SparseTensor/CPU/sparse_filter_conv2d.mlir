// RUN: mlir-opt %s \
// RUN:   --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
// RUN:   --sparsification --sparse-tensor-conversion \
// RUN:   --convert-vector-to-scf --convert-scf-to-std \
// RUN:   --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN:   --std-bufferize --finalizing-bufferize --lower-affine \
// RUN:   --convert-vector-to-llvm --convert-memref-to-llvm \
// RUN:   --convert-std-to-llvm --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
// Do the same run, but now with SIMDization as well. This should not change the outcome.
//
// RUN: mlir-opt %s \
// RUN:   --linalg-generalize-named-ops --linalg-fuse-elementwise-ops \
// RUN:   --sparsification="vectorization-strategy=2 vl=2" --sparse-tensor-conversion \
// RUN:   --convert-vector-to-scf --convert-scf-to-std \
// RUN:   --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN:   --std-bufferize --finalizing-bufferize --lower-affine \
// RUN:   --convert-vector-to-llvm --convert-memref-to-llvm \
// RUN:   --convert-std-to-llvm --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#DCSR = #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>

// An example of a 2D convolution with a sparse filter.
module {

  func @conv2d(%input:  tensor<8x8xi32>,
               %filter: tensor<3x3xi32, #DCSR>,
               %output: tensor<6x6xi32>) -> tensor<6x6xi32> {
    %0 = linalg.conv_2d
      ins  (%input, %filter: tensor<8x8xi32>, tensor<3x3xi32, #DCSR>)
      outs (%output: tensor<6x6xi32>) -> tensor<6x6xi32>
    return %0 : tensor<6x6xi32>
  }

  func @entry() {
    %c0 = constant 0 : index
    %i0 = constant 0 : i32

    // A typical edge detection filter.
    %filter = constant dense<[
      [  1,  0, -1 ],
      [  0,  0,  0 ],
      [ -1,  0,  1 ]
    ]> : tensor<3x3xi32>
    %sparse_filter = sparse_tensor.convert %filter
      : tensor<3x3xi32> to tensor<3x3xi32, #DCSR>

    %input = constant dense<[
      [  1,  2,  3,  4,  0,  6,  7,  8 ],
      [  2,  2,  4,  4,  0,  0,  6,  8 ],
      [  2,  2,  4,  4,  0,  0,  6,  8 ],
      [  2,  2,  3,  4,  0,  0,  7,  8 ],
      [  1,  3,  3,  4,  0,  0,  6,  8 ],
      [  3,  2,  3,  4,  0,  0,  7,  8 ],
      [  1,  3,  3,  4,  3,  6,  6,  8 ],
      [  1,  3,  3,  4,  3,  0,  7,  8 ]
    ]> : tensor<8x8xi32>

    // Call the kernel.
    %output = constant dense<0> : tensor<6x6xi32>
    %0 = call @conv2d(%input, %sparse_filter, %output)
       : (tensor<8x8xi32>,
          tensor<3x3xi32, #DCSR>, tensor<6x6xi32>) -> tensor<6x6xi32>

    // Verify the output.
    //
    // CHECK:    ( ( 0, 0, -1, -6, -1, 6 ),
    // CHECK-SAME: ( -1, 0, 1, 0, 1, 0 ),
    // CHECK-SAME: ( 0, -1, 1, 0, 0, 0 ),
    // CHECK-SAME: ( -1, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 3, 6, -3, -6 ),
    // CHECK-SAME: ( 2, -1, 3, 0, -3, 0 ) )
    //
    %m = memref.buffer_cast %0 : memref<6x6xi32>
    %v = vector.transfer_read %m[%c0, %c0], %i0
      : memref<6x6xi32>, vector<6x6xi32>
    vector.print %v : vector<6x6xi32>

    // Release the resources.
    sparse_tensor.release %sparse_filter : tensor<3x3xi32, #DCSR>
    memref.dealloc %m : memref<6x6xi32>

    return
  }
}
