// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#SparseVector = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>

module {

  //
  // Sparse kernel.
  //
  func.func @sparse_dot(%a: tensor<1024xf32, #SparseVector>,
                   %b: tensor<1024xf32, #SparseVector>) -> tensor<f32> {
    %x = linalg.init_tensor [] : tensor<f32>
    %dot = linalg.dot ins(%a, %b: tensor<1024xf32, #SparseVector>,
                                  tensor<1024xf32, #SparseVector>)
         outs(%x: tensor<f32>) -> tensor<f32>
    return %dot : tensor<f32>
  }

  //
  // Main driver.
  //
  func.func @entry() {
    // Setup two sparse vectors.
    %d1 = arith.constant sparse<
        [ [0], [1], [22], [23], [1022] ], [1.0, 2.0, 3.0, 4.0, 5.0]
    > : tensor<1024xf32>
    %d2 = arith.constant sparse<
      [ [22], [1022], [1023] ], [6.0, 7.0, 8.0]
    > : tensor<1024xf32>
    %s1 = sparse_tensor.convert %d1 : tensor<1024xf32> to tensor<1024xf32, #SparseVector>
    %s2 = sparse_tensor.convert %d2 : tensor<1024xf32> to tensor<1024xf32, #SparseVector>

    // Call the kernel and verify the output.
    //
    // CHECK: 53
    //
    %0 = call @sparse_dot(%s1, %s2) : (tensor<1024xf32, #SparseVector>,
                                       tensor<1024xf32, #SparseVector>) -> tensor<f32>
    %1 = tensor.extract %0[] : tensor<f32>
    vector.print %1 : f32

    // Release the resources.
    sparse_tensor.release %s1 : tensor<1024xf32, #SparseVector>
    sparse_tensor.release %s2 : tensor<1024xf32, #SparseVector>
    %m = bufferization.to_memref %0 : memref<f32>
    memref.dealloc %m : memref<f32>

    return
  }
}
