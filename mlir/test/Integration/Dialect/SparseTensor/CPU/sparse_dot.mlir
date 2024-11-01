// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
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

#SparseVector = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>

module {

  //
  // Sparse kernel.
  //
  func.func @sparse_dot(%a: tensor<1024xf32, #SparseVector>,
                        %b: tensor<1024xf32, #SparseVector>,
                        %x: tensor<f32>) -> tensor<f32> {
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
    %t = bufferization.alloc_tensor() : tensor<f32>
    %z = arith.constant 0.0 : f32
    %x = tensor.insert %z into %t[] : tensor<f32>
    %0 = call @sparse_dot(%s1, %s2, %x) : (tensor<1024xf32, #SparseVector>,
                                           tensor<1024xf32, #SparseVector>,
                                           tensor<f32>) -> tensor<f32>
    %1 = tensor.extract %0[] : tensor<f32>
    vector.print %1 : f32

    // Print number of entries in the sparse vectors.
    //
    // CHECK: 5
    // CHECK: 3
    //
    %noe1 = sparse_tensor.number_of_entries %s1 : tensor<1024xf32, #SparseVector>
    %noe2 = sparse_tensor.number_of_entries %s2 : tensor<1024xf32, #SparseVector>
    vector.print %noe1 : index
    vector.print %noe2 : index

    // Release the resources.
    bufferization.dealloc_tensor %s1 : tensor<1024xf32, #SparseVector>
    bufferization.dealloc_tensor %s2 : tensor<1024xf32, #SparseVector>

    return
  }
}
