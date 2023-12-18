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
// DEFINE: %{run_opts} = -e entry -entry-point-result=void
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

#SparseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

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
    %t = tensor.empty() : tensor<f32>
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
