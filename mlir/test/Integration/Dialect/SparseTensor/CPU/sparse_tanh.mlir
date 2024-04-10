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
// Do the same run, but now with vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

// Current fails for SVE, see https://github.com/llvm/llvm-project/issues/60626
// UNSUPPORTED: target=aarch64{{.*}}

#SparseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

#trait_op = {
  indexing_maps = [
    affine_map<(i) -> (i)>   // X (out)
  ],
  iterator_types = ["parallel"],
  doc = "X(i) = OP X(i)"
}

module {
  // Performs zero-preserving math to sparse vector.
  func.func @sparse_tanh(%vec: tensor<?xf64, #SparseVector>)
                       -> tensor<?xf64, #SparseVector> {
    %0 = linalg.generic #trait_op
      outs(%vec: tensor<?xf64, #SparseVector>) {
        ^bb(%x: f64):
          %1 = math.tanh %x : f64
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Driver method to call and verify vector kernels.
  func.func @main() {
    // Setup sparse vector.
    %v1 = arith.constant sparse<
       [ [0], [3], [11], [17], [20], [21], [28], [29], [31] ],
         [ -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0 ]
    > : tensor<32xf64>
    %sv1 = sparse_tensor.convert %v1
         : tensor<32xf64> to tensor<?xf64, #SparseVector>

    // Call sparse vector kernel.
    %0 = call @sparse_tanh(%sv1) : (tensor<?xf64, #SparseVector>)
                                 -> tensor<?xf64, #SparseVector>

    //
    // Verify the results (within some precision).
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 9
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 9
    // CHECK-NEXT: crd[0] : ( 0, 3, 11, 17, 20, 21, 28, 29, 31
    // CHECK-NEXT: values : ({{ -0.761[0-9]*, 0.761[0-9]*, 0.96[0-9]*, 0.99[0-9]*, 0.99[0-9]*, 0.99[0-9]*, 0.99[0-9]*, 0.99[0-9]*, 1}}
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %0 : tensor<?xf64, #SparseVector>

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xf64, #SparseVector>
    return
  }
}
