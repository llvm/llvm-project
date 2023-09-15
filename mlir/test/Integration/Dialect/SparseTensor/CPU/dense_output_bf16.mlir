//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparse_compiler_opts} = enable-runtime-library=true
// DEFINE: %{sparse_compiler_opts_sve} = enable-arm-sve=true %{sparse_compiler_opts}
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts_sve}"
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
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s

// UNSUPPORTED: target=aarch64{{.*}}

#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>
#DenseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : dense)}>

#trait_vec_op = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>,  // b (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"]
}

module {
  // Creates a dense vector using the minimum values from two input sparse vectors.
  // When there is no overlap, include the present value in the output.
  func.func @vector_min(%arga: tensor<?xbf16, #SparseVector>,
                        %argb: tensor<?xbf16, #SparseVector>) -> tensor<?xbf16, #DenseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xbf16, #SparseVector>
    %xv = tensor.empty (%d) : tensor<?xbf16, #DenseVector>
    %0 = linalg.generic #trait_vec_op
       ins(%arga, %argb: tensor<?xbf16, #SparseVector>, tensor<?xbf16, #SparseVector>)
        outs(%xv: tensor<?xbf16, #DenseVector>) {
        ^bb(%a: bf16, %b: bf16, %x: bf16):
          %1 = sparse_tensor.binary %a, %b : bf16, bf16 to bf16
            overlap={
              ^bb0(%a0: bf16, %b0: bf16):
                %cmp = arith.cmpf "olt", %a0, %b0 : bf16
                %2 = arith.select %cmp, %a0, %b0: bf16
                sparse_tensor.yield %2 : bf16
            }
            left=identity
            right=identity
          linalg.yield %1 : bf16
    } -> tensor<?xbf16, #DenseVector>
    return %0 : tensor<?xbf16, #DenseVector>
  }

  // Dumps a dense vector of type bf16.
  func.func @dump_vec(%arg0: tensor<?xbf16, #DenseVector>) {
    // Dump the values array to verify only sparse contents are stored.
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : bf16
    %0 = sparse_tensor.values %arg0 : tensor<?xbf16, #DenseVector> to memref<?xbf16>
    %1 = vector.transfer_read %0[%c0], %d0: memref<?xbf16>, vector<32xbf16>
    %f1 = arith.extf %1: vector<32xbf16> to vector<32xf32>
    vector.print %f1 : vector<32xf32>
    return
  }

  // Driver method to call and verify the kernel.
  func.func @entry() {
    %c0 = arith.constant 0 : index

    // Setup sparse vectors.
    %v1 = arith.constant sparse<
       [ [0], [3], [11], [17], [20], [21], [28], [29], [31] ],
         [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    > : tensor<32xbf16>
    %v2 = arith.constant sparse<
       [ [1], [3], [4], [10], [16], [18], [21], [28], [29], [31] ],
         [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0 ]
    > : tensor<32xbf16>
    %sv1 = sparse_tensor.convert %v1 : tensor<32xbf16> to tensor<?xbf16, #SparseVector>
    %sv2 = sparse_tensor.convert %v2 : tensor<32xbf16> to tensor<?xbf16, #SparseVector>

    // Call the sparse vector kernel.
    %0 = call @vector_min(%sv1, %sv2)
       : (tensor<?xbf16, #SparseVector>,
          tensor<?xbf16, #SparseVector>) -> tensor<?xbf16, #DenseVector>

    //
    // Verify the result.
    //
    // CHECK: ( 1, 11, 0, 2, 13, 0, 0, 0, 0, 0, 14, 3, 0, 0, 0, 0, 15, 4, 16, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 9 )
    call @dump_vec(%0) : (tensor<?xbf16, #DenseVector>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xbf16, #SparseVector>
    bufferization.dealloc_tensor %sv2 : tensor<?xbf16, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<?xbf16, #DenseVector>
    return
  }
}
