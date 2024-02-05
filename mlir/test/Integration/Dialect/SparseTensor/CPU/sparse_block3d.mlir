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
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#Sparse1 = #sparse_tensor.encoding<{
  map = (i, j, k) -> (
    j : compressed,
    k : compressed,
    i : dense
  )
}>

#Sparse2 = #sparse_tensor.encoding<{
  map = (i, j, k) -> (
    i floordiv 2 : compressed,
    j floordiv 2 : compressed,
    k floordiv 2 : compressed,
    i mod 2 : dense,
    j mod 2 : dense,
    k mod 2 : dense)
}>

module {

  //
  // Main driver that tests sparse tensor storage.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %i0 = arith.constant 0 : i32

    // Setup input dense tensor and convert to two sparse tensors.
    %d = arith.constant dense <[
       [ // i=0
         [ 1, 0, 0, 0 ],
         [ 0, 0, 0, 0 ],
         [ 0, 0, 0, 0 ],
         [ 0, 0, 5, 0 ] ],
       [ // i=1
         [ 2, 0, 0, 0 ],
         [ 0, 0, 0, 0 ],
         [ 0, 0, 0, 0 ],
         [ 0, 0, 6, 0 ] ],
       [ //i=2
         [ 3, 0, 0, 0 ],
         [ 0, 0, 0, 0 ],
         [ 0, 0, 0, 0 ],
         [ 0, 0, 7, 0 ] ],
	 //i=3
       [ [ 4, 0, 0, 0 ],
         [ 0, 0, 0, 0 ],
         [ 0, 0, 0, 0 ],
         [ 0, 0, 8, 0 ] ]
    ]> : tensor<4x4x4xi32>

    %a = sparse_tensor.convert %d : tensor<4x4x4xi32> to tensor<4x4x4xi32, #Sparse1>
    %b = sparse_tensor.convert %d : tensor<4x4x4xi32> to tensor<4x4x4xi32, #Sparse2>

    //
    // If we store the two "fibers" [1,2,3,4] starting at index (0,0,0) and
    // ending at index (3,0,0) and [5,6,7,8] starting at index (0,3,2) and
    // ending at index (3,3,2)) with a “DCSR-flavored” along (j,k) with
    // dense “fibers” in the i-dim, we end up with 8 stored entries.
    //
    // CHECK: 8
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8 )
    //
    %na = sparse_tensor.number_of_entries %a : tensor<4x4x4xi32, #Sparse1>
    vector.print %na : index
    %ma = sparse_tensor.values %a: tensor<4x4x4xi32, #Sparse1> to memref<?xi32>
    %va = vector.transfer_read %ma[%c0], %i0: memref<?xi32>, vector<8xi32>
    vector.print %va : vector<8xi32>

    //
    // If we store full 2x2x2 3-D blocks in the original index order
    // in a compressed fashion, we end up with 4 blocks to incorporate
    // all the nonzeros, and thus 32 stored entries.
    //
    // CHECK: 32
    // CHECK-NEXT: ( 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 5, 0, 0, 0, 6, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 7, 0, 0, 0, 8, 0 )
    //
    %nb = sparse_tensor.number_of_entries %b : tensor<4x4x4xi32, #Sparse2>
    vector.print %nb : index
    %mb = sparse_tensor.values %b: tensor<4x4x4xi32, #Sparse2> to memref<?xi32>
    %vb = vector.transfer_read %mb[%c0], %i0: memref<?xi32>, vector<32xi32>
    vector.print %vb : vector<32xi32>

    // Release the resources.
    bufferization.dealloc_tensor %a : tensor<4x4x4xi32, #Sparse1>
    bufferization.dealloc_tensor %b : tensor<4x4x4xi32, #Sparse2>

    return
  }
}
