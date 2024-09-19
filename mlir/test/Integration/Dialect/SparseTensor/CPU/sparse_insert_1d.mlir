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
// DEFINE: %{run_libs_sve} = -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils
// DEFINE: %{run_opts} = -e main -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs_sve}
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

#trait_mul_s = {
  indexing_maps = [
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = x(i) * 2.0"
}

module {
  func.func @main() {
    %f1    = arith.constant 1.0 : f32
    %f2    = arith.constant 2.0 : f32
    %f3    = arith.constant 3.0 : f32
    %f4    = arith.constant 4.0 : f32
    %c0    = arith.constant 0 : index
    %c1    = arith.constant 1 : index
    %c3    = arith.constant 3 : index
    %c8    = arith.constant 8 : index
    %c1023 = arith.constant 1023 : index

    // Build the sparse vector from straightline code.
    %0 = tensor.empty() : tensor<1024xf32, #SparseVector>
    %1 = tensor.insert %f1 into %0[%c0] : tensor<1024xf32, #SparseVector>
    %2 = tensor.insert %f2 into %1[%c1] : tensor<1024xf32, #SparseVector>
    %3 = tensor.insert %f3 into %2[%c3] : tensor<1024xf32, #SparseVector>
    %4 = tensor.insert %f4 into %3[%c1023] : tensor<1024xf32, #SparseVector>
    %5 = sparse_tensor.load %4 hasInserts : tensor<1024xf32, #SparseVector>

    //
    // CHECK:   ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 4
    // CHECK-NEXT: dim = ( 1024 )
    // CHECK-NEXT: lvl = ( 1024 )
    // CHECK-NEXT: pos[0] : ( 0, 4 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 3, 1023 )
    // CHECK-NEXT: values : ( 1, 2, 3, 4 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %5 : tensor<1024xf32, #SparseVector>

    // Build another sparse vector in a loop.
    %6 = tensor.empty() : tensor<1024xf32, #SparseVector>
    %7 = scf.for %i = %c0 to %c8 step %c1 iter_args(%vin = %6) -> tensor<1024xf32, #SparseVector> {
      %ii = arith.muli %i, %c3 : index
      %vout = tensor.insert %f1 into %vin[%ii] : tensor<1024xf32, #SparseVector>
      scf.yield %vout : tensor<1024xf32, #SparseVector>
    }
    %8 = sparse_tensor.load %7 hasInserts : tensor<1024xf32, #SparseVector>

    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 8
    // CHECK-NEXT: dim = ( 1024 )
    // CHECK-NEXT: lvl = ( 1024 )
    // CHECK-NEXT: pos[0] : ( 0, 8 )
    // CHECK-NEXT: crd[0] : ( 0, 3, 6, 9, 12, 15, 18, 21 )
    // CHECK-NEXT: values : ( 1, 1, 1, 1, 1, 1, 1, 1 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %8 : tensor<1024xf32, #SparseVector>

    // Free resources.
    bufferization.dealloc_tensor %5 : tensor<1024xf32, #SparseVector>
    bufferization.dealloc_tensor %8 : tensor<1024xf32, #SparseVector>
    return
  }
}
