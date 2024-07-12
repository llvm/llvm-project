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
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#DCSR  = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

module {
  //
  // Main driver.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %t1 = tensor.empty() : tensor<4x5xf64, #DCSR>
    %t2 = tensor.empty(%c2, %c3) : tensor<?x?xf64, #DCSR>

    %d1_0 = tensor.dim %t1, %c0 : tensor<4x5xf64, #DCSR>
    %d2_0 = tensor.dim %t2, %c0 : tensor<?x?xf64, #DCSR>
    %d1_1 = tensor.dim %t1, %c1 : tensor<4x5xf64, #DCSR>
    %d2_1 = tensor.dim %t2, %c1 : tensor<?x?xf64, #DCSR>

    // CHECK: 4
    vector.print %d1_0 : index
    // CHECK-NEXT: 2
    vector.print %d2_0 : index
    // CHECK-NEXT: 5
    vector.print %d1_1 : index
    // CHECK-NEXT: 3
    vector.print %d2_1 : index

    // Release resources.
    bufferization.dealloc_tensor %t1 : tensor<4x5xf64, #DCSR>
    bufferization.dealloc_tensor %t2 : tensor<?x?xf64, #DCSR>

    return
  }
}

