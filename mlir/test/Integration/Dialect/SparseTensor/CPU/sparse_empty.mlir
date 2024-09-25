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
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}


#map = affine_map<(d0) -> (d0)>

#SV  = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

module {

  // This directly yields an empty sparse vector.
  func.func @empty() -> tensor<10xf32, #SV> {
    %0 = tensor.empty() : tensor<10xf32, #SV>
    return %0 : tensor<10xf32, #SV>
  }

  // This also directly yields an empty sparse vector.
  func.func @empty_alloc() -> tensor<10xf32, #SV> {
    %0 = bufferization.alloc_tensor() : tensor<10xf32, #SV>
    return %0 : tensor<10xf32, #SV>
  }

  // This yields a hidden empty sparse vector (all zeros).
  func.func @zeros() -> tensor<10xf32, #SV> {
    %cst = arith.constant 0.0 : f32
    %0 = bufferization.alloc_tensor() : tensor<10xf32, #SV>
    %1 = linalg.generic {
        indexing_maps = [#map],
	iterator_types = ["parallel"]}
      outs(%0 : tensor<10xf32, #SV>) {
         ^bb0(%out: f32):
            linalg.yield %cst : f32
    } -> tensor<10xf32, #SV>
    return %1 : tensor<10xf32, #SV>
  }

  // This yields a filled sparse vector (all ones).
  func.func @ones() -> tensor<10xf32, #SV> {
    %cst = arith.constant 1.0 : f32
    %0 = bufferization.alloc_tensor() : tensor<10xf32, #SV>
    %1 = linalg.generic {
        indexing_maps = [#map],
	iterator_types = ["parallel"]}
      outs(%0 : tensor<10xf32, #SV>) {
         ^bb0(%out: f32):
            linalg.yield %cst : f32
    } -> tensor<10xf32, #SV>
    return %1 : tensor<10xf32, #SV>
  }

  //
  // Main driver.
  //
  func.func @main() {

    %0 = call @empty()       : () -> tensor<10xf32, #SV>
    %1 = call @empty_alloc() : () -> tensor<10xf32, #SV>
    %2 = call @zeros()       : () -> tensor<10xf32, #SV>
    %3 = call @ones()        : () -> tensor<10xf32, #SV>

    //
    // Verify the output. In particular, make sure that
    // all empty sparse vector data structures are properly
    // finalized with a pair (0,0) for positions.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 0
    // CHECK-NEXT: dim = ( 10 )
    // CHECK-NEXT: lvl = ( 10 )
    // CHECK-NEXT: pos[0] : ( 0, 0 )
    // CHECK-NEXT: crd[0] : ( )
    // CHECK-NEXT: values : ( )
    // CHECK-NEXT: ----
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 0
    // CHECK-NEXT: dim = ( 10 )
    // CHECK-NEXT: lvl = ( 10 )
    // CHECK-NEXT: pos[0] : ( 0, 0 )
    // CHECK-NEXT: crd[0] : ( )
    // CHECK-NEXT: values : ( )
    // CHECK-NEXT: ----
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 0
    // CHECK-NEXT: dim = ( 10 )
    // CHECK-NEXT: lvl = ( 10 )
    // CHECK-NEXT: pos[0] : ( 0, 0 )
    // CHECK-NEXT: crd[0] : ( )
    // CHECK-NEXT: values : ( )
    // CHECK-NEXT: ----
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 10
    // CHECK-NEXT: dim = ( 10 )
    // CHECK-NEXT: lvl = ( 10 )
    // CHECK-NEXT: pos[0] : ( 0, 10 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 )
    // CHECK-NEXT: values : ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %0 : tensor<10xf32, #SV>
    sparse_tensor.print %1 : tensor<10xf32, #SV>
    sparse_tensor.print %2 : tensor<10xf32, #SV>
    sparse_tensor.print %3 : tensor<10xf32, #SV>

    bufferization.dealloc_tensor %0 : tensor<10xf32, #SV>
    bufferization.dealloc_tensor %1 : tensor<10xf32, #SV>
    bufferization.dealloc_tensor %2 : tensor<10xf32, #SV>
    bufferization.dealloc_tensor %3 : tensor<10xf32, #SV>
    return
  }
}
