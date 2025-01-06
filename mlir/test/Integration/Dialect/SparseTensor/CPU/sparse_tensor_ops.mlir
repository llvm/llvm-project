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
// Do the same run, but now with vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#ST1 = #sparse_tensor.encoding<{map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : compressed)}>
#ST2 = #sparse_tensor.encoding<{map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : dense)}>

//
// Trait for 3-d tensor operation.
//
#trait_scale = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j,k)>,  // A (in)
    affine_map<(i,j,k) -> (i,j,k)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel", "parallel"],
  doc = "X(i,j,k) = A(i,j,k) * 2.0"
}

module {
  // Scales a sparse tensor into a new sparse tensor.
  func.func @tensor_scale(%arga: tensor<?x?x?xf64, #ST1>) -> tensor<?x?x?xf64, #ST2> {
    %s = arith.constant 2.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = tensor.dim %arga, %c0 : tensor<?x?x?xf64, #ST1>
    %d1 = tensor.dim %arga, %c1 : tensor<?x?x?xf64, #ST1>
    %d2 = tensor.dim %arga, %c2 : tensor<?x?x?xf64, #ST1>
    %xm = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf64, #ST2>
    %0 = linalg.generic #trait_scale
       ins(%arga: tensor<?x?x?xf64, #ST1>)
        outs(%xm: tensor<?x?x?xf64, #ST2>) {
        ^bb(%a: f64, %x: f64):
          %1 = arith.mulf %a, %s : f64
          linalg.yield %1 : f64
    } -> tensor<?x?x?xf64, #ST2>
    return %0 : tensor<?x?x?xf64, #ST2>
  }

  // Driver method to call and verify tensor kernel.
  func.func @main() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant -1.0 : f64

    // Setup sparse tensor.
    %t = arith.constant dense<
      [ [ [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0 ] ],
        [ [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ],
        [ [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
          [0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0 ],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] ] ]> : tensor<3x4x8xf64>
    %st = sparse_tensor.convert %t : tensor<3x4x8xf64> to tensor<?x?x?xf64, #ST1>

    // Call sparse vector kernels.
    %0 = call @tensor_scale(%st) : (tensor<?x?x?xf64, #ST1>) -> tensor<?x?x?xf64, #ST2>

    //
    // Sanity check on stored values.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 5
    // CHECK-NEXT: dim = ( 3, 4, 8 )
    // CHECK-NEXT: lvl = ( 3, 4, 8 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 0, 2 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3 )
    // CHECK-NEXT: crd[1] : ( 0, 3, 2 )
    // CHECK-NEXT: pos[2] : ( 0, 1, 2, 5 )
    // CHECK-NEXT: crd[2] : ( 0, 7, 1, 2, 7 )
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 24
    // CHECK-NEXT: dim = ( 3, 4, 8 )
    // CHECK-NEXT: lvl = ( 3, 4, 8 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 0, 2 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3 )
    // CHECK-NEXT: crd[1] : ( 0, 3, 2 )
    // CHECK-NEXT: values : ( 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 6, 8, 0, 0, 0, 0, 10 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %st : tensor<?x?x?xf64, #ST1>
    sparse_tensor.print %0  : tensor<?x?x?xf64, #ST2>

    // Release the resources.
    bufferization.dealloc_tensor %st : tensor<?x?x?xf64, #ST1>
    bufferization.dealloc_tensor %0  : tensor<?x?x?xf64, #ST2>
    return
  }
}
