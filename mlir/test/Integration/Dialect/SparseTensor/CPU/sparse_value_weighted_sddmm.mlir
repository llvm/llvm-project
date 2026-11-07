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
// DEFINE: %{run} = mlir-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs_sve}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

#trait_value_weighted_sddmm = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j)>,
    affine_map<(i,j,k) -> (i,k)>,
    affine_map<(i,j,k) -> (k,j)>,
    affine_map<(i,j,k) -> (i,j)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

module {
  func.func @value_weighted_sddmm(
      %sample: tensor<2x3xf64, #CSR>,
      %lhs: tensor<2x2xf64>,
      %rhs: tensor<2x3xf64>) -> tensor<2x3xf64, #CSR> {
    %empty = tensor.empty() : tensor<2x3xf64, #CSR>
    %result = linalg.generic #trait_value_weighted_sddmm
        ins(%sample, %lhs, %rhs : tensor<2x3xf64, #CSR>,
                                    tensor<2x2xf64>, tensor<2x3xf64>)
        outs(%empty : tensor<2x3xf64, #CSR>) {
      ^bb0(%s: f64, %a: f64, %b: f64, %acc: f64):
        %ab = arith.mulf %a, %b : f64
        %weighted = arith.mulf %s, %ab : f64
        %next = arith.addf %acc, %weighted : f64
        linalg.yield %next : f64
    } -> tensor<2x3xf64, #CSR>
    return %result : tensor<2x3xf64, #CSR>
  }

  func.func @main() {
    %dense_sample = arith.constant sparse<
        [[0, 0], [0, 2], [1, 1]], [2.0, 3.0, 4.0]> : tensor<2x3xf64>
    %sample = sparse_tensor.convert %dense_sample
        : tensor<2x3xf64> to tensor<2x3xf64, #CSR>
    %lhs = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]>
        : tensor<2x2xf64>
    %rhs = arith.constant dense<[[3.0, 6.0, 5.0], [9.0, 9.0, 11.0]]>
        : tensor<2x3xf64>

    %result = call @value_weighted_sddmm(%sample, %lhs, %rhs)
        : (tensor<2x3xf64, #CSR>, tensor<2x2xf64>, tensor<2x3xf64>)
          -> tensor<2x3xf64, #CSR>

    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 3
    // CHECK-NEXT: dim = ( 2, 3 )
    // CHECK-NEXT: lvl = ( 2, 3 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3 )
    // CHECK-NEXT: crd[1] : ( 0, 2, 1 )
    // CHECK-NEXT: values : ( 42, 81, 216 )
    // CHECK-NEXT: ----
    sparse_tensor.print %result : tensor<2x3xf64, #CSR>

    bufferization.dealloc_tensor %sample : tensor<2x3xf64, #CSR>
    bufferization.dealloc_tensor %result : tensor<2x3xf64, #CSR>
    return
  }
}
