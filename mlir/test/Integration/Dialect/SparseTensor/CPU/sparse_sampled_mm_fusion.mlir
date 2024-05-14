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

#SM = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>

#trait_sampled_dense_dense = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j)>,  // S
    affine_map<(i,j,k) -> (i,k)>,  // A
    affine_map<(i,j,k) -> (k,j)>,  // B
    affine_map<(i,j,k) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) += S(i,j) SUM_k A(i,k) B(k,j)"
}

#trait_matmul = {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d1, d0)>,
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d1, d2)>
  ],
  iterator_types = ["reduction", "parallel", "parallel"]
}

#trait_scale = {
  indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>
  ],
  iterator_types = ["parallel", "parallel"]
}

//
// Integration test for sampled dense dense matmul fusion.
//
module {
  //
  // A kernel that computes a direct sampled matrix matrix multiplication
  // (with dense result).
  //
  func.func @sampled_dd(%args: tensor<8x8xf64, #SM>,
                        %arga: tensor<8x8xf64>,
                        %argb: tensor<8x8xf64>) -> tensor<8x8xf64> {
    %1 = arith.constant dense<0.0> : tensor<8x8xf64>
    %2 = linalg.generic #trait_sampled_dense_dense
      ins(%args, %arga, %argb: tensor<8x8xf64, #SM>,
                               tensor<8x8xf64>, tensor<8x8xf64>)
      outs(%1: tensor<8x8xf64>) {
        ^bb(%s: f64, %a: f64, %b: f64, %x: f64):
          %p = arith.mulf %a, %b : f64
          %q = arith.mulf %s, %p : f64
          %r = arith.addf %x, %q : f64
          linalg.yield %r : f64
    } -> tensor<8x8xf64>
    return %2 : tensor<8x8xf64>
  }

  //
  // A kernel that computes an unfused sampled matrix matrix multiplication
  // (with dense result).
  //
  func.func @sampled_dd_unfused(%args: tensor<8x8xf64, #SM>,
                                %arga: tensor<8x8xf64>,
                                %argb: tensor<8x8xf64>) -> tensor<8x8xf64> {
    // Perform dense-dense matrix matrix multiplication.
    %1 = arith.constant dense<0.0> : tensor<8x8xf64>
    %2 = linalg.generic #trait_matmul
      ins(%arga, %argb : tensor<8x8xf64>, tensor<8x8xf64>)
      outs(%1 : tensor<8x8xf64>) {
        ^bb0(%a: f64, %b: f64, %x: f64):
          %p = arith.mulf %a, %b : f64
          %q = arith.addf %x, %p : f64
          linalg.yield %q : f64
    } -> tensor<8x8xf64>
    // Sample the result with elements-wise multiplication with sparse matrix.
    %3 = linalg.generic #trait_scale
      ins(%2, %args : tensor<8x8xf64>, tensor<8x8xf64, #SM>)
      outs(%1 : tensor<8x8xf64>) {
        ^bb0(%t: f64, %s: f64, %x: f64):
          %r = arith.mulf %t, %s : f64
          linalg.yield %r : f64
    } -> tensor<8x8xf64>
    bufferization.dealloc_tensor %2 : tensor<8x8xf64>
    return %3 : tensor<8x8xf64>
  }

  //
  // A kernel that computes a direct sampled matrix matrix multiplication
  // (with sparse result).
  //
  func.func @sparse_sampled_dd(%args: tensor<8x8xf64, #SM>,
                               %arga: tensor<8x8xf64>,
                               %argb: tensor<8x8xf64>) -> tensor<8x8xf64, #SM> {
    %1 = tensor.empty() : tensor<8x8xf64, #SM>
    %2 = linalg.generic #trait_sampled_dense_dense
      ins(%args, %arga, %argb: tensor<8x8xf64, #SM>,
                               tensor<8x8xf64>, tensor<8x8xf64>)
      outs(%1: tensor<8x8xf64, #SM>) {
        ^bb(%s: f64, %a: f64, %b: f64, %x: f64):
          %p = arith.mulf %a, %b : f64
          %q = arith.mulf %s, %p : f64
          %r = arith.addf %x, %q : f64
          linalg.yield %r : f64
    } -> tensor<8x8xf64, #SM>
    return %2 : tensor<8x8xf64, #SM>
  }

  //
  // A kernel that computes an unfused sampled matrix matrix multiplication
  // (with sparse result).
  //
  func.func @sparse_sampled_dd_unfused(
        %args: tensor<8x8xf64, #SM>,
        %arga: tensor<8x8xf64>,
        %argb: tensor<8x8xf64>) -> tensor<8x8xf64, #SM> {
    // Perform dense-dense matrix matrix multiplication.
    %1 = arith.constant dense<0.0> : tensor<8x8xf64>
    %2 = linalg.generic #trait_matmul
      ins(%arga, %argb : tensor<8x8xf64>, tensor<8x8xf64>)
      outs(%1 : tensor<8x8xf64>) {
        ^bb0(%a: f64, %b: f64, %x: f64):
          %p = arith.mulf %a, %b : f64
          %q = arith.addf %x, %p : f64
          linalg.yield %q : f64
    } -> tensor<8x8xf64>
    // Sample the result with elements-wise multiplication with sparse matrix.
    %3 = tensor.empty() : tensor<8x8xf64, #SM>
    %4 = linalg.generic #trait_scale
      ins(%2, %args : tensor<8x8xf64>, tensor<8x8xf64, #SM>)
      outs(%3 : tensor<8x8xf64, #SM>) {
        ^bb0(%t: f64, %s: f64, %x: f64):
          %r = arith.mulf %t, %s : f64
          linalg.yield %r : f64
    } -> tensor<8x8xf64, #SM>
    return %4 : tensor<8x8xf64, #SM>
  }

  //
  // Main driver.
  //
  func.func @main() {
    %d0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index

    %t = arith.constant sparse<[[0, 0], [7,7]], [1.0, 2.0]>
       : tensor<8x8xf64>
    %s = sparse_tensor.convert %t
       : tensor<8x8xf64> to tensor<8x8xf64, #SM>

    %a = arith.constant dense<3.0> : tensor<8x8xf64>
    %b = arith.constant dense<4.0> : tensor<8x8xf64>

    // Call the kernels.
    %0 = call @sampled_dd(%s, %a, %b)
      : (tensor<8x8xf64, #SM>,
         tensor<8x8xf64>, tensor<8x8xf64>) -> tensor<8x8xf64>
    %1 = call @sampled_dd_unfused(%s, %a, %b)
      : (tensor<8x8xf64, #SM>,
         tensor<8x8xf64>, tensor<8x8xf64>) -> tensor<8x8xf64>
    %2 = call @sparse_sampled_dd(%s, %a, %b)
      : (tensor<8x8xf64, #SM>,
         tensor<8x8xf64>, tensor<8x8xf64>) -> tensor<8x8xf64, #SM>
    %3 = call @sparse_sampled_dd_unfused(%s, %a, %b)
      : (tensor<8x8xf64, #SM>,
         tensor<8x8xf64>, tensor<8x8xf64>) -> tensor<8x8xf64, #SM>

    // Verify the outputs.
    //
    // CHECK:    ( ( 96, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 192 ) )
    //
    // CHECK:    ( ( 96, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 0 ),
    // CHECK-SAME: ( 0, 0, 0, 0, 0, 0, 0, 0 ), ( 0, 0, 0, 0, 0, 0, 0, 192 ) )
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 2
    // CHECK-NEXT: dim = ( 8, 8 )
    // CHECK-NEXT: lvl = ( 8, 8 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 0, 7 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 2 )
    // CHECK-NEXT: crd[1] : ( 0, 7 )
    // CHECK-NEXT: values : ( 96, 192 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 2
    // CHECK-NEXT: dim = ( 8, 8 )
    // CHECK-NEXT: lvl = ( 8, 8 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 0, 7 )
    // CHECK-NEXT: pos[1] : ( 0, 1, 2 )
    // CHECK-NEXT: crd[1] : ( 0, 7 )
    // CHECK-NEXT: values : ( 96, 192 )
    // CHECK-NEXT: ----
    //
    %v0 = vector.transfer_read %0[%c0, %c0], %d0
        : tensor<8x8xf64>, vector<8x8xf64>
    %v1 = vector.transfer_read %1[%c0, %c0], %d0
        : tensor<8x8xf64>, vector<8x8xf64>
    vector.print %v0 : vector<8x8xf64>
    vector.print %v1 : vector<8x8xf64>
    sparse_tensor.print %2 : tensor<8x8xf64, #SM>
    sparse_tensor.print %3 : tensor<8x8xf64, #SM>

    // Release the resources.
    bufferization.dealloc_tensor %s : tensor<8x8xf64, #SM>
    bufferization.dealloc_tensor %0 : tensor<8x8xf64>
    bufferization.dealloc_tensor %1 : tensor<8x8xf64>
    bufferization.dealloc_tensor %2 : tensor<8x8xf64, #SM>
    bufferization.dealloc_tensor %3 : tensor<8x8xf64, #SM>

    return
  }
}
