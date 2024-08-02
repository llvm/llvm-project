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

#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>

#trait_op = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>,  // b (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) OP b(i)"
}

module {
  func.func @cadd(%arga: tensor<?xcomplex<f32>, #SparseVector>,
                  %argb: tensor<?xcomplex<f32>, #SparseVector>)
                      -> tensor<?xcomplex<f32>, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xcomplex<f32>, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xcomplex<f32>, #SparseVector>
    %0 = linalg.generic #trait_op
       ins(%arga, %argb: tensor<?xcomplex<f32>, #SparseVector>,
                         tensor<?xcomplex<f32>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f32>, #SparseVector>) {
        ^bb(%a: complex<f32>, %b: complex<f32>, %x: complex<f32>):
          %1 = complex.add %a, %b : complex<f32>
          linalg.yield %1 : complex<f32>
    } -> tensor<?xcomplex<f32>, #SparseVector>
    return %0 : tensor<?xcomplex<f32>, #SparseVector>
  }

  func.func @cmul(%arga: tensor<?xcomplex<f32>, #SparseVector>,
                  %argb: tensor<?xcomplex<f32>, #SparseVector>)
                      -> tensor<?xcomplex<f32>, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xcomplex<f32>, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xcomplex<f32>, #SparseVector>
    %0 = linalg.generic #trait_op
       ins(%arga, %argb: tensor<?xcomplex<f32>, #SparseVector>,
                         tensor<?xcomplex<f32>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f32>, #SparseVector>) {
        ^bb(%a: complex<f32>, %b: complex<f32>, %x: complex<f32>):
          %1 = complex.mul %a, %b : complex<f32>
          linalg.yield %1 : complex<f32>
    } -> tensor<?xcomplex<f32>, #SparseVector>
    return %0 : tensor<?xcomplex<f32>, #SparseVector>
  }

  // Driver method to call and verify complex kernels.
  func.func @main() {
    // Setup sparse vectors.
    %v1 = arith.constant sparse<
       [ [0], [28], [31] ],
         [ (511.13, 2.0), (3.0, 4.0), (5.0, 6.0) ] > : tensor<32xcomplex<f32>>
    %v2 = arith.constant sparse<
       [ [1], [28], [31] ],
         [ (1.0, 0.0), (2.0, 0.0), (3.0, 0.0) ] > : tensor<32xcomplex<f32>>
    %sv1 = sparse_tensor.convert %v1 : tensor<32xcomplex<f32>> to tensor<?xcomplex<f32>, #SparseVector>
    %sv2 = sparse_tensor.convert %v2 : tensor<32xcomplex<f32>> to tensor<?xcomplex<f32>, #SparseVector>

    // Call sparse vector kernels.
    %0 = call @cadd(%sv1, %sv2)
       : (tensor<?xcomplex<f32>, #SparseVector>,
          tensor<?xcomplex<f32>, #SparseVector>) -> tensor<?xcomplex<f32>, #SparseVector>
    %1 = call @cmul(%sv1, %sv2)
       : (tensor<?xcomplex<f32>, #SparseVector>,
          tensor<?xcomplex<f32>, #SparseVector>) -> tensor<?xcomplex<f32>, #SparseVector>

    //
    // Verify the results.
    //
    // CHECK:   ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 4
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 4 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 28, 31 )
    // CHECK-NEXT: values : ( ( 511.13, 2 ), ( 1, 0 ), ( 5, 4 ), ( 8, 6 ) )
    // CHECK-NEXT: ----
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 2
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 2 )
    // CHECK-NEXT: crd[0] : ( 28, 31 )
    // CHECK-NEXT: values : ( ( 6, 8 ), ( 15, 18 ) )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %0 : tensor<?xcomplex<f32>, #SparseVector>
    sparse_tensor.print %1 : tensor<?xcomplex<f32>, #SparseVector>

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xcomplex<f32>, #SparseVector>
    bufferization.dealloc_tensor %sv2 : tensor<?xcomplex<f32>, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<?xcomplex<f32>, #SparseVector>
    bufferization.dealloc_tensor %1 : tensor<?xcomplex<f32>, #SparseVector>
    return
  }
}
