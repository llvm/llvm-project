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
// Do the same run, but now with vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>
#DenseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : dense)}>

//
// Traits for 1-d tensor (aka vector) operations.
//
#trait_scale = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) * 2.0"
}
#trait_scale_inpl = {
  indexing_maps = [
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) *= 2.0"
}
#trait_op = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>,  // b (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) OP b(i)"
}
#trait_dot = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>,  // b (in)
    affine_map<(i) -> ()>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) += a(i) * b(i)"
}

module {
  // Scales a sparse vector into a new sparse vector.
  func.func @vector_scale(%arga: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %s = arith.constant 2.0 : f64
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_scale
       ins(%arga: tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %x: f64):
          %1 = arith.mulf %a, %s : f64
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Scales a sparse vector in place.
  func.func @vector_scale_inplace(%argx: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %s = arith.constant 2.0 : f64
    %0 = linalg.generic #trait_scale_inpl
      outs(%argx: tensor<?xf64, #SparseVector>) {
        ^bb(%x: f64):
          %1 = arith.mulf %x, %s : f64
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Adds two sparse vectors into a new sparse vector.
  func.func @vector_add(%arga: tensor<?xf64, #SparseVector>,
                   %argb: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_op
       ins(%arga, %argb: tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = arith.addf %a, %b : f64
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Multiplies two sparse vectors into a new sparse vector.
  func.func @vector_mul(%arga: tensor<?xf64, #SparseVector>,
                   %argb: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_op
       ins(%arga, %argb: tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = arith.mulf %a, %b : f64
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Multiplies two sparse vectors into a new "annotated" dense vector.
  func.func @vector_mul_d(%arga: tensor<?xf64, #SparseVector>,
                     %argb: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #DenseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xf64, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xf64, #DenseVector>
    %0 = linalg.generic #trait_op
       ins(%arga, %argb: tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
        outs(%xv: tensor<?xf64, #DenseVector>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = arith.mulf %a, %b : f64
          linalg.yield %1 : f64
    } -> tensor<?xf64, #DenseVector>
    return %0 : tensor<?xf64, #DenseVector>
  }

  // Sum reduces dot product of two sparse vectors.
  func.func @vector_dotprod(%arga: tensor<?xf64, #SparseVector>,
                       %argb: tensor<?xf64, #SparseVector>,
                       %argx: tensor<f64>) -> tensor<f64> {
    %0 = linalg.generic #trait_dot
       ins(%arga, %argb: tensor<?xf64, #SparseVector>, tensor<?xf64, #SparseVector>)
        outs(%argx: tensor<f64>) {
        ^bb(%a: f64, %b: f64, %x: f64):
          %1 = arith.mulf %a, %b : f64
          %2 = arith.addf %x, %1 : f64
          linalg.yield %2 : f64
    } -> tensor<f64>
    return %0 : tensor<f64>
  }

  // Driver method to call and verify vector kernels.
  func.func @main() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant 1.1 : f64

    // Setup sparse vectors.
    %v1 = arith.constant sparse<
       [ [0], [3], [11], [17], [20], [21], [28], [29], [31] ],
         [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 ]
    > : tensor<32xf64>
    %v2 = arith.constant sparse<
       [ [1], [3], [4], [10], [16], [18], [21], [28], [29], [31] ],
         [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0 ]
    > : tensor<32xf64>
    %sv1 = sparse_tensor.convert %v1 : tensor<32xf64> to tensor<?xf64, #SparseVector>
    // TODO: Use %sv1 when copying sparse tensors is supported.
    %sv1_dup = sparse_tensor.convert %v1 : tensor<32xf64> to tensor<?xf64, #SparseVector>
    %sv2 = sparse_tensor.convert %v2 : tensor<32xf64> to tensor<?xf64, #SparseVector>

    // Setup memory for a single reduction scalar.
    %x = tensor.from_elements %d1 : tensor<f64>

    // Call sparse vector kernels.
    %0 = call @vector_scale(%sv1)
       : (tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>
    %1 = call @vector_scale_inplace(%sv1_dup)
       : (tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>
    %2 = call @vector_add(%1, %sv2)
       : (tensor<?xf64, #SparseVector>,
          tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>
    %3 = call @vector_mul(%1, %sv2)
       : (tensor<?xf64, #SparseVector>,
          tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector>
    %4 = call @vector_mul_d(%1, %sv2)
       : (tensor<?xf64, #SparseVector>,
          tensor<?xf64, #SparseVector>) -> tensor<?xf64, #DenseVector>
    %5 = call @vector_dotprod(%1, %sv2, %x)
       : (tensor<?xf64, #SparseVector>,
          tensor<?xf64, #SparseVector>, tensor<f64>) -> tensor<f64>

    //
    // Verify the results.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 9
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 9 )
    // CHECK-NEXT: crd[0] : ( 0, 3, 11, 17, 20, 21, 28, 29, 31 )
    // CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8, 9 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 10
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 10 )
    // CHECK-NEXT: crd[0] : ( 1, 3, 4, 10, 16, 18, 21, 28, 29, 31 )
    // CHECK-NEXT: values : ( 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 9
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 9 )
    // CHECK-NEXT: crd[0] : ( 0, 3, 11, 17, 20, 21, 28, 29, 31 )
    // CHECK-NEXT: values : ( 2, 4, 6, 8, 10, 12, 14, 16, 18 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 9
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 9 )
    // CHECK-NEXT: crd[0] : ( 0, 3, 11, 17, 20, 21, 28, 29, 31 )
    // CHECK-NEXT: values : ( 2, 4, 6, 8, 10, 12, 14, 16, 18 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 14
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 14 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 3, 4, 10, 11, 16, 17, 18, 20, 21, 28, 29, 31 )
    // CHECK-NEXT: values : ( 2, 11, 16, 13, 14, 6, 15, 8, 16, 10, 29, 32, 35, 38 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 5
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 5 )
    // CHECK-NEXT: crd[0] : ( 3, 21, 28, 29, 31 )
    // CHECK-NEXT: values : ( 48, 204, 252, 304, 360 )
    // CHECK-NEXT: ----
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 32
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: values : ( 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 204, 0, 0, 0, 0, 0, 0, 252, 304, 0, 360 )
    // CHECK-NEXT: ----
    // CHECK-NEXT: 1169.1
    //
    sparse_tensor.print %sv1 : tensor<?xf64, #SparseVector>
    sparse_tensor.print %sv2 : tensor<?xf64, #SparseVector>
    sparse_tensor.print %0 : tensor<?xf64, #SparseVector>
    sparse_tensor.print %1 : tensor<?xf64, #SparseVector>
    sparse_tensor.print %2 : tensor<?xf64, #SparseVector>
    sparse_tensor.print %3 : tensor<?xf64, #SparseVector>
    sparse_tensor.print %4 : tensor<?xf64, #DenseVector>
    %v5 = tensor.extract %5[] : tensor<f64>
    vector.print %v5 : f64

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %sv1_dup : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %sv2 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<?xf64, #SparseVector>
    // Note: No dealloc for %1 because it was inplace!
    bufferization.dealloc_tensor %2 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %3 : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %4 : tensor<?xf64, #DenseVector>
    bufferization.dealloc_tensor %5 : tensor<f64>
    return
  }
}
