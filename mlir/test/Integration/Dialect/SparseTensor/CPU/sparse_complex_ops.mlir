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

#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>

#trait_op1 = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = OP a(i)"
}

#trait_op2 = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>,  // b (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) OP b(i)"
}

module {
  func.func @cops(%arga: tensor<?xcomplex<f64>, #SparseVector>,
                  %argb: tensor<?xcomplex<f64>, #SparseVector>)
                 -> tensor<?xcomplex<f64>, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arga, %c0 : tensor<?xcomplex<f64>, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xcomplex<f64>, #SparseVector>
    %0 = linalg.generic #trait_op2
       ins(%arga, %argb: tensor<?xcomplex<f64>, #SparseVector>,
                         tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f64>, #SparseVector>) {
        ^bb(%a: complex<f64>, %b: complex<f64>, %x: complex<f64>):
          %1 = complex.neg %b : complex<f64>
          %2 = complex.sub %a, %1 : complex<f64>
          linalg.yield %2 : complex<f64>
    } -> tensor<?xcomplex<f64>, #SparseVector>
    return %0 : tensor<?xcomplex<f64>, #SparseVector>
  }

  func.func @csin(%arga: tensor<?xcomplex<f64>, #SparseVector>)
                 -> tensor<?xcomplex<f64>, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arga, %c0 : tensor<?xcomplex<f64>, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xcomplex<f64>, #SparseVector>
    %0 = linalg.generic #trait_op1
       ins(%arga: tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f64>, #SparseVector>) {
        ^bb(%a: complex<f64>, %x: complex<f64>):
          %1 = complex.sin %a : complex<f64>
          linalg.yield %1 : complex<f64>
    } -> tensor<?xcomplex<f64>, #SparseVector>
    return %0 : tensor<?xcomplex<f64>, #SparseVector>
  }

  func.func @complex_sqrt(%arga: tensor<?xcomplex<f64>, #SparseVector>)
                 -> tensor<?xcomplex<f64>, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arga, %c0 : tensor<?xcomplex<f64>, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xcomplex<f64>, #SparseVector>
    %0 = linalg.generic #trait_op1
       ins(%arga: tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f64>, #SparseVector>) {
        ^bb(%a: complex<f64>, %x: complex<f64>):
          %1 = complex.sqrt %a : complex<f64>
          linalg.yield %1 : complex<f64>
    } -> tensor<?xcomplex<f64>, #SparseVector>
    return %0 : tensor<?xcomplex<f64>, #SparseVector>
  }

  func.func @complex_tanh(%arga: tensor<?xcomplex<f64>, #SparseVector>)
                 -> tensor<?xcomplex<f64>, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arga, %c0 : tensor<?xcomplex<f64>, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xcomplex<f64>, #SparseVector>
    %0 = linalg.generic #trait_op1
       ins(%arga: tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f64>, #SparseVector>) {
       ^bb(%a: complex<f64>, %x: complex<f64>):
          %1 = complex.tanh %a : complex<f64>
          linalg.yield %1 : complex<f64>
   } -> tensor<?xcomplex<f64>, #SparseVector>
    return %0 : tensor<?xcomplex<f64>, #SparseVector>
  }

  func.func @clog1p_expm1(%arga: tensor<?xcomplex<f64>, #SparseVector>)
                 -> tensor<?xcomplex<f64>, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arga, %c0 : tensor<?xcomplex<f64>, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xcomplex<f64>, #SparseVector>
    %0 = linalg.generic #trait_op1
       ins(%arga: tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f64>, #SparseVector>) {
        ^bb(%a: complex<f64>, %x: complex<f64>):
          %1 = complex.log1p %a : complex<f64>
          %2 = complex.expm1 %1 : complex<f64>
          linalg.yield %2 : complex<f64>
    } -> tensor<?xcomplex<f64>, #SparseVector>
    return %0 : tensor<?xcomplex<f64>, #SparseVector>
  }

  func.func @cdiv(%arga: tensor<?xcomplex<f64>, #SparseVector>)
                 -> tensor<?xcomplex<f64>, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arga, %c0 : tensor<?xcomplex<f64>, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xcomplex<f64>, #SparseVector>
    %c = complex.constant [2.0 : f64, 0.0 : f64] : complex<f64>
    %0 = linalg.generic #trait_op1
       ins(%arga: tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f64>, #SparseVector>) {
        ^bb(%a: complex<f64>, %x: complex<f64>):
          %1 = complex.div %a, %c  : complex<f64>
          linalg.yield %1 : complex<f64>
    } -> tensor<?xcomplex<f64>, #SparseVector>
    return %0 : tensor<?xcomplex<f64>, #SparseVector>
  }

  func.func @cabs(%arga: tensor<?xcomplex<f64>, #SparseVector>)
                 -> tensor<?xf64, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arga, %c0 : tensor<?xcomplex<f64>, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_op1
       ins(%arga: tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: complex<f64>, %x: f64):
          %1 = complex.abs %a : complex<f64>
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  // Driver method to call and verify complex kernels.
  func.func @main() {
    // Setup sparse vectors.
    %v1 = arith.constant sparse<
       [ [0], [28], [31] ],
         [ (-5.13, 2.0), (3.0, 4.0), (5.0, 6.0) ] > : tensor<32xcomplex<f64>>
    %v2 = arith.constant sparse<
       [ [1], [28], [31] ],
         [ (1.0, 0.0), (-2.0, 0.0), (3.0, 0.0) ] > : tensor<32xcomplex<f64>>
    %sv1 = sparse_tensor.convert %v1 : tensor<32xcomplex<f64>> to tensor<?xcomplex<f64>, #SparseVector>
    %sv2 = sparse_tensor.convert %v2 : tensor<32xcomplex<f64>> to tensor<?xcomplex<f64>, #SparseVector>

    // Call sparse vector kernels.
    %0 = call @cops(%sv1, %sv2)
       : (tensor<?xcomplex<f64>, #SparseVector>,
          tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xcomplex<f64>, #SparseVector>
    %1 = call @csin(%sv1)
       : (tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xcomplex<f64>, #SparseVector>
    %2 = call @complex_sqrt(%sv1)
       : (tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xcomplex<f64>, #SparseVector>
    %3 = call @complex_tanh(%sv2)
       : (tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xcomplex<f64>, #SparseVector>
    %4 = call @clog1p_expm1(%sv1)
       : (tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xcomplex<f64>, #SparseVector>
    %5 = call @cdiv(%sv1)
       : (tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xcomplex<f64>, #SparseVector>
    %6 = call @cabs(%sv1)
       : (tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xf64, #SparseVector>

    //
    // Verify the results.
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 4
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 4 )
    // CHECK-NEXT: crd[0] : ( 0, 1, 28, 31 )
    // CHECK-NEXT: values : ( ( -5.13, 2 ), ( 1, 0 ), ( 1, 4 ), ( 8, 6 ) )
    // CHECK-NEXT: ----
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 3
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 0, 28, 31 )
    // CHECK-NEXT: values : ( ( 3.43887, 1.47097 ), ( 3.85374, -27.0168 ), ( -193.43, 57.2184 ) )
    // CHECK-NEXT: ----
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 3
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 0, 28, 31 )
    // CHECK-NEXT: values : ( ( 0.433635, 2.30609 ), ( 2, 1 ), ( 2.53083, 1.18538 ) )
    // CHECK-NEXT: ----
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 3
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 1, 28, 31 )
    // CHECK-NEXT: values : ( ( 0.761594, 0 ), ( -0.964028, 0 ), ( 0.995055, 0 ) )
    // CHECK-NEXT: ----
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 3
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 0, 28, 31 )
    // CHECK-NEXT: values : ( ( -5.13, 2 ), ( 3, 4 ), ( 5, 6 ) )
    // CHECK-NEXT: ----
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 3
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 0, 28, 31 )
    // CHECK-NEXT: values : ( ( -2.565, 1 ), ( 1.5, 2 ), ( 2.5, 3 ) )
    // CHECK-NEXT: ----
    //
    // CHECK-NEXT: ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 3
    // CHECK-NEXT: dim = ( 32 )
    // CHECK-NEXT: lvl = ( 32 )
    // CHECK-NEXT: pos[0] : ( 0, 3 )
    // CHECK-NEXT: crd[0] : ( 0, 28, 31 )
    // CHECK-NEXT: values : ( 5.50608, 5, 7.81025 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %0 : tensor<?xcomplex<f64>, #SparseVector>
    sparse_tensor.print %1 : tensor<?xcomplex<f64>, #SparseVector>
    sparse_tensor.print %2 : tensor<?xcomplex<f64>, #SparseVector>
    sparse_tensor.print %3 : tensor<?xcomplex<f64>, #SparseVector>
    sparse_tensor.print %4 : tensor<?xcomplex<f64>, #SparseVector>
    sparse_tensor.print %5 : tensor<?xcomplex<f64>, #SparseVector>
    sparse_tensor.print %6 : tensor<?xf64, #SparseVector>

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %sv2 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %1 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %2 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %3 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %4 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %5 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %6 : tensor<?xf64, #SparseVector>
    return
  }
}
