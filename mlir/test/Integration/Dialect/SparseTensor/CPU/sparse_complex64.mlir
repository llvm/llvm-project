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
// DEFINE: %{run_opts} = -e entry -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

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
  func.func @cadd(%arga: tensor<?xcomplex<f64>, #SparseVector>,
                  %argb: tensor<?xcomplex<f64>, #SparseVector>)
                      -> tensor<?xcomplex<f64>, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xcomplex<f64>, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xcomplex<f64>, #SparseVector>
    %0 = linalg.generic #trait_op
       ins(%arga, %argb: tensor<?xcomplex<f64>, #SparseVector>,
                         tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f64>, #SparseVector>) {
        ^bb(%a: complex<f64>, %b: complex<f64>, %x: complex<f64>):
          %1 = complex.add %a, %b : complex<f64>
          linalg.yield %1 : complex<f64>
    } -> tensor<?xcomplex<f64>, #SparseVector>
    return %0 : tensor<?xcomplex<f64>, #SparseVector>
  }

  func.func @cmul(%arga: tensor<?xcomplex<f64>, #SparseVector>,
                  %argb: tensor<?xcomplex<f64>, #SparseVector>)
                      -> tensor<?xcomplex<f64>, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xcomplex<f64>, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xcomplex<f64>, #SparseVector>
    %0 = linalg.generic #trait_op
       ins(%arga, %argb: tensor<?xcomplex<f64>, #SparseVector>,
                         tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f64>, #SparseVector>) {
        ^bb(%a: complex<f64>, %b: complex<f64>, %x: complex<f64>):
          %1 = complex.mul %a, %b : complex<f64>
          linalg.yield %1 : complex<f64>
    } -> tensor<?xcomplex<f64>, #SparseVector>
    return %0 : tensor<?xcomplex<f64>, #SparseVector>
  }

  func.func @dump(%arg0: tensor<?xcomplex<f64>, #SparseVector>, %d: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mem = sparse_tensor.values %arg0 : tensor<?xcomplex<f64>, #SparseVector> to memref<?xcomplex<f64>>
    scf.for %i = %c0 to %d step %c1 {
       %v = memref.load %mem[%i] : memref<?xcomplex<f64>>
       %real = complex.re %v : complex<f64>
       %imag = complex.im %v : complex<f64>
       vector.print %real : f64
       vector.print %imag : f64
    }
    return
  }

  // Driver method to call and verify complex kernels.
  func.func @entry() {
    // Setup sparse vectors.
    %v1 = arith.constant sparse<
       [ [0], [28], [31] ],
         [ (511.13, 2.0), (3.0, 4.0), (5.0, 6.0) ] > : tensor<32xcomplex<f64>>
    %v2 = arith.constant sparse<
       [ [1], [28], [31] ],
         [ (1.0, 0.0), (2.0, 0.0), (3.0, 0.0) ] > : tensor<32xcomplex<f64>>
    %sv1 = sparse_tensor.convert %v1 : tensor<32xcomplex<f64>> to tensor<?xcomplex<f64>, #SparseVector>
    %sv2 = sparse_tensor.convert %v2 : tensor<32xcomplex<f64>> to tensor<?xcomplex<f64>, #SparseVector>

    // Call sparse vector kernels.
    %0 = call @cadd(%sv1, %sv2)
       : (tensor<?xcomplex<f64>, #SparseVector>,
          tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xcomplex<f64>, #SparseVector>
    %1 = call @cmul(%sv1, %sv2)
       : (tensor<?xcomplex<f64>, #SparseVector>,
          tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xcomplex<f64>, #SparseVector>

    //
    // Verify the results.
    //
    // CHECK: 511.13
    // CHECK-NEXT: 2
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 5
    // CHECK-NEXT: 4
    // CHECK-NEXT: 8
    // CHECK-NEXT: 6
    // CHECK-NEXT: 6
    // CHECK-NEXT: 8
    // CHECK-NEXT: 15
    // CHECK-NEXT: 18
    //
    %d1 = arith.constant 4 : index
    %d2 = arith.constant 2 : index
    call @dump(%0, %d1) : (tensor<?xcomplex<f64>, #SparseVector>, index) -> ()
    call @dump(%1, %d2) : (tensor<?xcomplex<f64>, #SparseVector>, index) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %sv2 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %1 : tensor<?xcomplex<f64>, #SparseVector>
    return
  }
}
