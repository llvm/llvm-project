//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparse_compiler_opts} = enable-runtime-library=true
// DEFINE: %{sparse_compiler_opts_sve} = enable-arm-sve=true %{sparse_compiler_opts}
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_opts} = -e entry -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>

#trait_op = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = OP a(i)"
}

module {
  func.func @cre(%arga: tensor<?xcomplex<f32>, #SparseVector>)
                -> tensor<?xf32, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xcomplex<f32>, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xf32, #SparseVector>
    %0 = linalg.generic #trait_op
       ins(%arga: tensor<?xcomplex<f32>, #SparseVector>)
        outs(%xv: tensor<?xf32, #SparseVector>) {
        ^bb(%a: complex<f32>, %x: f32):
          %1 = complex.re %a : complex<f32>
          linalg.yield %1 : f32
    } -> tensor<?xf32, #SparseVector>
    return %0 : tensor<?xf32, #SparseVector>
  }

  func.func @cim(%arga: tensor<?xcomplex<f32>, #SparseVector>)
                -> tensor<?xf32, #SparseVector> {
    %c = arith.constant 0 : index
    %d = tensor.dim %arga, %c : tensor<?xcomplex<f32>, #SparseVector>
    %xv = tensor.empty(%d) : tensor<?xf32, #SparseVector>
    %0 = linalg.generic #trait_op
       ins(%arga: tensor<?xcomplex<f32>, #SparseVector>)
        outs(%xv: tensor<?xf32, #SparseVector>) {
        ^bb(%a: complex<f32>, %x: f32):
          %1 = complex.im %a : complex<f32>
          linalg.yield %1 : f32
    } -> tensor<?xf32, #SparseVector>
    return %0 : tensor<?xf32, #SparseVector>
  }

  func.func @dump(%arg0: tensor<?xf32, #SparseVector>) {
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : f32
    %n = sparse_tensor.number_of_entries %arg0 : tensor<?xf32, #SparseVector>
    vector.print %n : index
    %values = sparse_tensor.values %arg0 : tensor<?xf32, #SparseVector> to memref<?xf32>
    %0 = vector.transfer_read %values[%c0], %d0: memref<?xf32>, vector<3xf32>
    vector.print %0 : vector<3xf32>
    %coordinates = sparse_tensor.coordinates %arg0 { level = 0 : index } : tensor<?xf32, #SparseVector> to memref<?xindex>
    %1 = vector.transfer_read %coordinates[%c0], %c0: memref<?xindex>, vector<3xindex>
    vector.print %1 : vector<3xindex>
    return
  }

  // Driver method to call and verify functions cim and cre.
  func.func @entry() {
    // Setup sparse vectors.
    %v1 = arith.constant sparse<
       [ [0], [20], [31] ],
         [ (5.13, 2.0), (3.0, 4.0), (5.0, 6.0) ] > : tensor<32xcomplex<f32>>
    %sv1 = sparse_tensor.convert %v1 : tensor<32xcomplex<f32>> to tensor<?xcomplex<f32>, #SparseVector>

    // Call sparse vector kernels.
    %0 = call @cre(%sv1)
       : (tensor<?xcomplex<f32>, #SparseVector>) -> tensor<?xf32, #SparseVector>

    %1 = call @cim(%sv1)
       : (tensor<?xcomplex<f32>, #SparseVector>) -> tensor<?xf32, #SparseVector>

    //
    // Verify the results.
    //
    // CHECK:      3
    // CHECK-NEXT: ( 5.13, 3, 5 )
    // CHECK-NEXT: ( 0, 20, 31 )
    // CHECK-NEXT: 3
    // CHECK-NEXT: ( 2, 4, 6 )
    // CHECK-NEXT: ( 0, 20, 31 )
    //
    call @dump(%0) : (tensor<?xf32, #SparseVector>) -> ()
    call @dump(%1) : (tensor<?xf32, #SparseVector>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xcomplex<f32>, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<?xf32, #SparseVector>
    bufferization.dealloc_tensor %1 : tensor<?xf32, #SparseVector>
    return
  }
}
