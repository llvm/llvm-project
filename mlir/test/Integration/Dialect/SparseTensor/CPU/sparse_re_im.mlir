// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = enable-runtime-library=false
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

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
    %xv = bufferization.alloc_tensor(%d) : tensor<?xf32, #SparseVector>
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
    %xv = bufferization.alloc_tensor(%d) : tensor<?xf32, #SparseVector>
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
    %indices = sparse_tensor.indices %arg0 { dimension = 0 : index } : tensor<?xf32, #SparseVector> to memref<?xindex>
    %1 = vector.transfer_read %indices[%c0], %c0: memref<?xindex>, vector<3xindex>
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
