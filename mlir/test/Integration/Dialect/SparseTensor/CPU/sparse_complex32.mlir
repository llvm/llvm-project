// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true"
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
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
    %xv = bufferization.alloc_tensor(%d) : tensor<?xcomplex<f32>, #SparseVector>
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
    %xv = bufferization.alloc_tensor(%d) : tensor<?xcomplex<f32>, #SparseVector>
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

  func.func @dump(%arg0: tensor<?xcomplex<f32>, #SparseVector>, %d: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mem = sparse_tensor.values %arg0 : tensor<?xcomplex<f32>, #SparseVector> to memref<?xcomplex<f32>>
    scf.for %i = %c0 to %d step %c1 {
       %v = memref.load %mem[%i] : memref<?xcomplex<f32>>
       %real = complex.re %v : complex<f32>
       %imag = complex.im %v : complex<f32>
       vector.print %real : f32
       vector.print %imag : f32
    }
    return
  }

  // Driver method to call and verify complex kernels.
  func.func @entry() {
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
    call @dump(%0, %d1) : (tensor<?xcomplex<f32>, #SparseVector>, index) -> ()
    call @dump(%1, %d2) : (tensor<?xcomplex<f32>, #SparseVector>, index) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xcomplex<f32>, #SparseVector>
    bufferization.dealloc_tensor %sv2 : tensor<?xcomplex<f32>, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<?xcomplex<f32>, #SparseVector>
    bufferization.dealloc_tensor %1 : tensor<?xcomplex<f32>, #SparseVector>
    return
  }
}
