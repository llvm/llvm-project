// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

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
    %xv = sparse_tensor.init [%d] : tensor<?xf32, #SparseVector>
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
    %xv = sparse_tensor.init [%d] : tensor<?xf32, #SparseVector>
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
    %values = sparse_tensor.values %arg0 : tensor<?xf32, #SparseVector> to memref<?xf32>
    %0 = vector.transfer_read %values[%c0], %d0: memref<?xf32>, vector<4xf32>
    vector.print %0 : vector<4xf32>
    %indices = sparse_tensor.indices %arg0, %c0 : tensor<?xf32, #SparseVector> to memref<?xindex>
    %1 = vector.transfer_read %indices[%c0], %c0: memref<?xindex>, vector<4xindex>
    vector.print %1 : vector<4xindex>
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
    // CHECK: ( 5.13, 3, 5, -1 )
    // CHECK-NEXT: ( 0, 20, 31, 0 )
    // CHECK-NEXT: ( 2, 4, 6, -1 )
    // CHECK-NEXT: ( 0, 20, 31, 0 )
    //
    call @dump(%0) : (tensor<?xf32, #SparseVector>) -> ()
    call @dump(%1) : (tensor<?xf32, #SparseVector>) -> ()

    // Release the resources.
    sparse_tensor.release %sv1 : tensor<?xcomplex<f32>, #SparseVector>
    sparse_tensor.release %0 : tensor<?xf32, #SparseVector>
    sparse_tensor.release %1 : tensor<?xf32, #SparseVector>
    return
  }
}
