// RUN: mlir-opt %s \
// RUN:  -convert-scf-to-std -convert-vector-to-scf \
// RUN:  -convert-linalg-to-llvm -convert-vector-to-llvm | \
// RUN: TENSOR0="%mlir_integration_test_dir/data/test.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

module {
  //
  // Example of using the sparse runtime support library to read a sparse matrix
  // in the Matrix Market Exchange Format (https://math.nist.gov/MatrixMarket).
  //
  func private @openTensor(!llvm.ptr<i8>, memref<?xindex>) -> (!llvm.ptr<i8>)
  func private @readTensorItem(!llvm.ptr<i8>, memref<?xindex>, memref<?xf64>) -> ()
  func private @closeTensor(!llvm.ptr<i8>) -> ()
  func private @getTensorFilename(index) -> (!llvm.ptr<i8>)

  func @entry() {
    %d0  = constant 0.0 : f64
    %c0  = constant 0 : index
    %c1  = constant 1 : index
    %c2  = constant 2 : index
    %c3  = constant 3 : index
    %c4  = constant 4 : index
    %c5  = constant 5 : index

    //
    // Setup memrefs to get meta data, indices, and values.
    //
    %idata = alloc(%c4) : memref<?xindex>
    %ddata = alloc(%c1) : memref<?xf64>

    //
    // Obtain the sparse matrix filename through this test helper.
    //
    %fileName = call @getTensorFilename(%c0) : (index) -> (!llvm.ptr<i8>)

    //
    // Read a sparse matrix. The call yields a pointer to an opaque
    // memory-resident sparse tensor object that is only understood by
    // other methods in the sparse runtime support library. This call also
    // provides the rank (always 2 for the Matrix Market), number of
    // nonzero elements (nnz), and the size (m x n) through a memref array.
    //
    %tensor = call @openTensor(%fileName, %idata)
      : (!llvm.ptr<i8>, memref<?xindex>) -> (!llvm.ptr<i8>)
    %rank = load %idata[%c0] : memref<?xindex>
    %nnz  = load %idata[%c1] : memref<?xindex>
    %m    = load %idata[%c2] : memref<?xindex>
    %n    = load %idata[%c3] : memref<?xindex>

    //
    // At this point, code should prepare a proper sparse storage scheme for
    // an m x n matrix with nnz nonzero elements. For simplicity, here we
    // simply intialize a dense m x n matrix to all zeroes.
    //
    %a = alloc(%m, %n) : memref<?x?xf64>
    scf.for %ii = %c0 to %m step %c1 {
      scf.for %jj = %c0 to %n step %c1 {
        store %d0, %a[%ii, %jj] : memref<?x?xf64>
      }
    }

    //
    // Now we are ready to read in nnz nonzero elements of the sparse matrix
    // and insert these into a sparse storage scheme. In this example, we
    // simply insert them in the dense matrix.
    //
    scf.for %k = %c0 to %nnz step %c1 {
      call @readTensorItem(%tensor, %idata, %ddata)
        : (!llvm.ptr<i8>, memref<?xindex>, memref<?xf64>) -> ()
      %i = load %idata[%c0] : memref<?xindex>
      %j = load %idata[%c1] : memref<?xindex>
      %d = load %ddata[%c0] : memref<?xf64>
      store %d, %a[%i, %j] : memref<?x?xf64>
    }

    //
    // Since at this point we have copied the sparse matrix to our own
    // storage scheme, make sure to close the matrix to release its
    // memory resources.
    //
    call @closeTensor(%tensor) : (!llvm.ptr<i8>) -> ()

    //
    // Verify that the results are as expected.
    //
    %A = vector.transfer_read %a[%c0, %c0], %d0 : memref<?x?xf64>, vector<5x5xf64>
    vector.print %rank : index
    vector.print %nnz  : index
    vector.print %m    : index
    vector.print %n    : index
    vector.print %A    : vector<5x5xf64>
    //
    // CHECK: 2
    // CHECK: 9
    // CHECK: 5
    // CHECK: 5
    //
    // CHECK:      ( ( 1, 0, 0, 1.4, 0 ),
    // CHECK-SAME:   ( 0, 2, 0, 0, 2.5 ),
    // CHECK-SAME:   ( 0, 0, 3, 0, 0 ),
    // CHECK-SAME:   ( 4.1, 0, 0, 4, 0 ),
    // CHECK-SAME:   ( 0, 5.2, 0, 0, 5 ) )

    //
    // Free.
    //
    dealloc %idata : memref<?xindex>
    dealloc %ddata : memref<?xf64>
    dealloc %a     : memref<?x?xf64>

    return
  }
}
