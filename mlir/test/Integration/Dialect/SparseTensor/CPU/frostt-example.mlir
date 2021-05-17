// RUN: mlir-opt %s \
// RUN:  -convert-scf-to-std -convert-vector-to-scf \
// RUN:  -convert-linalg-to-llvm -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: TENSOR0="%mlir_integration_test_dir/data/test.tns" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

//
// Use descriptive names for opaque pointers.
//
!Filename = type !llvm.ptr<i8>
!Tensor   = type !llvm.ptr<i8>

module {
  //
  // Example of using the sparse runtime support library to read a sparse tensor
  // in the FROSTT file format (http://frostt.io/tensors/file-formats.html).
  //
  func private @getTensorFilename(index) -> (!Filename)
  func private @openTensor(!Filename, memref<?xindex>) -> (!Tensor)
  func private @readTensorItem(!Tensor, memref<?xindex>, memref<?xf64>) -> ()
  func private @closeTensor(!Tensor) -> ()

  func @entry() {
    %d0  = constant 0.0 : f64
    %i0  = constant 0   : i64
    %c0  = constant 0   : index
    %c1  = constant 1   : index
    %c2  = constant 2   : index
    %c10 = constant 10  : index

    //
    // Setup memrefs to get meta data, indices and values.
    // The index array should provide sufficient space.
    //
    %idata = memref.alloc(%c10) : memref<?xindex>
    %ddata = memref.alloc(%c1)  : memref<?xf64>

    //
    // Obtain the sparse tensor filename through this test helper.
    //
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)

    //
    // Read a sparse tensor. The call yields a pointer to an opaque
    // memory-resident sparse tensor object that is only understood by
    // other methods in the sparse runtime support library. This call also
    // provides the rank and the number of nonzero elements (nnz) through
    // a memref array.
    //
    %tensor = call @openTensor(%fileName, %idata) : (!Filename, memref<?xindex>) -> (!Tensor)

    //
    // Print some meta data.
    //
    %rank = memref.load %idata[%c0] : memref<?xindex>
    %nnz  = memref.load %idata[%c1] : memref<?xindex>
    vector.print %rank : index
    vector.print %nnz  : index
    scf.for %r = %c2 to %c10 step %c1 {
      %d = memref.load %idata[%r] : memref<?xindex>
      vector.print %d : index
    }

    //
    // Now we are ready to read in the nonzero elements of the sparse tensor
    // and insert these into a sparse storage scheme. In this example, we
    // simply print the elements on the fly.
    //
    scf.for %k = %c0 to %nnz step %c1 {
      call @readTensorItem(%tensor, %idata, %ddata) : (!Tensor, memref<?xindex>, memref<?xf64>) -> ()
      //
      // Build index vector and print element (here, using the
      // knowledge that the read sparse tensor has rank 8).
      //
      %0 = vector.broadcast %i0 : i64 to vector<8xi64>
      %1 = scf.for %r = %c0 to %rank step %c1 iter_args(%in = %0) -> vector<8xi64> {
        %i  = memref.load %idata[%r] : memref<?xindex>
        %ii = index_cast %i : index to i64
        %ri = index_cast %r : index to i32
        %out = vector.insertelement %ii, %in[%ri : i32] : vector<8xi64>
        scf.yield %out : vector<8xi64>
      }
      %2 = memref.load %ddata[%c0] : memref<?xf64>
      vector.print %1 : vector<8xi64>
      vector.print %2 : f64
    }

    //
    // Since at this point we have processed the contents, make sure to
    // close the sparse tensor to release its memory resources.
    //
    call @closeTensor(%tensor) : (!Tensor) -> ()

    //
    // Verify that the results are as expected.
    //
    // CHECK: 8
    // CHECK: 16
    // CHECK: 7
    // CHECK: 3
    // CHECK: 3
    // CHECK: 3
    // CHECK: 3
    // CHECK: 3
    // CHECK: 5
    // CHECK: 3
    //
    // CHECK:      ( 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: 1
    // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 2 )
    // CHECK-NEXT: 1.3
    // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 4, 0 )
    // CHECK-NEXT: 1.5
    // CHECK-NEXT: ( 0, 0, 0, 1, 0, 0, 0, 1 )
    // CHECK-NEXT: 1.22
    // CHECK-NEXT: ( 0, 0, 0, 1, 0, 0, 0, 2 )
    // CHECK-NEXT: 1.23
    // CHECK-NEXT: ( 1, 0, 1, 0, 1, 1, 1, 0 )
    // CHECK-NEXT: 2.111
    // CHECK-NEXT: ( 1, 0, 1, 0, 1, 1, 1, 2 )
    // CHECK-NEXT: 2.113
    // CHECK-NEXT: ( 1, 1, 1, 0, 1, 1, 1, 0 )
    // CHECK-NEXT: 2.11
    // CHECK-NEXT: ( 1, 1, 1, 0, 1, 1, 1, 1 )
    // CHECK-NEXT: 2.1
    // CHECK-NEXT: ( 1, 1, 1, 1, 1, 1, 1, 1 )
    // CHECK-NEXT: 2
    // CHECK-NEXT: ( 2, 2, 2, 2, 0, 0, 1, 2 )
    // CHECK-NEXT: 3.112
    // CHECK-NEXT: ( 2, 2, 2, 2, 0, 1, 0, 2 )
    // CHECK-NEXT: 3.121
    // CHECK-NEXT: ( 2, 2, 2, 2, 0, 1, 1, 2 )
    // CHECK-NEXT: 3.122
    // CHECK-NEXT: ( 2, 2, 2, 2, 0, 2, 2, 2 )
    // CHECK-NEXT: 3.1
    // CHECK-NEXT: ( 2, 2, 2, 2, 2, 2, 2, 2 )
    // CHECK-NEXT: 3
    // CHECK-NEXT: ( 6, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: 7
    //

    //
    // Free.
    //
    memref.dealloc %idata : memref<?xindex>
    memref.dealloc %ddata : memref<?xf64>

    return
  }
}
