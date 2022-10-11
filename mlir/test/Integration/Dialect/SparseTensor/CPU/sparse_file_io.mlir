// RUN: mlir-opt %s --sparse-compiler | \
// RUN: TENSOR0="%mlir_src_dir/test/Integration/data/wide.mtx" \
// RUN: TENSOR1="" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

!Filename = !llvm.ptr<i8>
!TensorReader = !llvm.ptr<i8>
!TensorWriter = !llvm.ptr<i8>

module {

  func.func private @getTensorFilename(index) -> (!Filename)

  func.func private @createSparseTensorReader(!Filename) -> (!TensorReader)
  func.func private @delSparseTensorReader(!TensorReader) -> ()
  func.func private @getSparseTensorReaderRank(!TensorReader) -> (index)
  func.func private @getSparseTensorReaderNNZ(!TensorReader) -> (index)
  func.func private @getSparseTensorReaderIsSymmetric(!TensorReader) -> (i1)
  func.func private @getSparseTensorReaderDimSizes(!TensorReader, memref<?xindex>)
    -> () attributes { llvm.emit_c_interface }
  func.func private @getSparseTensorReaderNextF32(!TensorReader, memref<?xindex>)
    -> (f32) attributes { llvm.emit_c_interface }

  func.func private @createSparseTensorWriter(!Filename) -> (!TensorWriter)
  func.func private @delSparseTensorWriter(!TensorWriter)
  func.func private @outSparseTensorWriterMetaData(!TensorWriter, index, index,
    memref<?xindex>) -> () attributes { llvm.emit_c_interface }
  func.func private @outSparseTensorWriterNextF32(!TensorWriter, index,
    memref<?xindex>, f32) -> () attributes { llvm.emit_c_interface }

  func.func @dumpi(%arg0: memref<?xindex>) {
    %c0 = arith.constant 0 : index
    %v = vector.transfer_read %arg0[%c0], %c0: memref<?xindex>, vector<17xindex>
    vector.print %v : vector<17xindex>
    return
  }

  func.func @dumpf(%arg0: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %d0 = arith.constant 0.0 : f32
    %v = vector.transfer_read %arg0[%c0], %d0: memref<?xf32>, vector<17xf32>
    vector.print %v : vector<17xf32>
    return
  }

  // Returns the indices and values of the tensor.
  func.func @readTensorFile(%tensor: !TensorReader)
    -> (memref<?xindex>, memref<?xindex>, memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %rank = call @getSparseTensorReaderRank(%tensor) : (!TensorReader) -> index
    %nnz = call @getSparseTensorReaderNNZ(%tensor) : (!TensorReader) -> index

    // Assume rank == 2.
    %x0s = memref.alloc(%nnz) : memref<?xindex>
    %x1s = memref.alloc(%nnz) : memref<?xindex>
    %vs = memref.alloc(%nnz) : memref<?xf32>
    %indices = memref.alloc(%rank) : memref<?xindex>
    scf.for %i = %c0 to %nnz step %c1 {
      %v = func.call @getSparseTensorReaderNextF32(%tensor, %indices)
        : (!TensorReader, memref<?xindex>) -> f32
      memref.store %v, %vs[%i] : memref<?xf32>
      %i0 = memref.load %indices[%c0] : memref<?xindex>
      memref.store %i0, %x0s[%i] : memref<?xindex>
      %i1 = memref.load %indices[%c1] : memref<?xindex>
      memref.store %i1, %x1s[%i] : memref<?xindex>
    }

    // Release the resource for the indices.
    memref.dealloc %indices : memref<?xindex>
    return %x0s, %x1s, %vs : memref<?xindex>, memref<?xindex>, memref<?xf32>
  }

  // Reads a COO tensor from the given file name and prints its content.
  func.func @readTensorFileAndDump(%fileName: !Filename) {
    %tensor = call @createSparseTensorReader(%fileName)
      : (!Filename) -> (!TensorReader)
    %rank = call @getSparseTensorReaderRank(%tensor) : (!TensorReader) -> index
    vector.print %rank : index
    %nnz = call @getSparseTensorReaderNNZ(%tensor) : (!TensorReader) -> index
    vector.print %nnz : index
    %symmetric = call @getSparseTensorReaderIsSymmetric(%tensor)
      : (!TensorReader) -> i1
    vector.print %symmetric : i1
    %dimSizes = memref.alloc(%rank) : memref<?xindex>
    func.call @getSparseTensorReaderDimSizes(%tensor, %dimSizes)
      : (!TensorReader, memref<?xindex>) -> ()
    call @dumpi(%dimSizes) : (memref<?xindex>) -> ()
    %x0s, %x1s, %vs = call @readTensorFile(%tensor)
      : (!TensorReader) -> (memref<?xindex>, memref<?xindex>, memref<?xf32>)

    call @dumpi(%x0s) : (memref<?xindex>) -> ()
    call @dumpi(%x1s) : (memref<?xindex>) -> ()
    call @dumpf(%vs) : (memref<?xf32>) -> ()

    // Release the resources.
    call @delSparseTensorReader(%tensor) : (!TensorReader) -> ()
    memref.dealloc %dimSizes : memref<?xindex>
    memref.dealloc %x0s : memref<?xindex>
    memref.dealloc %x1s : memref<?xindex>
    memref.dealloc %vs : memref<?xf32>

    return
  }

  // Reads a COO tensor from a file with fileName0 and writes its content to
  // another file with fileName1.
  func.func @createTensorFileFrom(%fileName0: !Filename, %fileName1: !Filename) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %tensor0 = call @createSparseTensorReader(%fileName0)
      : (!Filename) -> (!TensorReader)
    %tensor1 = call @createSparseTensorWriter(%fileName1)
      : (!Filename) -> (!TensorWriter)

    %rank = call @getSparseTensorReaderRank(%tensor0) : (!TensorReader) -> index
    %nnz = call @getSparseTensorReaderNNZ(%tensor0) : (!TensorReader) -> index
    %dimSizes = memref.alloc(%rank) : memref<?xindex>
    func.call @getSparseTensorReaderDimSizes(%tensor0,%dimSizes)
      : (!TensorReader, memref<?xindex>) -> ()
    call @outSparseTensorWriterMetaData(%tensor1, %rank, %nnz, %dimSizes)
      : (!TensorWriter, index, index, memref<?xindex>) -> ()

    //TODO: handle isSymmetric.
    // Assume rank == 2.
    %indices = memref.alloc(%rank) : memref<?xindex>
    scf.for %i = %c0 to %nnz step %c1 {
      %v = func.call @getSparseTensorReaderNextF32(%tensor0, %indices)
        : (!TensorReader, memref<?xindex>) -> f32
      func.call @outSparseTensorWriterNextF32(%tensor1, %rank, %indices, %v)
        : (!TensorWriter, index, memref<?xindex>, f32) -> ()
    }

    // Release the resources.
    call @delSparseTensorReader(%tensor0) : (!TensorReader) -> ()
    call @delSparseTensorWriter(%tensor1) : (!TensorWriter) -> ()
    memref.dealloc %dimSizes : memref<?xindex>
    memref.dealloc %indices : memref<?xindex>

    return
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %fileName0 = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %fileName1 = call @getTensorFilename(%c1) : (index) -> (!Filename)

    // Write the sparse tensor data from file through the SparseTensorReader and
    // print the data.
    // CHECK: 2
    // CHECK: 17
    // CHECK: 0
    // CHECK: ( 4, 256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK: ( 0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 )
    // CHECK: ( 0, 126, 127, 254, 1, 253, 2, 0, 1, 3, 98, 126, 127, 128, 249, 253, 255 )
    // CHECK: ( -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17 )
    call @readTensorFileAndDump(%fileName0) : (!Filename) -> ()

    // Write the sparse tensor data to std::cout through the SparseTensorWriter.
    // CHECK: # extended FROSTT format
    // CHECK: 2 17
    // CHECK: 4 256
    // CHECK: 1 1 -1
    // CHECK: 1 127 2
    // CHECK: 1 128 -3
    // CHECK: 1 255 4
    // CHECK: 2 2 -5
    // CHECK: 2 254 6
    // CHECK: 3 3 -7
    // CHECK: 4 1 8
    // CHECK: 4 2 -9
    // CHECK: 4 4 10
    // CHECK: 4 99 -11
    // CHECK: 4 127 12
    // CHECK: 4 128 -13
    // CHECK: 4 129 14
    // CHECK: 4 250 -15
    // CHECK: 4 254 16
    // CHECK: 4 256 -17
    call @createTensorFileFrom(%fileName0, %fileName1)
      : (!Filename, !Filename) -> ()

    return
  }
}
