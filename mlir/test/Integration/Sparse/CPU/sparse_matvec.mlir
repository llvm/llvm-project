// RUN: mlir-opt %s \
// RUN:   --test-sparsification="lower ptr-type=4 ind-type=4" \
// RUN:   --convert-linalg-to-loops --convert-vector-to-scf --convert-scf-to-std \
// RUN:   --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN:   --std-bufferize --finalizing-bufferize  \
// RUN:   --convert-vector-to-llvm --convert-std-to-llvm | \
// RUN: TENSOR0="%mlir_integration_test_dir/data/wide.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
// RUN: mlir-opt %s \
// RUN:   --test-sparsification="lower vectorization-strategy=2 ptr-type=4 ind-type=4 vl=16" \
// RUN:   --convert-linalg-to-loops --convert-vector-to-scf --convert-scf-to-std \
// RUN:   --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN:   --std-bufferize --finalizing-bufferize  \
// RUN:   --convert-vector-to-llvm --convert-std-to-llvm | \
// RUN: TENSOR0="%mlir_integration_test_dir/data/wide.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

//
// Use descriptive names for opaque pointers.
//
!Filename     = type !llvm.ptr<i8>
!SparseTensor = type !llvm.ptr<i8>

#matvec = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (j)>,   // b
    affine_map<(i,j) -> (i)>    // x (out)
  ],
  sparse = [
    [ "D", "S" ], // A
    [ "D"      ], // b
    [ "D"      ]  // x
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "X(i) += A(i,j) * B(j)"
}

//
// Integration test that lowers a kernel annotated as sparse to
// actual sparse code, initializes a matching sparse storage scheme
// from file, and runs the resulting code with the JIT compiler.
//
module {
  //
  // The kernel expressed as an annotated Linalg op. The kernel multiplies
  // a sparse matrix A with a dense vector b into a dense vector x.
  //
  func @kernel_matvec(%argA: !SparseTensor,
                      %argb: tensor<?xi32>,
                      %argx: tensor<?xi32>) -> tensor<?xi32> {
    %arga = linalg.sparse_tensor %argA : !SparseTensor to tensor<?x?xi32>
    %0 = linalg.generic #matvec
      ins(%arga, %argb: tensor<?x?xi32>, tensor<?xi32>)
      outs(%argx: tensor<?xi32>) {
      ^bb(%a: i32, %b: i32, %x: i32):
        %0 = muli %a, %b : i32
        %1 = addi %x, %0 : i32
        linalg.yield %1 : i32
    } -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  //
  // Runtime support library that is called directly from here.
  //
  func private @getTensorFilename(index) -> (!Filename)
  func private @newSparseTensor(!Filename, memref<?xi1>, index, index, index) -> (!SparseTensor)
  func private @delSparseTensor(!SparseTensor) -> ()

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func @entry() {
    %i0 = constant 0 : i32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c4 = constant 4 : index
    %c256 = constant 256 : index

    // Mark inner dimension of the matrix as sparse and encode the
    // storage scheme types (this must match the metadata in the
    // alias above and compiler switches). In this case, we test
    // that 8-bit indices and pointers work correctly on a matrix
    // with i32 elements.
    %annotations = memref.alloc(%c2) : memref<?xi1>
    %sparse = constant true
    %dense = constant false
    memref.store %dense, %annotations[%c0] : memref<?xi1>
    memref.store %sparse, %annotations[%c1] : memref<?xi1>
    %u8 = constant 4 : index
    %i32 = constant 3 : index

    // Read the sparse matrix from file, construct sparse storage.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %a = call @newSparseTensor(%fileName, %annotations, %u8, %u8, %i32)
      : (!Filename, memref<?xi1>, index, index, index) -> (!SparseTensor)

    // Initialize dense vectors.
    %bdata = memref.alloc(%c256) : memref<?xi32>
    %xdata = memref.alloc(%c4) : memref<?xi32>
    scf.for %i = %c0 to %c256 step %c1 {
      %k = addi %i, %c1 : index
      %j = index_cast %k : index to i32
      memref.store %j, %bdata[%i] : memref<?xi32>
    }
    scf.for %i = %c0 to %c4 step %c1 {
      memref.store %i0, %xdata[%i] : memref<?xi32>
    }
    %b = memref.tensor_load %bdata : memref<?xi32>
    %x = memref.tensor_load %xdata : memref<?xi32>

    // Call kernel.
    %0 = call @kernel_matvec(%a, %b, %x)
      : (!SparseTensor, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

    // Print the result for verification.
    //
    // CHECK: ( 889, 1514, -21, -3431 )
    //
    %m = memref.buffer_cast %0 : memref<?xi32>
    %v = vector.transfer_read %m[%c0], %i0: memref<?xi32>, vector<4xi32>
    vector.print %v : vector<4xi32>

    // Release the resources.
    call @delSparseTensor(%a) : (!SparseTensor) -> ()
    memref.dealloc %bdata : memref<?xi32>
    memref.dealloc %xdata : memref<?xi32>

    return
  }
}
