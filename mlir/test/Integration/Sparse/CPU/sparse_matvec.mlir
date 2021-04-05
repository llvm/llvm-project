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
                      %argb: tensor<?xf32>,
                      %argx: tensor<?xf32>) -> tensor<?xf32> {
    %arga = linalg.sparse_tensor %argA : !SparseTensor to tensor<?x?xf32>
    %0 = linalg.generic #matvec
      ins(%arga, %argb: tensor<?x?xf32>, tensor<?xf32>)
      outs(%argx: tensor<?xf32>) {
      ^bb(%a: f32, %b: f32, %x: f32):
        %0 = mulf %a, %b : f32
        %1 = addf %x, %0 : f32
        linalg.yield %1 : f32
    } -> tensor<?xf32>
    return %0 : tensor<?xf32>
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
    %f0 = constant 0.0 : f32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c4 = constant 4 : index
    %c256 = constant 256 : index

    // Mark inner dimension of the matrix as sparse and encode the
    // storage scheme types (this must match the metadata in the
    // alias above and compiler switches). In this case, we test
    // that 8-bit indices and pointers work correctly.
    %annotations = memref.alloc(%c2) : memref<?xi1>
    %sparse = constant true
    %dense = constant false
    memref.store %dense, %annotations[%c0] : memref<?xi1>
    memref.store %sparse, %annotations[%c1] : memref<?xi1>
    %u8 = constant 4 : index
    %f32 = constant 2 : index

    // Read the sparse matrix from file, construct sparse storage.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %a = call @newSparseTensor(%fileName, %annotations, %u8, %u8, %f32)
      : (!Filename, memref<?xi1>, index, index, index) -> (!SparseTensor)

    // Initialize dense vectors.
    %bdata = memref.alloc(%c256) : memref<?xf32>
    %xdata = memref.alloc(%c4) : memref<?xf32>
    scf.for %i = %c0 to %c256 step %c1 {
      %k = addi %i, %c1 : index
      %l = index_cast %k : index to i32
      %f = sitofp %l : i32 to f32
      memref.store %f, %bdata[%i] : memref<?xf32>
    }
    scf.for %i = %c0 to %c4 step %c1 {
      memref.store %f0, %xdata[%i] : memref<?xf32>
    }
    %b = memref.tensor_load %bdata : memref<?xf32>
    %x = memref.tensor_load %xdata : memref<?xf32>

    // Call kernel.
    %0 = call @kernel_matvec(%a, %b, %x)
      : (!SparseTensor, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>

    // Print the result for verification.
    //
    // CHECK: ( 1659, 1534, 21, 18315 )
    //
    %m = memref.buffer_cast %0 : memref<?xf32>
    %v = vector.transfer_read %m[%c0], %f0: memref<?xf32>, vector<4xf32>
    vector.print %v : vector<4xf32>

    // Release the resources.
    call @delSparseTensor(%a) : (!SparseTensor) -> ()
    memref.dealloc %bdata : memref<?xf32>
    memref.dealloc %xdata : memref<?xf32>

    return
  }
}
