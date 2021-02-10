// RUN: mlir-opt %s \
// RUN:   --test-sparsification="lower" \
// RUN:   --convert-linalg-to-loops \
// RUN:   --func-bufferize --tensor-constant-bufferize --tensor-bufferize --finalizing-bufferize  \
// RUN:   --convert-scf-to-std --convert-vector-to-llvm --convert-std-to-llvm | \
// RUN: TENSOR0="%mlir_integration_test_dir/data/test.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

//
// Use descriptive names for opaque pointers.
//
!Filename     = type !llvm.ptr<i8>
!SparseTensor = type !llvm.ptr<i8>

#trait_sum_reduce = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> ()>     // x (out)
  ],
  sparse = [
    [ "S", "S" ], // A
    [          ]  // x
  ],
  iterator_types = ["reduction", "reduction"],
  doc = "x += A(i,j)"
}

//
// Integration test that lowers a kernel annotated as sparse to
// actual sparse code, initializes a matching sparse storage scheme
// from file, and runs the resulting code with the JIT compiler.
//
module {
  //
  // The kernel expressed as an annotated Linalg op. The kernel
  // sum reduces a matrix to a single scalar.
  //
  func @kernel_sum_reduce(%argA: !SparseTensor,
                          %argx: tensor<f64>) -> tensor<f64> {
    %arga = linalg.sparse_tensor %argA : !SparseTensor to tensor<?x?xf64>
    %0 = linalg.generic #trait_sum_reduce
      ins(%arga: tensor<?x?xf64>)
      outs(%argx: tensor<f64>) {
      ^bb(%a: f64, %x: f64):
        %0 = addf %x, %a : f64
        linalg.yield %0 : f64
    } -> tensor<f64>
    return %0 : tensor<f64>
  }

  //
  // Runtime support library that is called directly from here.
  //
  func private @getTensorFilename(index) -> (!Filename)
  func private @newSparseTensor(!Filename, memref<?xi1>) -> (!SparseTensor)
  func private @delSparseTensor(!SparseTensor) -> ()
  func private @print_memref_f64(%ptr : tensor<*xf64>)

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func @entry() {
    %d0 = constant 0.0 : f64
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index

    // Mark both dimensions of the matrix as sparse
    // (this must match the annotation in the trait).
    %annotations = alloc(%c2) : memref<?xi1>
    %sparse = constant true
    store %sparse, %annotations[%c0] : memref<?xi1>
    store %sparse, %annotations[%c1] : memref<?xi1>

    // Setup memory for a single reduction scalar,
    // initialized to zero.
    %xdata = alloc() : memref<f64>
    store %d0, %xdata[] : memref<f64>
    %x = tensor_load %xdata : memref<f64>

    // Read the sparse matrix from file, construct sparse storage
    // according to <sparse,sparse> in memory, and call the kernel.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %a = call @newSparseTensor(%fileName, %annotations)
      : (!Filename, memref<?xi1>) -> (!SparseTensor)
    %0 = call @kernel_sum_reduce(%a, %x)
      : (!SparseTensor, tensor<f64>) -> tensor<f64>

    // Print the result for verification.
    //
    // CHECK: 28.2
    //
    %m = tensor_to_memref %0 : memref<f64>
    %v = load %m[] : memref<f64>
    vector.print %v : f64

    // Release the resources.
    call @delSparseTensor(%a) : (!SparseTensor) -> ()
    dealloc %xdata : memref<f64>

    return
  }
}
