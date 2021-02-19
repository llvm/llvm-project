// RUN: mlir-opt %s \
// RUN:   --test-sparsification="lower ptr-type=2 ind-type=2 fast-output" \
// RUN:   --convert-linalg-to-loops \
// RUN:   --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN:   --std-bufferize --finalizing-bufferize  \
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

#trait_sampled_dense_dense = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j)>,  // S
    affine_map<(i,j,k) -> (i,k)>,  // A
    affine_map<(i,j,k) -> (k,j)>,  // B
    affine_map<(i,j,k) -> (i,j)>   // X (out)
  ],
  sparse = [
    [ "S", "S" ],  // S
    [ "D", "D" ],  // A
    [ "D", "D" ],  // B
    [ "D", "D" ]   // X
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) += S(i,j) SUM_k A(i,k) B(k,j)"
}

//
// Integration test that lowers a kernel annotated as sparse to
// actual sparse code, initializes a matching sparse storage scheme
// from file, and runs the resulting code with the JIT compiler.
//
module {
  //
  // The kernel expressed as an annotated Linalg op. The kernel
  // computes a sampled matrix matrix multiplication.
  //
  func @sampled_dense_dense(%argS: !SparseTensor,
                            %arga: tensor<?x?xf32>,
                            %argb: tensor<?x?xf32>,
                            %argx: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %args = linalg.sparse_tensor %argS : !SparseTensor to tensor<?x?xf32>
    %0 = linalg.generic #trait_sampled_dense_dense
      ins(%args, %arga, %argb: tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%argx: tensor<?x?xf32>) {
        ^bb(%s: f32, %a: f32, %b: f32, %x: f32):
          %0 = mulf %a, %b : f32
          %1 = mulf %s, %0 : f32
          %2 = addf %x, %1 : f32
          linalg.yield %2 : f32
    } -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }

  //
  // Runtime support library that is called directly from here.
  //
  func private @getTensorFilename(index) -> (!Filename)
  func private @newSparseTensor(!Filename, memref<?xi1>, index, index, index) -> (!SparseTensor)
  func private @delSparseTensor(!SparseTensor) -> ()
  func private @print_memref_f32(%ptr : tensor<*xf32>)

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func @entry() {
    %d0 = constant 0.0 : f32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c5 = constant 5 : index
    %c10 = constant 10 : index

    // Mark both dimensions of the matrix as sparse and encode the
    // storage scheme types (this must match the metadata in the
    // trait and compiler switches).
    %annotations = alloc(%c2) : memref<?xi1>
    %sparse = constant true
    store %sparse, %annotations[%c0] : memref<?xi1>
    store %sparse, %annotations[%c1] : memref<?xi1>
    %i32 = constant 3 : index
    %f32 = constant 1 : index

    // Setup memory for the dense matrices and initialize.
    %adata = alloc(%c5, %c10) : memref<?x?xf32>
    %bdata = alloc(%c10, %c5) : memref<?x?xf32>
    %xdata = alloc(%c5,  %c5) : memref<?x?xf32>
    scf.for %i = %c0 to %c5 step %c1 {
      scf.for %j = %c0 to %c5 step %c1 {
        store %d0, %xdata[%i, %j] : memref<?x?xf32>
      }
      %p = addi %i, %c1 : index
      %q = index_cast %p : index to i32
      %d = sitofp %q : i32 to f32
      scf.for %j = %c0 to %c10 step %c1 {
        store %d, %adata[%i, %j] : memref<?x?xf32>
        store %d, %bdata[%j, %i] : memref<?x?xf32>
      }
    }
    %a = tensor_load %adata : memref<?x?xf32>
    %b = tensor_load %bdata : memref<?x?xf32>
    %x = tensor_load %xdata : memref<?x?xf32>

    // Read the sparse matrix from file, construct sparse storage
    // according to <sparse,sparse> in memory, and call the kernel.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %s = call @newSparseTensor(%fileName, %annotations, %i32, %i32, %f32)
      : (!Filename, memref<?xi1>, index, index, index) -> (!SparseTensor)
    %0 = call @sampled_dense_dense(%s, %a, %b, %x)
       : (!SparseTensor, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

    // Print the result for verification.
    //
    // CHECK: ( 10, 0, 0, 56, 0 )
    // CHECK: ( 0, 80, 0, 0, 250 )
    // CHECK: ( 0, 0, 270, 0, 0 )
    // CHECK: ( 164, 0, 0, 640, 0 )
    // CHECK: ( 0, 520, 0, 0, 1250 )
    //
    %r = tensor_to_memref %0 : memref<?x?xf32>
    scf.for %i = %c0 to %c5 step %c1 {
      %v = vector.transfer_read %r[%i, %c0], %d0: memref<?x?xf32>, vector<5xf32>
      vector.print %v : vector<5xf32>
    }

    // Release the resources.
    call @delSparseTensor(%s) : (!SparseTensor) -> ()
    dealloc %adata : memref<?x?xf32>
    dealloc %bdata : memref<?x?xf32>
    dealloc %xdata : memref<?x?xf32>

    return
  }
}
