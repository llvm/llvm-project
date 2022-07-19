// RUN: mlir-opt %s --sparse-compiler | \
// RUN: TENSOR0="%mlir_integration_test_dir/data/test.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
// Do the same run, but now with SIMDization as well. This should not change the outcome.
//
// RUN: mlir-opt %s \
// RUN:   --sparse-compiler="vectorization-strategy=2 vl=4 enable-simd-index32" | \
// RUN: TENSOR0="%mlir_integration_test_dir/data/test.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

!Filename = !llvm.ptr<i8>

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  pointerBitWidth = 32,
  indexBitWidth = 32
}>

#trait_sampled_dense_dense = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j)>,  // S
    affine_map<(i,j,k) -> (i,k)>,  // A
    affine_map<(i,j,k) -> (k,j)>,  // B
    affine_map<(i,j,k) -> (i,j)>   // X (out)
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
  // A kernel that computes a sampled matrix matrix multiplication.
  //
  func.func @sampled_dense_dense(%args: tensor<?x?xf32, #SparseMatrix>,
                                 %arga: tensor<?x?xf32>,
                                 %argb: tensor<?x?xf32>,
                                 %argx: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.generic #trait_sampled_dense_dense
      ins(%args, %arga, %argb: tensor<?x?xf32, #SparseMatrix>, tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%argx: tensor<?x?xf32>) {
        ^bb(%s: f32, %a: f32, %b: f32, %x: f32):
          %0 = arith.mulf %a, %b : f32
          %1 = arith.mulf %s, %0 : f32
          %2 = arith.addf %x, %1 : f32
          linalg.yield %2 : f32
    } -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func.func @entry() {
    %d0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c10 = arith.constant 10 : index

    // Setup memory for the dense matrices and initialize.
    %adata = memref.alloc(%c5, %c10) : memref<?x?xf32>
    %bdata = memref.alloc(%c10, %c5) : memref<?x?xf32>
    %xdata = memref.alloc(%c5,  %c5) : memref<?x?xf32>
    scf.for %i = %c0 to %c5 step %c1 {
      scf.for %j = %c0 to %c5 step %c1 {
        memref.store %d0, %xdata[%i, %j] : memref<?x?xf32>
      }
      %p = arith.addi %i, %c1 : index
      %q = arith.index_cast %p : index to i32
      %d = arith.sitofp %q : i32 to f32
      scf.for %j = %c0 to %c10 step %c1 {
        memref.store %d, %adata[%i, %j] : memref<?x?xf32>
        memref.store %d, %bdata[%j, %i] : memref<?x?xf32>
      }
    }
    %a = bufferization.to_tensor %adata : memref<?x?xf32>
    %b = bufferization.to_tensor %bdata : memref<?x?xf32>
    %x = bufferization.to_tensor %xdata : memref<?x?xf32>

    // Read the sparse matrix from file, construct sparse storage.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %s = sparse_tensor.new %fileName : !Filename to tensor<?x?xf32, #SparseMatrix>

    // Call the kernel.
    %0 = call @sampled_dense_dense(%s, %a, %b, %x)
       : (tensor<?x?xf32, #SparseMatrix>,
          tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

    // Print the result for verification.
    //
    // CHECK: ( 10, 0, 0, 56, 0 )
    // CHECK: ( 0, 80, 0, 0, 250 )
    // CHECK: ( 0, 0, 270, 0, 0 )
    // CHECK: ( 164, 0, 0, 640, 0 )
    // CHECK: ( 0, 520, 0, 0, 1250 )
    //
    %r = bufferization.to_memref %0 : memref<?x?xf32>
    scf.for %i = %c0 to %c5 step %c1 {
      %v = vector.transfer_read %r[%i, %c0], %d0: memref<?x?xf32>, vector<5xf32>
      vector.print %v : vector<5xf32>
    }

    // Release the resources.
    memref.dealloc %adata : memref<?x?xf32>
    memref.dealloc %bdata : memref<?x?xf32>
    memref.dealloc %xdata : memref<?x?xf32>
    bufferization.dealloc_tensor %s : tensor<?x?xf32, #SparseMatrix>

    return
  }
}
