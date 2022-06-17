// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

!Filename = !llvm.ptr<i8>

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

#trait_sum_reduce = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> ()>     // x (out)
  ],
  iterator_types = ["reduction", "reduction"],
  doc = "x += A(i,j)"
}

module {
  //
  // A kernel that sum-reduces a matrix to a single scalar.
  //
  func.func @kernel_sum_reduce(%arga: tensor<?x?xbf16, #SparseMatrix>,
                          %argx: tensor<bf16> {linalg.inplaceable = true}) -> tensor<bf16> {
    %0 = linalg.generic #trait_sum_reduce
      ins(%arga: tensor<?x?xbf16, #SparseMatrix>)
      outs(%argx: tensor<bf16>) {
      ^bb(%a: bf16, %x: bf16):
        %0 = arith.addf %x, %a : bf16
        linalg.yield %0 : bf16
    } -> tensor<bf16>
    return %0 : tensor<bf16>
  }

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func.func @entry() {
    // Setup input sparse matrix from compressed constant.
    %d = arith.constant dense <[
       [ 1.1,  1.2,  0.0,  1.4 ],
       [ 0.0,  0.0,  0.0,  0.0 ],
       [ 3.1,  0.0,  3.3,  3.4 ]
    ]> : tensor<3x4xbf16>
    %a = sparse_tensor.convert %d : tensor<3x4xbf16> to tensor<?x?xbf16, #SparseMatrix>

    %d0 = arith.constant 0.0 : bf16
    // Setup memory for a single reduction scalar,
    // initialized to zero.
    %xdata = memref.alloc() : memref<bf16>
    memref.store %d0, %xdata[] : memref<bf16>
    %x = bufferization.to_tensor %xdata : memref<bf16>

    // Call the kernel.
    %0 = call @kernel_sum_reduce(%a, %x)
      : (tensor<?x?xbf16, #SparseMatrix>, tensor<bf16>) -> tensor<bf16>

    // Print the result for verification.
    //
    // CHECK: 13.5
    //
    %m = bufferization.to_memref %0 : memref<bf16>
    %v = memref.load %m[] : memref<bf16>
    %vf = arith.extf %v: bf16 to f32
    vector.print %vf : f32

    // Release the resources.
    memref.dealloc %xdata : memref<bf16>
    sparse_tensor.release %a : tensor<?x?xbf16, #SparseMatrix>

    return
  }
}
