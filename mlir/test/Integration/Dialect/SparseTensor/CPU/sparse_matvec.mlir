// RUN: mlir-opt %s --sparse-compiler | \
// RUN: TENSOR0="%mlir_src_dir/test/Integration/data/wide.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
// Do the same run, but now with SIMDization as well. This should not change the outcome.
//
// RUN: mlir-opt %s \
// RUN:   --sparse-compiler="vectorization-strategy=any-storage-inner-loop vl=16 enable-simd-index32" | \
// RUN: TENSOR0="%mlir_src_dir/test/Integration/data/wide.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

!Filename = !llvm.ptr<i8>

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  pointerBitWidth = 8,
  indexBitWidth = 8
}>

#matvec = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (j)>,   // b
    affine_map<(i,j) -> (i)>    // x (out)
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
  // A kernel that multiplies a sparse matrix A with a dense vector b
  // into a dense vector x.
  //
  func.func @kernel_matvec(%arga: tensor<?x?xi32, #SparseMatrix>,
                           %argb: tensor<?xi32>,
                           %argx: tensor<?xi32>)
		      -> tensor<?xi32> {
    %0 = linalg.generic #matvec
      ins(%arga, %argb: tensor<?x?xi32, #SparseMatrix>, tensor<?xi32>)
      outs(%argx: tensor<?xi32>) {
      ^bb(%a: i32, %b: i32, %x: i32):
        %0 = arith.muli %a, %b : i32
        %1 = arith.addi %x, %0 : i32
        linalg.yield %1 : i32
    } -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func.func @entry() {
    %i0 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index

    // Read the sparse matrix from file, construct sparse storage.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %a = sparse_tensor.new %fileName : !Filename to tensor<?x?xi32, #SparseMatrix>

    // Initialize dense vectors.
    %b = tensor.generate %c256 {
    ^bb0(%i : index):
      %k = arith.addi %i, %c1 : index
      %j = arith.index_cast %k : index to i32
      tensor.yield %j : i32
    } : tensor<?xi32>

    %x = tensor.generate %c4 {
      ^bb0(%i : index):
        tensor.yield %i0 : i32
    } : tensor<?xi32>

    // Call kernel.
    %0 = call @kernel_matvec(%a, %b, %x)
      : (tensor<?x?xi32, #SparseMatrix>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

    // Print the result for verification.
    //
    // CHECK: ( 889, 1514, -21, -3431 )
    //
    %v = vector.transfer_read %0[%c0], %i0: tensor<?xi32>, vector<4xi32>
    vector.print %v : vector<4xi32>

    // Release the resources.
    bufferization.dealloc_tensor %a : tensor<?x?xi32, #SparseMatrix>

    // TODO(springerm): auto release!
    bufferization.dealloc_tensor %b : tensor<?xi32>
    bufferization.dealloc_tensor %x : tensor<?xi32>

    return
  }
}
