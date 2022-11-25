// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{command} = mlir-opt %s --sparse-compiler=%{option} | \
// DEFINE: TENSOR0="%mlir_src_dir/test/Integration/data/test_symmetric_complex.mtx" \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{command}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = enable-runtime-library=false
// RUN: %{command}

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

//
// Integration test that lowers a kernel annotated as sparse to
// actual sparse code, initializes a matching sparse storage scheme
// from file, and runs the resulting code with the JIT compiler.
//
module {
  //
  // A kernel that sum-reduces a matrix to a single scalar.
  //
  func.func @kernel_sum_reduce(%arga: tensor<?x?xcomplex<f64>, #SparseMatrix>,
                               %argx: tensor<complex<f64>>) -> tensor<complex<f64>> {
    %0 = linalg.generic #trait_sum_reduce
      ins(%arga: tensor<?x?xcomplex<f64>, #SparseMatrix>)
      outs(%argx: tensor<complex<f64>>) {
      ^bb(%a: complex<f64>, %x: complex<f64>):
        %0 = complex.add %x, %a : complex<f64>
        linalg.yield %0 : complex<f64>
    } -> tensor<complex<f64>>
    return %0 : tensor<complex<f64>>
  }

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func.func @entry() {
    //%d0 = arith.constant 0.0 : complex<f64>
    %d0 = complex.constant [0.0 : f64, 0.0 : f64] : complex<f64>
    %c0 = arith.constant 0 : index

    // Setup memory for a single reduction scalar,
    // initialized to zero.
    // TODO: tensor.from_elements does not support complex.
    %alloc = bufferization.alloc_tensor() : tensor<complex<f64>>
    %x = tensor.insert %d0 into %alloc[] : tensor<complex<f64>>

    // Read the sparse matrix from file, construct sparse storage.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %a = sparse_tensor.new expand_symmetry %fileName : !Filename to tensor<?x?xcomplex<f64>, #SparseMatrix>

    // Call the kernel.
    %0 = call @kernel_sum_reduce(%a, %x)
      : (tensor<?x?xcomplex<f64>, #SparseMatrix>, tensor<complex<f64>>) -> tensor<complex<f64>>

    // Print the result for verification.
    //
    // CHECK: 30.2
    // CHECK-NEXT: 22.2
    //
    %v = tensor.extract %0[] : tensor<complex<f64>>
    %real = complex.re %v : complex<f64>
    %imag = complex.im %v : complex<f64>
    vector.print %real : f64
    vector.print %imag : f64

    // Release the resources.
    bufferization.dealloc_tensor %a : tensor<?x?xcomplex<f64>, #SparseMatrix>

    return
  }
}
