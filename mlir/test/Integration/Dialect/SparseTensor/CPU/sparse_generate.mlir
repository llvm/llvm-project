//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparsifier_opts} = enable-runtime-library=true
// DEFINE: %{sparsifier_opts_sve} = enable-arm-sve=true %{sparsifier_opts}
// DEFINE: %{compile} = mlir-opt %s --sparsifier="%{sparsifier_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparsifier="%{sparsifier_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_libs_sve} = -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils
// DEFINE: %{run_opts} = -e main -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs_sve}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s

//
// Integration test that generates a tensor with specified sparsity level.
//

!Generator = !llvm.ptr
!Array = !llvm.ptr

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

module {
  func.func private @rtsrand(index) -> (!Generator)
  func.func private @rtrand(!Generator, index) -> (index)
  func.func private @rtdrand(!Generator) -> ()
  func.func private @shuffle(memref<?xi64>, !Generator) -> () attributes { llvm.emit_c_interface }

  //
  // Main driver.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %f0 = arith.constant 0.0 : f64
    %c99 = arith.constant 99 : index
    %c100 = arith.constant 100 : index

    // Set up input size and sparsity level.
    %size = arith.constant 50 : index
    %sparsity = arith.constant 90 : index
    %zeros = arith.muli %size, %sparsity : index
    %nz = arith.floordivsi %zeros, %c100 : index
    %nse = arith.subi %size, %nz : index

    // Set up an empty vector.
    %empty = tensor.empty(%size) : tensor<?xf64>
    %zero_vec = linalg.fill ins(%f0 : f64) outs(%empty : tensor<?xf64>) -> tensor<?xf64>

    // Generate shuffled indices in the range of [0, %size).
    %array = memref.alloc (%size) : memref<?xi64>
    %g = func.call @rtsrand(%c0) : (index) ->(!Generator)
    func.call @shuffle(%array, %g) : (memref<?xi64>, !Generator) -> ()

    // Iterate through the number of nse indices to insert values.
    %output = scf.for %iv = %c0 to %nse step %c1 iter_args(%iter = %zero_vec) -> tensor<?xf64> {
      // Fetch the index to insert value from shuffled index array.
      %val = memref.load %array[%iv] : memref<?xi64>
      %idx = arith.index_cast %val : i64 to index
      // Generate a random number from 1 to 100.
      %ri0 = func.call @rtrand(%g, %c99) : (!Generator, index) -> (index)
      %ri1 = arith.addi %ri0, %c1 : index
      %r0 = arith.index_cast %ri1 : index to i64
      %fr = arith.uitofp %r0 : i64 to f64
      // Insert the random number to current index.
      %out = tensor.insert %fr into %iter[%idx] : tensor<?xf64>
      scf.yield %out : tensor<?xf64>
    }

    %sv = sparse_tensor.convert %output : tensor<?xf64> to tensor<?xf64, #SparseVector>
    %n0 = sparse_tensor.number_of_entries %sv : tensor<?xf64, #SparseVector>

    // Print the number of non-zeros for verification
    // as shuffle may generate different numbers.
    //
    // CHECK: 5
    vector.print %n0 : index

    // Release the resources.
    bufferization.dealloc_tensor %sv : tensor<?xf64, #SparseVector>
    bufferization.dealloc_tensor %empty : tensor<?xf64>
    memref.dealloc %array : memref<?xi64>
    func.call @rtdrand(%g) : (!Generator) -> ()

    return
  }
}
