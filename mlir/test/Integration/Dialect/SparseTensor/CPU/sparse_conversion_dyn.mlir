//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparse_compiler_opts} = enable-runtime-library=true
// DEFINE: %{sparse_compiler_opts_sve} = enable-arm-sve=true %{sparse_compiler_opts}
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_opts} = -e entry -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false enable-buffer-initialization=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#DCSR  = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#DCSC  = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : compressed, d0 : compressed)
}>

//
// Integration test that tests conversions between sparse tensors,
// where the dynamic sizes of the shape of the enveloping tensor
// may change (the actual underlying sizes obviously never change).
//
module {

  func.func private @printMemref1dF64(%ptr : memref<?xf64>) attributes { llvm.emit_c_interface }

  //
  // Helper method to print values array. The transfer actually
  // reads more than required to verify size of buffer as well.
  //
  func.func @dump(%arg0: memref<?xf64>) {
    call @printMemref1dF64(%arg0) : (memref<?xf64>) -> ()
    return
  }

  func.func @entry() {
    %t1 = arith.constant sparse<
      [ [0,0], [0,1], [0,63], [1,0], [1,1], [31,0], [31,63] ],
        [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 ]> : tensor<32x64xf64>
    %t2 = tensor.cast %t1 : tensor<32x64xf64> to tensor<?x?xf64>

    // Four dense to sparse conversions.
    %1 = sparse_tensor.convert %t1 : tensor<32x64xf64> to tensor<?x?xf64, #DCSR>
    %2 = sparse_tensor.convert %t1 : tensor<32x64xf64> to tensor<?x?xf64, #DCSC>
    %3 = sparse_tensor.convert %t2 : tensor<?x?xf64> to tensor<?x?xf64, #DCSR>
    %4 = sparse_tensor.convert %t2 : tensor<?x?xf64> to tensor<?x?xf64, #DCSC>

    // Two cross conversions.
    %5 = sparse_tensor.convert %3 : tensor<?x?xf64, #DCSR> to tensor<?x?xf64, #DCSC>
    %6 = sparse_tensor.convert %4 : tensor<?x?xf64, #DCSC> to tensor<?x?xf64, #DCSR>

//
    // Check number_of_entries.
    //
    // CHECK-COUNT-6: 7
    %n1 = sparse_tensor.number_of_entries %1 : tensor<?x?xf64, #DCSR>
    %n2 = sparse_tensor.number_of_entries %2 : tensor<?x?xf64, #DCSC>
    %n3 = sparse_tensor.number_of_entries %3 : tensor<?x?xf64, #DCSR>
    %n4 = sparse_tensor.number_of_entries %4 : tensor<?x?xf64, #DCSC>
    %n5 = sparse_tensor.number_of_entries %5 : tensor<?x?xf64, #DCSC>
    %n6 = sparse_tensor.number_of_entries %6 : tensor<?x?xf64, #DCSR>
    vector.print %n1 : index
    vector.print %n2 : index
    vector.print %n3 : index
    vector.print %n4 : index
    vector.print %n5 : index
    vector.print %n6 : index

    //
    // All proper row-/column-wise?
    //
    // CHECK: [1,  2,  3,  4,  5,  6,  7
    // CHECK: [1,  4,  6,  2,  5,  3,  7
    // CHECK: [1,  2,  3,  4,  5,  6,  7
    // CHECK: [1,  4,  6,  2,  5,  3,  7
    // CHECK: [1,  4,  6,  2,  5,  3,  7
    // CHECK: [1,  2,  3,  4,  5,  6,  7
    //
    %m1 = sparse_tensor.values %1 : tensor<?x?xf64, #DCSR> to memref<?xf64>
    %m2 = sparse_tensor.values %2 : tensor<?x?xf64, #DCSC> to memref<?xf64>
    %m3 = sparse_tensor.values %3 : tensor<?x?xf64, #DCSR> to memref<?xf64>
    %m4 = sparse_tensor.values %4 : tensor<?x?xf64, #DCSC> to memref<?xf64>
    %m5 = sparse_tensor.values %5 : tensor<?x?xf64, #DCSC> to memref<?xf64>
    %m6 = sparse_tensor.values %6 : tensor<?x?xf64, #DCSR> to memref<?xf64>
    call @dump(%m1) : (memref<?xf64>) -> ()
    call @dump(%m2) : (memref<?xf64>) -> ()
    call @dump(%m3) : (memref<?xf64>) -> ()
    call @dump(%m4) : (memref<?xf64>) -> ()
    call @dump(%m5) : (memref<?xf64>) -> ()
    call @dump(%m6) : (memref<?xf64>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %1 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %2 : tensor<?x?xf64, #DCSC>
    bufferization.dealloc_tensor %3 : tensor<?x?xf64, #DCSR>
    bufferization.dealloc_tensor %4 : tensor<?x?xf64, #DCSC>
    bufferization.dealloc_tensor %5 : tensor<?x?xf64, #DCSC>
    bufferization.dealloc_tensor %6 : tensor<?x?xf64, #DCSR>

    return
  }
}
