// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true"
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true vl=4 reassociate-fp-reductions=true enable-index-optimizations=true enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext --dlopen=%mlir_lib_dir/libmlir_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#DCSR  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

#DCSC  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
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
