// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils | \
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
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#DCSR  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  posWidth = 8,
  crdWidth = 8
}>

#DCSC  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  posWidth = 64,
  crdWidth = 64
}>

#CSC  = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  posWidth = 16,
  crdWidth = 32
}>

//
// Integration test that tests conversions between sparse tensors,
// where the position and index sizes in the overhead storage change
// in addition to layout.
//
module {

  //
  // Helper method to print values and indices arrays. The transfer actually
  // reads more than required to verify size of buffer as well.
  //
  func.func @dumpf64(%arg0: memref<?xf64>) {
    %c = arith.constant 0 : index
    %d = arith.constant 0.0 : f64
    %0 = vector.transfer_read %arg0[%c], %d: memref<?xf64>, vector<8xf64>
    vector.print %0 : vector<8xf64>
    return
  }
  func.func @dumpi08(%arg0: memref<?xi8>) {
    %c = arith.constant 0 : index
    %d = arith.constant 0 : i8
    %0 = vector.transfer_read %arg0[%c], %d: memref<?xi8>, vector<8xi8>
    vector.print %0 : vector<8xi8>
    return
  }
  func.func @dumpi32(%arg0: memref<?xi32>) {
    %c = arith.constant 0 : index
    %d = arith.constant 0 : i32
    %0 = vector.transfer_read %arg0[%c], %d: memref<?xi32>, vector<8xi32>
    vector.print %0 : vector<8xi32>
    return
  }
  func.func @dumpi64(%arg0: memref<?xi64>) {
    %c = arith.constant 0 : index
    %d = arith.constant 0 : i64
    %0 = vector.transfer_read %arg0[%c], %d: memref<?xi64>, vector<8xi64>
    vector.print %0 : vector<8xi64>
    return
  }

  func.func @entry() {
    %c1 = arith.constant 1 : index
    %t1 = arith.constant sparse<
      [ [0,0], [0,1], [0,63], [1,0], [1,1], [31,0], [31,63] ],
       [ 1.0,   2.0,   3.0,    4.0,   5.0,   6.0,    7.0 ]> : tensor<32x64xf64>
    %t2 = tensor.cast %t1 : tensor<32x64xf64> to tensor<?x?xf64>

    // Dense to sparse.
    %1 = sparse_tensor.convert %t1 : tensor<32x64xf64> to tensor<32x64xf64, #DCSR>
    %2 = sparse_tensor.convert %t1 : tensor<32x64xf64> to tensor<32x64xf64, #DCSC>
    %3 = sparse_tensor.convert %t1 : tensor<32x64xf64> to tensor<32x64xf64, #CSC>

    // Sparse to sparse.
    %4 = sparse_tensor.convert %1 : tensor<32x64xf64, #DCSR> to tensor<32x64xf64, #DCSC>
    %5 = sparse_tensor.convert %2 : tensor<32x64xf64, #DCSC> to tensor<32x64xf64, #DCSR>
    %6 = sparse_tensor.convert %3 : tensor<32x64xf64, #CSC>  to tensor<32x64xf64, #DCSR>

    //
    // All proper row-/column-wise?
    //
    // CHECK:      ( 1, 2, 3, 4, 5, 6, 7, 0 )
    // CHECK-NEXT: ( 1, 4, 6, 2, 5, 3, 7, 0 )
    // CHECK-NEXT: ( 1, 4, 6, 2, 5, 3, 7, 0 )
    // CHECK-NEXT: ( 1, 4, 6, 2, 5, 3, 7, 0 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 0 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 0 )
    //
    %m1 = sparse_tensor.values %1 : tensor<32x64xf64, #DCSR> to memref<?xf64>
    %m2 = sparse_tensor.values %2 : tensor<32x64xf64, #DCSC> to memref<?xf64>
    %m3 = sparse_tensor.values %3 : tensor<32x64xf64, #CSC>  to memref<?xf64>
    %m4 = sparse_tensor.values %4 : tensor<32x64xf64, #DCSC> to memref<?xf64>
    %m5 = sparse_tensor.values %5 : tensor<32x64xf64, #DCSR> to memref<?xf64>
    %m6 = sparse_tensor.values %6 : tensor<32x64xf64, #DCSR> to memref<?xf64>
    call @dumpf64(%m1) : (memref<?xf64>) -> ()
    call @dumpf64(%m2) : (memref<?xf64>) -> ()
    call @dumpf64(%m3) : (memref<?xf64>) -> ()
    call @dumpf64(%m4) : (memref<?xf64>) -> ()
    call @dumpf64(%m5) : (memref<?xf64>) -> ()
    call @dumpf64(%m6) : (memref<?xf64>) -> ()

    //
    // Sanity check on indices.
    //
    // CHECK-NEXT: ( 0, 1, 63, 0, 1, 0, 63, 0 )
    // CHECK-NEXT: ( 0, 1, 31, 0, 1, 0, 31, 0 )
    // CHECK-NEXT: ( 0, 1, 31, 0, 1, 0, 31, 0 )
    // CHECK-NEXT: ( 0, 1, 31, 0, 1, 0, 31, 0 )
    // CHECK-NEXT: ( 0, 1, 63, 0, 1, 0, 63, 0 )
    // CHECK-NEXT: ( 0, 1, 63, 0, 1, 0, 63, 0 )
    //
    %i1 = sparse_tensor.coordinates %1 { level = 1 : index } : tensor<32x64xf64, #DCSR> to memref<?xi8>
    %i2 = sparse_tensor.coordinates %2 { level = 1 : index } : tensor<32x64xf64, #DCSC> to memref<?xi64>
    %i3 = sparse_tensor.coordinates %3 { level = 1 : index } : tensor<32x64xf64, #CSC>  to memref<?xi32>
    %i4 = sparse_tensor.coordinates %4 { level = 1 : index } : tensor<32x64xf64, #DCSC> to memref<?xi64>
    %i5 = sparse_tensor.coordinates %5 { level = 1 : index } : tensor<32x64xf64, #DCSR> to memref<?xi8>
    %i6 = sparse_tensor.coordinates %6 { level = 1 : index } : tensor<32x64xf64, #DCSR> to memref<?xi8>
    call @dumpi08(%i1) : (memref<?xi8>)  -> ()
    call @dumpi64(%i2) : (memref<?xi64>) -> ()
    call @dumpi32(%i3) : (memref<?xi32>) -> ()
    call @dumpi64(%i4) : (memref<?xi64>) -> ()
    call @dumpi08(%i5) : (memref<?xi08>) -> ()
    call @dumpi08(%i6) : (memref<?xi08>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %1 : tensor<32x64xf64, #DCSR>
    bufferization.dealloc_tensor %2 : tensor<32x64xf64, #DCSC>
    bufferization.dealloc_tensor %3 : tensor<32x64xf64, #CSC>
    bufferization.dealloc_tensor %4 : tensor<32x64xf64, #DCSC>
    bufferization.dealloc_tensor %5 : tensor<32x64xf64, #DCSR>
    bufferization.dealloc_tensor %6 : tensor<32x64xf64, #DCSR>

    return
  }
}
