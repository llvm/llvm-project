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
// DEFINE: %{run_opts} = -e entry -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#ID_MAP = affine_map<(d0, d1, d2) -> (d1, d2, d0)>

module {
  // Stores 5 values to the memref buffer.
  func.func @storeValuesTo(%b: memref<?xi32>, %v0: i32, %v1: i32, %v2: i32,
    %v3: i32, %v4: i32) -> () {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index
    %i4 = arith.constant 4 : index
    memref.store %v0, %b[%i0] : memref<?xi32>
    memref.store %v1, %b[%i1] : memref<?xi32>
    memref.store %v2, %b[%i2] : memref<?xi32>
    memref.store %v3, %b[%i3] : memref<?xi32>
    memref.store %v4, %b[%i4] : memref<?xi32>
    return
  }

  // Stores 5 values to the memref buffer.
  func.func @storeValuesToStrided(%b: memref<?xi32, strided<[4], offset: ?>>, %v0: i32, %v1: i32, %v2: i32,
    %v3: i32, %v4: i32) -> () {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index
    %i4 = arith.constant 4 : index
    memref.store %v0, %b[%i0] : memref<?xi32, strided<[4], offset: ?>>
    memref.store %v1, %b[%i1] : memref<?xi32, strided<[4], offset: ?>>
    memref.store %v2, %b[%i2] : memref<?xi32, strided<[4], offset: ?>>
    memref.store %v3, %b[%i3] : memref<?xi32, strided<[4], offset: ?>>
    memref.store %v4, %b[%i4] : memref<?xi32, strided<[4], offset: ?>>
    return
  }

  // The main driver.
  func.func @entry() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c5 = arith.constant 5 : i32
    %c6 = arith.constant 6 : i32
    %c7 = arith.constant 7 : i32
    %c8 = arith.constant 8 : i32
    %c9 = arith.constant 9 : i32
    %c10 = arith.constant 10 : i32
    %c100 = arith.constant 100 : i32

    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index
    %i4 = arith.constant 4 : index
    %i5 = arith.constant 5 : index

    // Prepare a buffer for x0, x1, x2, y0 and a buffer for y1.
    %xys = memref.alloc() : memref<20xi32>
    %xy = memref.cast %xys : memref<20xi32> to memref<?xi32>
    %x0 = memref.subview %xy[%i0][%i5][%i4] : memref<?xi32> to memref<?xi32, strided<[4], offset: ?>>
    %x1 = memref.subview %xy[%i1][%i5][%i4] : memref<?xi32> to memref<?xi32, strided<[4], offset: ?>>
    %x2 = memref.subview %xy[%i2][%i5][%i4] : memref<?xi32> to memref<?xi32, strided<[4], offset: ?>>
    %y0 = memref.subview %xy[%i3][%i5][%i4] : memref<?xi32> to memref<?xi32, strided<[4], offset: ?>>
    %y1s = memref.alloc() : memref<7xi32>
    %y1 = memref.cast %y1s : memref<7xi32> to memref<?xi32>

    // Sort "parallel arrays".
    // CHECK: ( 1, 1, 2, 5, 10 )
    // CHECK: ( 9, 9, 4, 7, 2 )
    // CHECK: ( 3, 3, 1, 10, 1 )
    // CHECK: ( 7, 8, 10, 9, 6 )
    // CHECK: ( 7, 4, 7, 9, 5 )
    call @storeValuesToStrided(%x0, %c1, %c1, %c3, %c10, %c3)
      : (memref<?xi32, strided<[4], offset: ?>>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesToStrided(%x1, %c10, %c2, %c1, %c5, %c1)
      : (memref<?xi32, strided<[4], offset: ?>>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesToStrided(%x2, %c2, %c4, %c9, %c7, %c9)
      : (memref<?xi32, strided<[4], offset: ?>>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesToStrided(%y0, %c6, %c10, %c8, %c9, %c7)
      : (memref<?xi32, strided<[4], offset: ?>>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%y1, %c5, %c7, %c4, %c9, %c7)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    sparse_tensor.sort quick_sort %i5, %xy jointly %y1 {perm_map = #ID_MAP, ny = 1 : index}
      : memref<?xi32> jointly memref<?xi32>
    // Dumps memory in the same order as the perm_map such that the output is ordered.
    %x1v = vector.transfer_read %x1[%i0], %c100: memref<?xi32, strided<[4], offset: ?>>, vector<5xi32>
    vector.print %x1v : vector<5xi32>
    %x2v = vector.transfer_read %x2[%i0], %c100: memref<?xi32, strided<[4], offset: ?>>, vector<5xi32>
    vector.print %x2v : vector<5xi32>
    %x0v = vector.transfer_read %x0[%i0], %c100: memref<?xi32, strided<[4], offset: ?>>, vector<5xi32>
    vector.print %x0v : vector<5xi32>
    %y0v = vector.transfer_read %y0[%i0], %c100: memref<?xi32, strided<[4], offset: ?>>, vector<5xi32>
    vector.print %y0v : vector<5xi32>
    %y1v = vector.transfer_read %y1[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %y1v : vector<5xi32>
    // Stable sort.
    // CHECK: ( 1, 1, 2, 5, 10 )
    // CHECK: ( 9, 9, 4, 7, 2 )
    // CHECK: ( 3, 3, 1, 10, 1 )
    // CHECK: ( 8, 7, 10, 9, 6 )
    // CHECK: ( 4, 7, 7, 9, 5 )
    call @storeValuesToStrided(%x0, %c1, %c1, %c3, %c10, %c3)
      : (memref<?xi32, strided<[4], offset: ?>>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesToStrided(%x1, %c10, %c2, %c1, %c5, %c1)
      : (memref<?xi32, strided<[4], offset: ?>>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesToStrided(%x2, %c2, %c4, %c9, %c7, %c9)
      : (memref<?xi32, strided<[4], offset: ?>>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesToStrided(%y0, %c6, %c10, %c8, %c9, %c7)
      : (memref<?xi32, strided<[4], offset: ?>>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%y1, %c5, %c7, %c4, %c9, %c7)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    sparse_tensor.sort insertion_sort_stable %i5, %xy jointly %y1 {perm_map = #ID_MAP, ny = 1 : index}
      : memref<?xi32> jointly memref<?xi32>
    %x1v2 = vector.transfer_read %x1[%i0], %c100: memref<?xi32, strided<[4], offset: ?>>, vector<5xi32>
    vector.print %x1v2 : vector<5xi32>
    %x2v2 = vector.transfer_read %x2[%i0], %c100: memref<?xi32, strided<[4], offset: ?>>, vector<5xi32>
    vector.print %x2v2 : vector<5xi32>
    %x0v2 = vector.transfer_read %x0[%i0], %c100: memref<?xi32, strided<[4], offset: ?>>, vector<5xi32>
    vector.print %x0v2 : vector<5xi32>
    %y0v2 = vector.transfer_read %y0[%i0], %c100: memref<?xi32, strided<[4], offset: ?>>, vector<5xi32>
    vector.print %y0v2 : vector<5xi32>
    %y1v2 = vector.transfer_read %y1[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %y1v2 : vector<5xi32>
    // Heap sort.
    // CHECK: ( 1, 1, 2, 5, 10 )
    // CHECK: ( 9, 9, 4, 7, 2 )
    // CHECK: ( 3, 3, 1, 10, 1 )
    // CHECK: ( 7, 8, 10, 9, 6 )
    // CHECK: ( 7, 4, 7, 9, 5 )
    call @storeValuesToStrided(%x0, %c1, %c1, %c3, %c10, %c3)
      : (memref<?xi32, strided<[4], offset: ?>>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesToStrided(%x1, %c10, %c2, %c1, %c5, %c1)
      : (memref<?xi32, strided<[4], offset: ?>>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesToStrided(%x2, %c2, %c4, %c9, %c7, %c9)
      : (memref<?xi32, strided<[4], offset: ?>>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesToStrided(%y0, %c6, %c10, %c8, %c9, %c7)
      : (memref<?xi32, strided<[4], offset: ?>>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%y1, %c5, %c7, %c4, %c9, %c7)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    sparse_tensor.sort heap_sort %i5, %xy jointly %y1 {perm_map = #ID_MAP, ny = 1 : index}
      : memref<?xi32> jointly memref<?xi32>
    %x1v3 = vector.transfer_read %x1[%i0], %c100: memref<?xi32, strided<[4], offset: ?>>, vector<5xi32>
    vector.print %x1v3 : vector<5xi32>
    %x2v3 = vector.transfer_read %x2[%i0], %c100: memref<?xi32, strided<[4], offset: ?>>, vector<5xi32>
    vector.print %x2v3 : vector<5xi32>
    %x0v3 = vector.transfer_read %x0[%i0], %c100: memref<?xi32, strided<[4], offset: ?>>, vector<5xi32>
    vector.print %x0v3 : vector<5xi32>
    %y0v3 = vector.transfer_read %y0[%i0], %c100: memref<?xi32, strided<[4], offset: ?>>, vector<5xi32>
    vector.print %y0v3 : vector<5xi32>
    %y1v3 = vector.transfer_read %y1[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %y1v3 : vector<5xi32>

    // Release the buffers.
    memref.dealloc %xy : memref<?xi32>
    memref.dealloc %y1 : memref<?xi32>
    return
  }
}
