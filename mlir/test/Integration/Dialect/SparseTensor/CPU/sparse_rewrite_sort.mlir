// RUN: mlir-opt %s --sparse-compiler=enable-runtime-library=false | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

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
    %i4 = arith.constant 4 : index
    %i5 = arith.constant 5 : index

    // Prepare a buffer.
    %x0s = memref.alloc() : memref<5xi32>
    %x0 = memref.cast %x0s : memref<5xi32> to memref<?xi32>
    call @storeValuesTo(%x0, %c10, %c2, %c0, %c5, %c1)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()

    // Sort 0 elements.
    // CHECK: ( 10, 2, 0, 5, 1 )
    sparse_tensor.sort %i0, %x0 : memref<?xi32>
    %x0v0 = vector.transfer_read %x0[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %x0v0 : vector<5xi32>
    // Stable sort.
    // CHECK: ( 10, 2, 0, 5, 1 )
    sparse_tensor.sort stable %i0, %x0 : memref<?xi32>
    %x0v0s = vector.transfer_read %x0[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %x0v0s : vector<5xi32>

    // Sort the first 4 elements, with the last valid value untouched.
    // CHECK: ( 0, 2, 5, 10, 1 )
    sparse_tensor.sort %i4, %x0 : memref<?xi32>
    %x0v1 = vector.transfer_read %x0[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %x0v1 : vector<5xi32>
    // Stable sort.
    // CHECK: ( 0, 2, 5, 10, 1 )
    call @storeValuesTo(%x0, %c10, %c2, %c0, %c5, %c1)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    sparse_tensor.sort stable %i4, %x0 : memref<?xi32>
    %x0v1s = vector.transfer_read %x0[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %x0v1s : vector<5xi32>

    // Prepare more buffers of different dimensions.
    %x1s = memref.alloc() : memref<10xi32>
    %x1 = memref.cast %x1s : memref<10xi32> to memref<?xi32>
    %x2s = memref.alloc() : memref<6xi32>
    %x2 = memref.cast %x2s : memref<6xi32> to memref<?xi32>
    %y0s = memref.alloc() : memref<7xi32>
    %y0 = memref.cast %y0s : memref<7xi32> to memref<?xi32>

    // Sort "parallel arrays".
    // CHECK: ( 1, 1, 2, 5, 10 )
    // CHECK: ( 3, 3, 1, 10, 1 )
    // CHECK: ( 9, 9, 4, 7, 2 )
    // CHECK: ( 7, 8, 10, 9, 6 )
    call @storeValuesTo(%x0, %c10, %c2, %c1, %c5, %c1)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%x1, %c1, %c1, %c3, %c10, %c3)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%x2, %c2, %c4, %c9, %c7, %c9)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%y0, %c6, %c10, %c8, %c9, %c7)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    sparse_tensor.sort %i5, %x0, %x1, %x2 jointly %y0
      : memref<?xi32>, memref<?xi32>, memref<?xi32> jointly memref<?xi32>
    %x0v2 = vector.transfer_read %x0[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %x0v2 : vector<5xi32>
    %x1v = vector.transfer_read %x1[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %x1v : vector<5xi32>
    %x2v = vector.transfer_read %x2[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %x2v : vector<5xi32>
    %y0v = vector.transfer_read %y0[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %y0v : vector<5xi32>
    // Stable sort.
    // CHECK: ( 1, 1, 2, 5, 10 )
    // CHECK: ( 3, 3, 1, 10, 1 )
    // CHECK: ( 9, 9, 4, 7, 2 )
    // CHECK: ( 8, 7, 10, 9, 6 )
    call @storeValuesTo(%x0, %c10, %c2, %c1, %c5, %c1)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%x1, %c1, %c1, %c3, %c10, %c3)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%x2, %c2, %c4, %c9, %c7, %c9)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%y0, %c6, %c10, %c8, %c9, %c7)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    sparse_tensor.sort stable %i5, %x0, %x1, %x2 jointly %y0
      : memref<?xi32>, memref<?xi32>, memref<?xi32> jointly memref<?xi32>
    %x0v2s = vector.transfer_read %x0[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %x0v2s : vector<5xi32>
    %x1vs = vector.transfer_read %x1[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %x1vs : vector<5xi32>
    %x2vs = vector.transfer_read %x2[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %x2vs : vector<5xi32>
    %y0vs = vector.transfer_read %y0[%i0], %c100: memref<?xi32>, vector<5xi32>
    vector.print %y0vs : vector<5xi32>

    // Release the buffers.
    memref.dealloc %x0 : memref<?xi32>
    memref.dealloc %x1 : memref<?xi32>
    memref.dealloc %x2 : memref<?xi32>
    memref.dealloc %y0 : memref<?xi32>
    return
  }
}
