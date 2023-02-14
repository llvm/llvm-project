// DEFINE: %{option} = enable-runtime-library=false
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext --dlopen=%mlir_lib_dir/libmlir_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

module {
  func.func private @printMemref1dI32(%ptr : memref<?xi32>) attributes { llvm.emit_c_interface }

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
    // Quick sort.
    // CHECK: [10,  2,  0,  5,  1]
    sparse_tensor.sort quick_sort %i0, %x0 : memref<?xi32>
    call @printMemref1dI32(%x0) : (memref<?xi32>) -> ()
    // Stable sort.
    // CHECK: [10,  2,  0,  5,  1]
    sparse_tensor.sort insertion_sort_stable %i0, %x0 : memref<?xi32>
    call @printMemref1dI32(%x0) : (memref<?xi32>) -> ()
    // Heap sort.
    // CHECK: [10,  2,  0,  5,  1]
    sparse_tensor.sort heap_sort %i0, %x0 : memref<?xi32>
    call @printMemref1dI32(%x0) : (memref<?xi32>) -> ()
    // Hybrid sort.
    // CHECK: [10,  2,  0,  5,  1]
    sparse_tensor.sort hybrid_quick_sort %i0, %x0 : memref<?xi32>
    call @printMemref1dI32(%x0) : (memref<?xi32>) -> ()

    // Sort the first 4 elements, with the last valid value untouched.
    // Quick sort.
    // CHECK: [0,  2,  5, 10,  1]
    sparse_tensor.sort quick_sort %i4, %x0 : memref<?xi32>
    call @printMemref1dI32(%x0) : (memref<?xi32>) -> ()
    // Stable sort.
    // CHECK: [0,  2,  5,  10,  1]
    call @storeValuesTo(%x0, %c10, %c2, %c0, %c5, %c1)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    sparse_tensor.sort insertion_sort_stable %i4, %x0 : memref<?xi32>
    call @printMemref1dI32(%x0) : (memref<?xi32>) -> ()
    // Heap sort.
    // CHECK: [0,  2,  5,  10,  1]
    call @storeValuesTo(%x0, %c10, %c2, %c0, %c5, %c1)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    sparse_tensor.sort heap_sort %i4, %x0 : memref<?xi32>
    call @printMemref1dI32(%x0) : (memref<?xi32>) -> ()
    // Hybrid sort.
    // CHECK: [0,  2,  5, 10,  1]
    sparse_tensor.sort hybrid_quick_sort %i4, %x0 : memref<?xi32>
    call @printMemref1dI32(%x0) : (memref<?xi32>) -> ()

    // Prepare more buffers of different dimensions.
    %x1s = memref.alloc() : memref<10xi32>
    %x1 = memref.cast %x1s : memref<10xi32> to memref<?xi32>
    %x2s = memref.alloc() : memref<6xi32>
    %x2 = memref.cast %x2s : memref<6xi32> to memref<?xi32>
    %y0s = memref.alloc() : memref<7xi32>
    %y0 = memref.cast %y0s : memref<7xi32> to memref<?xi32>

    // Sort "parallel arrays".
    // CHECK: [1,  1,  2,  5,  10]
    // CHECK: [3,  3,  1,  10,  1
    // CHECK: [9,  9,  4,  7,  2
    // CHECK: [7,  8,  10,  9,  6
    call @storeValuesTo(%x0, %c10, %c2, %c1, %c5, %c1)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%x1, %c1, %c1, %c3, %c10, %c3)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%x2, %c2, %c4, %c9, %c7, %c9)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%y0, %c6, %c10, %c8, %c9, %c7)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    sparse_tensor.sort quick_sort %i5, %x0, %x1, %x2 jointly %y0
      : memref<?xi32>, memref<?xi32>, memref<?xi32> jointly memref<?xi32>
    call @printMemref1dI32(%x0) : (memref<?xi32>) -> ()
    call @printMemref1dI32(%x1) : (memref<?xi32>) -> ()
    call @printMemref1dI32(%x2) : (memref<?xi32>) -> ()
    call @printMemref1dI32(%y0) : (memref<?xi32>) -> ()
    // Stable sort.
    // CHECK: [1,  1,  2,  5,  10]
    // CHECK: [3,  3,  1,  10,  1
    // CHECK: [9,  9,  4,  7,  2
    // CHECK: [8,  7,  10,  9,  6
    call @storeValuesTo(%x0, %c10, %c2, %c1, %c5, %c1)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%x1, %c1, %c1, %c3, %c10, %c3)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%x2, %c2, %c4, %c9, %c7, %c9)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%y0, %c6, %c10, %c8, %c9, %c7)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    sparse_tensor.sort insertion_sort_stable %i5, %x0, %x1, %x2 jointly %y0
      : memref<?xi32>, memref<?xi32>, memref<?xi32> jointly memref<?xi32>
    call @printMemref1dI32(%x0) : (memref<?xi32>) -> ()
    call @printMemref1dI32(%x1) : (memref<?xi32>) -> ()
    call @printMemref1dI32(%x2) : (memref<?xi32>) -> ()
    call @printMemref1dI32(%y0) : (memref<?xi32>) -> ()
    // Heap sort.
    // CHECK: [1,  1,  2,  5,  10]
    // CHECK: [3,  3,  1,  10,  1
    // CHECK: [9,  9,  4,  7,  2
    // CHECK: [7,  8,  10,  9,  6
    call @storeValuesTo(%x0, %c10, %c2, %c1, %c5, %c1)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%x1, %c1, %c1, %c3, %c10, %c3)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%x2, %c2, %c4, %c9, %c7, %c9)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    call @storeValuesTo(%y0, %c6, %c10, %c8, %c9, %c7)
      : (memref<?xi32>, i32, i32, i32, i32, i32) -> ()
    sparse_tensor.sort heap_sort %i5, %x0, %x1, %x2 jointly %y0
      : memref<?xi32>, memref<?xi32>, memref<?xi32> jointly memref<?xi32>
    call @printMemref1dI32(%x0) : (memref<?xi32>) -> ()
    call @printMemref1dI32(%x1) : (memref<?xi32>) -> ()
    call @printMemref1dI32(%x2) : (memref<?xi32>) -> ()
    call @printMemref1dI32(%y0) : (memref<?xi32>) -> ()

    // Release the buffers.
    memref.dealloc %x0 : memref<?xi32>
    memref.dealloc %x1 : memref<?xi32>
    memref.dealloc %x2 : memref<?xi32>
    memref.dealloc %y0 : memref<?xi32>
    return
  }
}
