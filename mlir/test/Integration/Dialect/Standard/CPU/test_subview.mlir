// RUN: mlir-opt %s -convert-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext | FileCheck %s

memref.global "private" constant @__constant_5x3xf32 : memref<5x3xf32> =
dense<[[0.0, 1.0, 2.0],
       [3.0, 4.0, 5.0],
       [6.0, 7.0, 8.0],
       [9.0, 10.0, 11.0],
       [12.0, 13.0, 14.0]]>

func.func @main() {
  %0 = memref.get_global @__constant_5x3xf32 : memref<5x3xf32>

  /// Subview with only leading operands.
  %1 = memref.subview %0[2, 0][3, 3][1, 1]: memref<5x3xf32> to memref<3x3xf32, strided<[3, 1], offset: 6>>
  %unranked = memref.cast %1 : memref<3x3xf32, strided<[3, 1], offset: 6>> to memref<*xf32>
  call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()

  //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
  // CHECK-SAME: rank = 2 offset = 6 sizes = [3, 3] strides = [3, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME:  [6,   7,   8],
  // CHECK-NEXT:  [9,   10,   11],
  // CHECK-NEXT:  [12,   13,   14]
  // CHECK-SAME: ]

  /// Regular subview.
  %2 = memref.subview %0[0, 2][5, 1][1, 1]: memref<5x3xf32> to memref<5x1xf32, strided<[3, 1], offset: 2>>
  %unranked2 = memref.cast %2 : memref<5x1xf32, strided<[3, 1], offset: 2>> to memref<*xf32>
  call @printMemrefF32(%unranked2) : (memref<*xf32>) -> ()

  //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
  // CHECK-SAME: rank = 2 offset = 2 sizes = [5, 1] strides = [3, 1] data =
  // CHECK-NEXT: [
  // CHECK-SAME:  [2],
  // CHECK-NEXT:  [5],
  // CHECK-NEXT:  [8],
  // CHECK-NEXT:  [11],
  // CHECK-NEXT:  [14]
  // CHECK-SAME: ]

  /// Rank-reducing subview.
  %3 = memref.subview %0[0, 2][5, 1][1, 1]: memref<5x3xf32> to memref<5xf32, strided<[3], offset: 2>>
  %unranked3 = memref.cast %3 : memref<5xf32, strided<[3], offset: 2>> to memref<*xf32>
  call @printMemrefF32(%unranked3) : (memref<*xf32>) -> ()

  //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
  // CHECK-SAME: rank = 1 offset = 2 sizes = [5] strides = [3] data =
  // CHECK-NEXT: [2,  5,  8,  11,  14]

  /// Rank-reducing subview with only leading operands.
  %4 = memref.subview %0[1, 0][1, 3][1, 1]: memref<5x3xf32> to memref<3xf32, strided<[1], offset: 3>>
  %unranked4 = memref.cast %4 : memref<3xf32, strided<[1], offset: 3>> to memref<*xf32>
  call @printMemrefF32(%unranked4) : (memref<*xf32>) -> ()
  //      CHECK: Unranked Memref base@ = {{0x[-9a-f]*}}
  // CHECK-SAME: rank = 1 offset = 3 sizes = [3] strides = [1] data =
  // CHECK-NEXT: [3,  4,  5]

  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
