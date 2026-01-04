// RUN: mlir-opt -test-strided-metadata-range-analysis %s 2>&1 | FileCheck %s

func.func @memref_subview(%arg0: memref<8x16x4xf32, strided<[64, 4, 1]>>, %arg1: memref<1x128x1x32x1xf32, strided<[4096, 32, 32, 1, 1]>>, %arg2: memref<8x16x4xf32, strided<[1, 64, 8], offset: 16>>, %arg3: index, %arg4: index, %arg5: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = test.with_bounds {smax = 13 : index, smin = 11 : index, umax = 13 : index, umin = 11 : index} : index
  %1 = test.with_bounds {smax = 7 : index, smin = 5 : index, umax = 7 : index, umin = 5 : index} : index

  // Test subview with unknown sizes, and constant offsets and strides.
  // CHECK: Op:  %[[SV0:.*]] = memref.subview
  // CHECK-NEXT: result[0]: strided_metadata<
  // CHECK-SAME: offset = [{unsigned : [1, 1] signed : [1, 1]}]
  // CHECK-SAME: sizes = [{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}, {unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}, {unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}]
  // CHECK-SAME: strides = [{unsigned : [64, 64] signed : [64, 64]}, {unsigned : [4, 4] signed : [4, 4]}, {unsigned : [1, 1] signed : [1, 1]}]
  %subview = memref.subview %arg0[%c0, %c0, %c1] [%arg3, %arg4, %arg5] [%c1, %c1, %c1] : memref<8x16x4xf32, strided<[64, 4, 1]>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // Test a subview of a subview, with bounded dynamic offsets.
  // CHECK: Op:  %[[SV1:.*]] = memref.subview
  // CHECK-NEXT: result[0]: strided_metadata<
  // CHECK-SAME: offset = [{unsigned : [346, 484] signed : [346, 484]}]
  // CHECK-SAME: sizes = [{unsigned : [2, 2] signed : [2, 2]}, {unsigned : [2, 2] signed : [2, 2]}, {unsigned : [2, 2] signed : [2, 2]}]
  // CHECK-SAME: strides = [{unsigned : [704, 832] signed : [704, 832]}, {unsigned : [44, 52] signed : [44, 52]}, {unsigned : [11, 13] signed : [11, 13]}]
  %subview_0 = memref.subview %subview[%1, %1, %1] [%c2, %c2, %c2] [%0, %0, %0] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // Test a subview of a subview, with constant operands.
  // CHECK: Op:  %[[SV2:.*]] = memref.subview
  // CHECK-NEXT: result[0]: strided_metadata<
  // CHECK-SAME: offset = [{unsigned : [368, 510] signed : [368, 510]}]
  // CHECK-SAME: sizes = [{unsigned : [2, 2] signed : [2, 2]}, {unsigned : [2, 2] signed : [2, 2]}, {unsigned : [2, 2] signed : [2, 2]}]
  // CHECK-SAME: strides = [{unsigned : [704, 832] signed : [704, 832]}, {unsigned : [44, 52] signed : [44, 52]}, {unsigned : [11, 13] signed : [11, 13]}]
  %subview_1 = memref.subview %subview_0[%c0, %c0, %c2] [%c2, %c2, %c2] [%c1, %c1, %c1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>

  // Test a rank-reducing subview.
  // CHECK: Op:  %[[SV3:.*]] = memref.subview
  // CHECK-NEXT: result[0]: strided_metadata<
  // CHECK-SAME: offset = [{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}]
  // CHECK-SAME: sizes = [{unsigned : [64, 64] signed : [64, 64]}, {unsigned : [16, 16] signed : [16, 16]}]
  // CHECK-SAME: strides = [{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}, {unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}]
  %subview_2 = memref.subview %arg1[%arg4, %arg4, %arg4, %arg4, %arg4] [1, 64, 1, 16, 1] [%arg5, %arg5, %arg5, %arg5, %arg5] : memref<1x128x1x32x1xf32, strided<[4096, 32, 32, 1, 1]>> to memref<64x16xf32, strided<[?, ?], offset: ?>>

  // Test a subview of a rank-reducing subview
  // CHECK: Op:  %[[SV4:.*]] = memref.subview
  // CHECK-NEXT: result[0]: strided_metadata<
  // CHECK-SAME: offset = [{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}]
  // CHECK-SAME: sizes = [{unsigned : [5, 7] signed : [5, 7]}]
  // CHECK-SAME: strides = [{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}]
  %subview_3 = memref.subview %subview_2[%c0, %0] [1, %1] [%c1, %c2] : memref<64x16xf32, strided<[?, ?], offset: ?>> to memref<?xf32, strided<[?], offset: ?>>

  // Test a subview with mixed bounded and unbound dynamic sizes.
  // CHECK: Op:  %[[SV5:.*]] = memref.subview
  // CHECK-NEXT: result[0]: strided_metadata<
  // CHECK-SAME: offset = [{unsigned : [32, 32] signed : [32, 32]}]
  // CHECK-SAME: sizes = [{unsigned : [11, 13] signed : [11, 13]}, {unsigned : [5, 7] signed : [5, 7]}, {unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}]
  // CHECK-SAME: strides = [{unsigned : [1, 1] signed : [1, 1]}, {unsigned : [64, 64] signed : [64, 64]}, {unsigned : [8, 8] signed : [8, 8]}]
  %subview_4 = memref.subview %arg2[%c0, %c0, %c2] [%0, %1, %arg5] [%c1, %c1, %c1] : memref<8x16x4xf32, strided<[1, 64, 8], offset: 16>> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  return
}

// CHECK:       func.func @memref_subview
// CHECK:       %[[A0:.*]]: memref<8x16x4xf32, strided<[64, 4, 1]>>
// CHECK:       %[[SV0]] = memref.subview %[[A0]]
// CHECK-NEXT:  %[[SV1]] = memref.subview
// CHECK-NEXT:  %[[SV2]] = memref.subview
// CHECK-NEXT:  %[[SV3]] = memref.subview
// CHECK-NEXT:  %[[SV4]] = memref.subview
// CHECK-NEXT:  %[[SV5]] = memref.subview
