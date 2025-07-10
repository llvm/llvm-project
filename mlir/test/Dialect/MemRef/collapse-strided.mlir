// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: test_collapse(
func.func @test_collapse(%arg0: memref<1x?xf32, strided<[5, 1]>>) {
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x?xf32, strided<[5, 1]>> into memref<?xf32, strided<[1]>>
  return
}

// CHECK-LABEL: test_collapse_5d_middle_dynamic(
func.func @test_collapse_5d_middle_dynamic(%arg0: memref<1x5x1x?x1xf32, strided<[540, 108, 18, 2, 1]>>) {
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2, 3, 4]]
    : memref<1x5x1x?x1xf32, strided<[540, 108, 18, 2, 1]>> into memref<?xf32, strided<[?]>>
  return
}

// CHECK-LABEL: test_collapse_5d_mostly_units(
func.func @test_collapse_5d_mostly_units(%arg0: memref<1x1x1x?x1xf32, strided<[320, 80, 16, 2, 1]>>) {
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2, 3, 4]]
    : memref<1x1x1x?x1xf32, strided<[320, 80, 16, 2, 1]>> into memref<?xf32, strided<[2]>>
  return
}

// CHECK-LABEL: test_partial_collapse_6d(
func.func @test_partial_collapse_6d(%arg0: memref<1x?x1x1x5x1xf32, strided<[3360, 420, 140, 35, 7, 1]>>) {
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2, 3], [4, 5]]
    : memref<1x?x1x1x5x1xf32, strided<[3360, 420, 140, 35, 7, 1]>> into memref<?x5xf32, strided<[420, 7]>>
  return
}

// CHECK-LABEL: test_collapse_5d_grouped(
func.func @test_collapse_5d_grouped(%arg0: memref<1x5x1x?x1xf32, strided<[540, 108, 18, 2, 1]>>) {
  %collapse_shape = memref.collapse_shape %arg0 [[0], [1, 2, 3, 4]]
    : memref<1x5x1x?x1xf32, strided<[540, 108, 18, 2, 1]>> into memref<1x?xf32, strided<[540, ?]>>
  return
}

// CHECK-LABEL: test_collapse_all_units(
func.func @test_collapse_all_units(%arg0: memref<1x1x1x1x1xf32, strided<[100, 50, 25, 10, 1]>>) {
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2, 3, 4]]
    : memref<1x1x1x1x1xf32, strided<[100, 50, 25, 10, 1]>> into memref<1xf32, strided<[100]>>
  return
}