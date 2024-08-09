// RUN: mlir-opt %s --optimize-allocation-liveness --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func private @optimize_alloc_location(
// CHECK-SAME:                             %[[VAL_0:.*]]: memref<45x24x256xf32, 1>,
// CHECK-SAME:                             %[[VAL_1:.*]]: memref<24x256xf32, 1>,
// CHECK-SAME:                             %[[VAL_2:.*]]: memref<256xf32, 1>) {
// CHECK:           %[[VAL_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<45x6144xf32, 1>
// CHECK:           %[[VAL_4:.*]] = memref.expand_shape %[[VAL_3]] {{\[\[}}0], [1, 2]] output_shape [45, 24, 256] : memref<45x6144xf32, 1> into memref<45x24x256xf32, 1>
// CHECK:           memref.dealloc %[[VAL_3]] : memref<45x6144xf32, 1>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() {alignment = 64 : i64} : memref<24x256xf32, 1>
// CHECK:           linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_1]], %[[VAL_2]] : memref<24x256xf32, 1>, memref<256xf32, 1>) outs(%[[VAL_5]] : memref<24x256xf32, 1>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f32, %[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : f32
// CHECK:             linalg.yield %[[VAL_9]] : f32
// CHECK:           }
// CHECK:           memref.dealloc %[[VAL_5]] : memref<24x256xf32, 1>
// CHECK:           return
// CHECK:         }

// this will optimize the location of the %alloc deallocation
func.func private @optimize_alloc_location(%arg0: memref<45x24x256xf32, 1> , %arg1: memref<24x256xf32, 1> , %arg2: memref<256xf32, 1>) -> () {

  %alloc = memref.alloc() {alignment = 64 : i64} : memref<45x6144xf32, 1>
  %expand_shape = memref.expand_shape %alloc [[0], [1, 2]] output_shape [45, 24, 256] : memref<45x6144xf32, 1> into memref<45x24x256xf32, 1>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<24x256xf32, 1>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg2 : memref<24x256xf32, 1>, memref<256xf32, 1>) outs(%alloc_1 : memref<24x256xf32, 1>) {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %0 = arith.addf %in, %in_3 : f32
    linalg.yield %0 : f32
  }
  memref.dealloc %alloc : memref<45x6144xf32, 1>
  memref.dealloc %alloc_1 : memref<24x256xf32, 1>
  return
}

// -----

// CHECK-LABEL:   func.func private @test_multiple_deallocation_moves(
// CHECK-SAME:                                                        %[[VAL_0:.*]]: memref<45x24x256xf32, 1>,
// CHECK-SAME:                                                        %[[VAL_1:.*]]: memref<24x256xf32, 1>,
// CHECK-SAME:                                                        %[[VAL_2:.*]]: memref<256xf32, 1>) {
// CHECK:           %[[VAL_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<45x6144xf32, 1>
// CHECK:           %[[VAL_4:.*]] = memref.expand_shape %[[VAL_3]] {{\[\[}}0], [1, 2]] output_shape [45, 24, 256] : memref<45x6144xf32, 1> into memref<45x24x256xf32, 1>
// CHECK:           memref.dealloc %[[VAL_3]] : memref<45x6144xf32, 1>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() {alignment = 64 : i64} : memref<24x256xf32, 1>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() {alignment = 64 : i64} : memref<45x6144xf32, 1>
// CHECK:           %[[VAL_7:.*]] = memref.expand_shape %[[VAL_6]] {{\[\[}}0], [1, 2]] output_shape [45, 24, 256] : memref<45x6144xf32, 1> into memref<45x24x256xf32, 1>
// CHECK:           memref.dealloc %[[VAL_6]] : memref<45x6144xf32, 1>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() {alignment = 64 : i64} : memref<45x6144xf32, 1>
// CHECK:           %[[VAL_9:.*]] = memref.expand_shape %[[VAL_8]] {{\[\[}}0], [1, 2]] output_shape [45, 24, 256] : memref<45x6144xf32, 1> into memref<45x24x256xf32, 1>
// CHECK:           memref.dealloc %[[VAL_8]] : memref<45x6144xf32, 1>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() {alignment = 64 : i64} : memref<45x6144xf32, 1>
// CHECK:           %[[VAL_11:.*]] = memref.expand_shape %[[VAL_10]] {{\[\[}}0], [1, 2]] output_shape [45, 24, 256] : memref<45x6144xf32, 1> into memref<45x24x256xf32, 1>
// CHECK:           memref.dealloc %[[VAL_10]] : memref<45x6144xf32, 1>
// CHECK:           linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_1]], %[[VAL_2]] : memref<24x256xf32, 1>, memref<256xf32, 1>) outs(%[[VAL_5]] : memref<24x256xf32, 1>) {
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: f32):
// CHECK:             %[[VAL_15:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:             linalg.yield %[[VAL_15]] : f32
// CHECK:           }
// CHECK:           memref.dealloc %[[VAL_5]] : memref<24x256xf32, 1>
// CHECK:           return
// CHECK:         }

// This tests creates multiple deallocation rearrangements. 
func.func private @test_multiple_deallocation_moves(%arg0: memref<45x24x256xf32, 1> , %arg1: memref<24x256xf32, 1> , %arg2: memref<256xf32, 1>) -> () {

  %alloc = memref.alloc() {alignment = 64 : i64} : memref<45x6144xf32, 1>
  %expand_shape = memref.expand_shape %alloc [[0], [1, 2]] output_shape [45, 24, 256] : memref<45x6144xf32, 1> into memref<45x24x256xf32, 1>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<24x256xf32, 1>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<45x6144xf32, 1>
  %expand_shape2 = memref.expand_shape %alloc_2 [[0], [1, 2]] output_shape [45, 24, 256] : memref<45x6144xf32, 1> into memref<45x24x256xf32, 1>
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<45x6144xf32, 1>
  %expand_shape3 = memref.expand_shape %alloc_3 [[0], [1, 2]] output_shape [45, 24, 256] : memref<45x6144xf32, 1> into memref<45x24x256xf32, 1>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<45x6144xf32, 1>
  %expand_shape4 = memref.expand_shape %alloc_4 [[0], [1, 2]] output_shape [45, 24, 256] : memref<45x6144xf32, 1> into memref<45x24x256xf32, 1>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg2 : memref<24x256xf32, 1>, memref<256xf32, 1>) outs(%alloc_1 : memref<24x256xf32, 1>) {
  ^bb0(%in: f32, %in_3: f32, %out: f32):
    %0 = arith.addf %in, %in_3 : f32
    linalg.yield %0 : f32
  }
  memref.dealloc %alloc : memref<45x6144xf32, 1>
  memref.dealloc %alloc_1 : memref<24x256xf32, 1>
  memref.dealloc %alloc_2 : memref<45x6144xf32, 1>
  memref.dealloc %alloc_3 : memref<45x6144xf32, 1>
  memref.dealloc %alloc_4 : memref<45x6144xf32, 1>
  return
}

// -----
// CHECK-LABEL:   func.func private @test_users_in_different_blocks_linalig_generic(
// CHECK-SAME:                                                                      %[[VAL_0:.*]]: memref<1x20x20xf32, 1>) -> (memref<8x32x1x4xf32, 1>, memref<1x32x32xf32, 1>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x32x32xf32, 1>
// CHECK:           %[[VAL_4:.*]] = memref.subview %[[VAL_3]][0, 0, 0] [1, 20, 20] [1, 1, 1] : memref<1x32x32xf32, 1> to memref<1x20x20xf32, strided<[1024, 32, 1]>, 1>
// CHECK:           memref.copy %[[VAL_0]], %[[VAL_4]] : memref<1x20x20xf32, 1> to memref<1x20x20xf32, strided<[1024, 32, 1]>, 1>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32, 1>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x8x32x1x4xf32, 1>
// CHECK:           linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL_6]] : memref<1x8x32x1x4xf32, 1>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32):
// CHECK:             %[[VAL_8:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_9:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_8]], %[[VAL_8]], %[[VAL_8]], %[[VAL_2]]] : memref<1x32x32x1xf32, 1>
// CHECK:             linalg.yield %[[VAL_9]] : f32
// CHECK:           }
// CHECK:           memref.dealloc %[[VAL_5]] : memref<1x32x32x1xf32, 1>
// CHECK:           %[[VAL_10:.*]] = memref.collapse_shape %[[VAL_6]] {{\[\[}}0, 1], [2], [3], [4]] : memref<1x8x32x1x4xf32, 1> into memref<8x32x1x4xf32, 1>
// CHECK:           memref.dealloc %[[VAL_6]] : memref<1x8x32x1x4xf32, 1>
// CHECK:           return %[[VAL_10]], %[[VAL_3]] : memref<8x32x1x4xf32, 1>, memref<1x32x32xf32, 1>
// CHECK:         }


// This test will optimize the location of the %alloc_0 deallocation, since the last user of this allocation is the last linalg.generic operation
// it will move the deallocation right after the last linalg.generic operation
// %alloc_1 will not be moved becuase of the collapse shape op.
func.func private @test_users_in_different_blocks_linalig_generic(%arg0: memref<1x20x20xf32, 1>) -> (memref<8x32x1x4xf32, 1> , memref<1x32x32xf32, 1> ) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x32x32xf32, 1>
  %subview = memref.subview %alloc[0, 0, 0] [1, 20, 20] [1, 1, 1] : memref<1x32x32xf32, 1> to memref<1x20x20xf32, strided<[1024, 32, 1]>, 1>
  memref.copy %arg0, %subview : memref<1x20x20xf32, 1> to memref<1x20x20xf32, strided<[1024, 32, 1]>, 1>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x32x32x1xf32, 1>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x8x32x1x4xf32, 1>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%alloc_1 : memref<1x8x32x1x4xf32, 1>) {
  ^bb0(%out: f32):
    %0 = linalg.index 0 : index
    %8 = memref.load %alloc_0[%0, %0, %0, %c0] : memref<1x32x32x1xf32, 1>
    linalg.yield %8 : f32
  }
  %collapse_shape = memref.collapse_shape %alloc_1 [[0, 1], [2], [3], [4]] : memref<1x8x32x1x4xf32, 1> into memref<8x32x1x4xf32, 1>
  memref.dealloc %alloc_0 : memref<1x32x32x1xf32, 1>
  memref.dealloc %alloc_1 : memref<1x8x32x1x4xf32, 1>
  return %collapse_shape, %alloc : memref<8x32x1x4xf32, 1>, memref<1x32x32xf32, 1>
}

// -----
// CHECK-LABEL:   func.func private @test_deallocs_in_different_block_forops(
// CHECK-SAME:                                                               %[[VAL_0:.*]]: memref<45x24x256xf32, 1>,
// CHECK-SAME:                                                               %[[VAL_1:.*]]: memref<24x256xf32, 1>,
// CHECK-SAME:                                                               %[[VAL_2:.*]]: memref<256xf32, 1>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 45 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 24 : index
// CHECK:           %[[VAL_8:.*]] = memref.alloc() {alignment = 64 : i64} : memref<45x6144xf32, 1>
// CHECK:           %[[VAL_9:.*]] = memref.expand_shape %[[VAL_8]] {{\[\[}}0], [1, 2]] output_shape [45, 24, 256] : memref<45x6144xf32, 1> into memref<45x24x256xf32, 1>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() {alignment = 64 : i64} : memref<24x256xf32, 1>
// CHECK:           %[[VAL_11:.*]] = memref.alloc() {alignment = 64 : i64} : memref<45x6144xf32, 1>
// CHECK:           %[[VAL_12:.*]] = memref.expand_shape %[[VAL_11]] {{\[\[}}0], [1, 2]] output_shape [45, 24, 256] : memref<45x6144xf32, 1> into memref<45x24x256xf32, 1>
// CHECK:           memref.dealloc %[[VAL_11]] : memref<45x6144xf32, 1>
// CHECK:           scf.for %[[VAL_13:.*]] = %[[VAL_3]] to %[[VAL_6]] step %[[VAL_4]] {
// CHECK:             scf.for %[[VAL_14:.*]] = %[[VAL_3]] to %[[VAL_7]] step %[[VAL_5]] {
// CHECK:               %[[VAL_15:.*]] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_13]], %[[VAL_14]], 0] [1, 8, 256] [1, 1, 1] : memref<45x24x256xf32, 1> to memref<1x8x256xf32, strided<[6144, 256, 1], offset: ?>, 1>
// CHECK:               %[[VAL_16:.*]] = memref.subview %[[VAL_10]]{{\[}}%[[VAL_14]], 0] [8, 256] [1, 1] : memref<24x256xf32, 1> to memref<8x256xf32, strided<[256, 1], offset: ?>, 1>
// CHECK:             }
// CHECK:           }
// CHECK:           memref.dealloc %[[VAL_10]] : memref<24x256xf32, 1>
// CHECK:           memref.dealloc %[[VAL_8]] : memref<45x6144xf32, 1>
// CHECK:           return
// CHECK:         }

// This test will not move the deallocations %alloc and %alloc1 since they are used in the last scf.for operation
// %alloc_2 will move right after its last user the expand_shape operation
func.func private @test_deallocs_in_different_block_forops(%arg0: memref<45x24x256xf32, 1>, %arg1: memref<24x256xf32, 1> , %arg2: memref<256xf32, 1> ) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c45 = arith.constant 45 : index
  %c24 = arith.constant 24 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<45x6144xf32, 1>
  %expand_shape = memref.expand_shape %alloc [[0], [1, 2]] output_shape [45, 24, 256] : memref<45x6144xf32, 1> into memref<45x24x256xf32, 1>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<24x256xf32, 1>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<45x6144xf32, 1>
  %expand_shape2 = memref.expand_shape %alloc_2 [[0], [1, 2]] output_shape [45, 24, 256] : memref<45x6144xf32, 1> into memref<45x24x256xf32, 1>
  scf.for %arg3 = %c0 to %c45 step %c1 {
    scf.for %arg4 = %c0 to %c24 step %c8 {
      %subview = memref.subview %expand_shape[%arg3, %arg4, 0] [1, 8, 256] [1, 1, 1] : memref<45x24x256xf32, 1> to memref<1x8x256xf32, strided<[6144, 256, 1], offset: ?>, 1>
      %subview_3 = memref.subview %alloc_1[%arg4, 0] [8, 256] [1, 1] : memref<24x256xf32, 1> to memref<8x256xf32, strided<[256, 1], offset: ?>, 1>    
    }
  }
  memref.dealloc %alloc : memref<45x6144xf32, 1>
  memref.dealloc %alloc_1 : memref<24x256xf32, 1>
  memref.dealloc %alloc_2 : memref<45x6144xf32, 1>
  return
}