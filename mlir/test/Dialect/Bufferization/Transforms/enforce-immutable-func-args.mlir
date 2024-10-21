// RUN: mlir-opt --split-input-file --enforce-immutable-func-args %s -o - | FileCheck %s


// CHECK-LABEL:   func.func @func_no_input() {
// CHECK:           return
// CHECK:         }

func.func @func_no_input() {
  return
}

// -----

// CHECK-LABEL:   func.func private @func_with_returned_argument(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: memref<1x13x21x3xf32>) -> memref<1x13x21x3xf32> {
// CHECK:           return %[[VAL_0]] : memref<1x13x21x3xf32>
// CHECK:         }

func.func private @func_with_returned_argument(%arg0: memref<1x13x21x3xf32>) -> (memref<1x13x21x3xf32>) {
  return %arg0 : memref<1x13x21x3xf32>
}

// -----

// CHECK-LABEL:   func.func private @func_with_modified_argument_directly(
// CHECK-SAME:                                                            %[[VAL_0:.*]]: memref<1x13x21x3xf32>, %[[VAL_1:.*]]: memref<1x13x21x3xf32>) -> memref<1x13x21x3xf32> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() : memref<1x13x21x3xf32>
// CHECK:           memref.copy %[[VAL_0]], %[[VAL_2]] : memref<1x13x21x3xf32> to memref<1x13x21x3xf32>
// CHECK:           %[[VAL_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x13x21x3xf32>
// CHECK:           linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_2]], %[[VAL_1]] : memref<1x13x21x3xf32>, memref<1x13x21x3xf32>) outs(%[[VAL_2]] : memref<1x13x21x3xf32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:             %[[VAL_7:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f32
// CHECK:             linalg.yield %[[VAL_7]] : f32
// CHECK:           }
// CHECK:           return %[[VAL_3]] : memref<1x13x21x3xf32>
// CHECK:         }

func.func private @func_with_modified_argument_directly(%arg0: memref<1x13x21x3xf32>, %arg1: memref<1x13x21x3xf32>) -> (memref<1x13x21x3xf32>){
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x13x21x3xf32>
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  }
  ins(%arg0, %arg1 : memref<1x13x21x3xf32>, memref<1x13x21x3xf32>)
  outs(%arg0 : memref<1x13x21x3xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %0 = arith.addf %in, %in_0 : f32
    linalg.yield %0 : f32
  }
  return %alloc : memref<1x13x21x3xf32>
}

// -----

// CHECK-LABEL:   func.func private @func_with_modified_argument_directly_and_returned(
// CHECK-SAME:                                                                         %[[VAL_0:.*]]: memref<1x13x21x3xf32>, %[[VAL_1:.*]]: memref<1x13x21x3xf32>) -> memref<1x13x21x3xf32> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() : memref<1x13x21x3xf32>
// CHECK:           memref.copy %[[VAL_0]], %[[VAL_2]] : memref<1x13x21x3xf32> to memref<1x13x21x3xf32>
// CHECK:           linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_2]], %[[VAL_1]] : memref<1x13x21x3xf32>, memref<1x13x21x3xf32>) outs(%[[VAL_2]] : memref<1x13x21x3xf32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32):
// CHECK:             %[[VAL_6:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             linalg.yield %[[VAL_6]] : f32
// CHECK:           }
// CHECK:           return %[[VAL_2]] : memref<1x13x21x3xf32>
// CHECK:         }

func.func private @func_with_modified_argument_directly_and_returned(%arg0: memref<1x13x21x3xf32>, %arg1: memref<1x13x21x3xf32>) -> (memref<1x13x21x3xf32>){
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  }
  ins(%arg0, %arg1 : memref<1x13x21x3xf32>, memref<1x13x21x3xf32>)
  outs(%arg0 : memref<1x13x21x3xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %0 = arith.addf %in, %in_0 : f32
    linalg.yield %0 : f32
  }
  return %arg0 : memref<1x13x21x3xf32>
}

// -----

// CHECK-LABEL:   func.func private @func_with_modified_argument_directly_twice(
// CHECK-SAME:                                                                  %[[VAL_0:.*]]: memref<1x13x21x3xf32>, %[[VAL_1:.*]]: memref<1x13x21x3xf32>) -> memref<1x13x21x3xf32> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() : memref<1x13x21x3xf32>
// CHECK:           memref.copy %[[VAL_0]], %[[VAL_2]] : memref<1x13x21x3xf32> to memref<1x13x21x3xf32>
// CHECK:           %[[VAL_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x13x21x3xf32>
// CHECK:           linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_2]], %[[VAL_1]] : memref<1x13x21x3xf32>, memref<1x13x21x3xf32>) outs(%[[VAL_2]] : memref<1x13x21x3xf32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:             %[[VAL_7:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f32
// CHECK:             linalg.yield %[[VAL_7]] : f32
// CHECK:           }
// CHECK:           linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL_2]], %[[VAL_1]] : memref<1x13x21x3xf32>, memref<1x13x21x3xf32>) outs(%[[VAL_2]] : memref<1x13x21x3xf32>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: f32, %[[VAL_9:.*]]: f32, %[[VAL_10:.*]]: f32):
// CHECK:             %[[VAL_11:.*]] = arith.addf %[[VAL_8]], %[[VAL_9]] : f32
// CHECK:             linalg.yield %[[VAL_11]] : f32
// CHECK:           }
// CHECK:           return %[[VAL_3]] : memref<1x13x21x3xf32>
// CHECK:         }

func.func private @func_with_modified_argument_directly_twice(%arg0: memref<1x13x21x3xf32>, %arg1: memref<1x13x21x3xf32>) -> (memref<1x13x21x3xf32>){
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x13x21x3xf32>
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  }
  ins(%arg0, %arg1 : memref<1x13x21x3xf32>, memref<1x13x21x3xf32>)
  outs(%arg0 : memref<1x13x21x3xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %0 = arith.addf %in, %in_0 : f32
    linalg.yield %0 : f32
  }
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  }
  ins(%arg0, %arg1 : memref<1x13x21x3xf32>, memref<1x13x21x3xf32>)
  outs(%arg0 : memref<1x13x21x3xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %0 = arith.addf %in, %in_0 : f32
    linalg.yield %0 : f32
  }
  return %alloc : memref<1x13x21x3xf32>
}

// -----

// CHECK-LABEL:   func.func private @func_with_modified_argument_directly(
// CHECK-SAME:                                                            %[[VAL_0:.*]]: memref<5xi32, 1>, %[[VAL_1:.*]]: memref<5xi32, 1>, %[[VAL_2:.*]]: memref<5xi32, 1>) -> memref<5xi32, 1> {
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<5xi32, 1>
// CHECK:           memref.copy %[[VAL_2]], %[[VAL_3]] : memref<5xi32, 1> to memref<5xi32, 1>
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           scf.for %[[VAL_7:.*]] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_4]] {
// CHECK:             %[[VAL_8:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_7]]] : memref<5xi32, 1>
// CHECK:             %[[VAL_9:.*]] = arith.index_cast %[[VAL_8]] : i32 to index
// CHECK:             %[[VAL_10:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_7]]] : memref<5xi32, 1>
// CHECK:             %[[VAL_11:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_9]]] : memref<5xi32, 1>
// CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_10]], %[[VAL_11]] : i32
// CHECK:             memref.store %[[VAL_12]], %[[VAL_3]]{{\[}}%[[VAL_9]]] : memref<5xi32, 1>
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = memref.alloc() : memref<5xi32, 1>
// CHECK:           memref.copy %[[VAL_3]], %[[VAL_13]] : memref<5xi32, 1> to memref<5xi32, 1>
// CHECK:           return %[[VAL_13]] : memref<5xi32, 1>
// CHECK:         }

func.func private @func_with_modified_argument_directly(%arg0: memref<5xi32, 1>, %arg1: memref<5xi32, 1>, %arg2: memref<5xi32, 1>) -> (memref<5xi32, 1>){
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c0 = arith.constant 0 : index
  scf.for %arg3 = %c0 to %c5 step %c1 {
    %0 = memref.load %arg0[%arg3] : memref<5xi32, 1>
    %1 = arith.index_cast %0 : i32 to index
    %2 = memref.load %arg1[%arg3] : memref<5xi32, 1>
    %3 = memref.load %arg2[%1] : memref<5xi32, 1>
    %4 = arith.addi %2, %3 : i32
    memref.store %4, %arg2[%1] : memref<5xi32, 1>
  }
  %alloc = memref.alloc() : memref<5xi32, 1>
  memref.copy %arg2, %alloc : memref<5xi32, 1> to memref<5xi32, 1>
  return %alloc : memref<5xi32, 1>
}

// -----

// CHECK-LABEL:   func.func private @func_with_modified_argument_indirectly(
// CHECK-SAME:                                                              %[[VAL_0:.*]]: memref<3x3x4xf32, 1>) -> memref<3x3x4xf32, 1> {
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<3x3x4xf32, 1>
// CHECK:           memref.copy %[[VAL_0]], %[[VAL_1]] : memref<3x3x4xf32, 1> to memref<3x3x4xf32, 1>
// CHECK:           %[[VAL_2:.*]] = memref.collapse_shape %[[VAL_1]] {{\[\[}}0, 1], [2]] : memref<3x3x4xf32, 1> into memref<9x4xf32, 1>
// CHECK:           %[[VAL_3:.*]] = memref.expand_shape %[[VAL_2]] {{\[\[}}0, 1], [2]] output_shape [3, 3, 4] : memref<9x4xf32, 1> into memref<3x3x4xf32, 1>
// CHECK:           linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%[[VAL_3]] : memref<3x3x4xf32, 1>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: f32):
// CHECK:             %[[VAL_5:.*]] = arith.addf %[[VAL_4]], %[[VAL_4]] : f32
// CHECK:             linalg.yield %[[VAL_5]] : f32
// CHECK:           }
// CHECK:           return %[[VAL_3]] : memref<3x3x4xf32, 1>
// CHECK:         }

func.func private @func_with_modified_argument_indirectly(%arg0: memref<3x3x4xf32, 1>) -> (memref<3x3x4xf32, 1>) {
  %collapse_arg = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<3x3x4xf32, 1> into memref<9x4xf32, 1>
  %expand_arg = memref.expand_shape %collapse_arg [[0, 1], [2]] output_shape [3, 3, 4] : memref<9x4xf32, 1> into memref<3x3x4xf32, 1>
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  }
  outs(%expand_arg : memref<3x3x4xf32, 1>) {
  ^bb0(%out: f32):
    %0 = arith.addf %out, %out : f32
    linalg.yield %0 : f32
  }
  return %expand_arg: memref<3x3x4xf32, 1>
}

// -----

// CHECK-LABEL:   func.func private @func_with_modified_argument_subview(
// CHECK-SAME:                                                           %[[VAL_0:.*]]: memref<2x4x4xi32, 1>) -> memref<4x4xi32, 1> {
// CHECK:           %[[VAL_1:.*]] = memref.alloc() : memref<2x4x4xi32, 1>
// CHECK:           memref.copy %[[VAL_0]], %[[VAL_1]] : memref<2x4x4xi32, 1> to memref<2x4x4xi32, 1>
// CHECK:           %[[VAL_2:.*]] = memref.subview %[[VAL_1]][0, 0, 0] [1, 4, 4] [1, 1, 1] : memref<2x4x4xi32, 1> to memref<1x4x4xi32, strided<[16, 4, 1]>, 1>
// CHECK:           %[[VAL_3:.*]] = memref.collapse_shape %[[VAL_2]] {{\[\[}}0, 1], [2]] : memref<1x4x4xi32, strided<[16, 4, 1]>, 1> into memref<4x4xi32, strided<[4, 1]>, 1>
// CHECK:           %[[VAL_4:.*]] = memref.cast %[[VAL_3]] : memref<4x4xi32, strided<[4, 1]>, 1> to memref<4x4xi32, 1>
// CHECK:           linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%[[VAL_4]] : memref<4x4xi32, 1>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: i32):
// CHECK:             %[[VAL_6:.*]] = arith.addi %[[VAL_5]], %[[VAL_5]] : i32
// CHECK:             linalg.yield %[[VAL_6]] : i32
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<4x4xi32, 1>
// CHECK:           memref.copy %[[VAL_4]], %[[VAL_7]] : memref<4x4xi32, 1> to memref<4x4xi32, 1>
// CHECK:           return %[[VAL_7]] : memref<4x4xi32, 1>
// CHECK:         }

func.func private @func_with_modified_argument_subview(%arg0: memref<2x4x4xi32, 1>) -> ( memref<4x4xi32, 1>){
  %subview = memref.subview %arg0[0, 0, 0] [1, 4, 4] [1, 1, 1] : memref<2x4x4xi32, 1> to memref<1x4x4xi32, strided<[16, 4, 1]>, 1>
  %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x4x4xi32, strided<[16, 4, 1]>, 1> into memref<4x4xi32, strided<[4, 1]>, 1>
  %cast = memref.cast %collapse_shape : memref<4x4xi32, strided<[4, 1]>, 1> to memref<4x4xi32, 1>
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  }
  outs(%cast : memref<4x4xi32, 1>) {
  ^bb0(%out: i32):
    %0 = arith.addi %out, %out : i32
    linalg.yield %0 : i32
  }
  %alloc = memref.alloc() : memref<4x4xi32, 1>
  memref.copy %cast, %alloc : memref<4x4xi32, 1> to memref<4x4xi32, 1>
  return %alloc : memref<4x4xi32, 1>
}

