// RUN: mlir-opt %s -test-linalg-elementwise-fusion-patterns=collapse-dimensions-control=2,3 -split-input-file | FileCheck %s

func.func @collapse_reduction(
    %arg0: tensor<2x32x10x4096xf32>, %arg1: tensor<2x32xf32>) -> tensor<2x32xf32> {
  %0 = linalg.generic {
    indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
  ins(%arg0 : tensor<2x32x10x4096xf32>) outs(%arg1 : tensor<2x32xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    %1 = arith.addf %arg3, %arg4 : f32
    linalg.yield %1 : f32
  } -> tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @collapse_reduction
//       CHECK:   %[[T:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0], [1], [2, 3]] : tensor<2x32x10x4096xf32> into tensor<2x32x40960xf32>
//       CHECK:   linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]],
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]}
//  CHECK-SAME:     ins(%[[T]] : tensor<2x32x40960xf32>) outs(%{{.*}} : tensor<2x32xf32>) {
//       CHECK:   } -> tensor<2x32xf32>

// -----

func.func @collapse_parallel(
    %arg0: tensor<32x2x10x4096xf32>, %arg1: tensor<2x32x10x4096xf32>) -> tensor<2x32x10x4096xf32> {
  %0 = linalg.generic {
    indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
  iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  ins(%arg0 : tensor<32x2x10x4096xf32>) outs(%arg1 : tensor<2x32x10x4096xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    %1 = arith.addf %arg3, %arg4 : f32
    linalg.yield %1 : f32
  } -> tensor<2x32x10x4096xf32>
  return %0 : tensor<2x32x10x4096xf32>
}

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func @collapse_parallel
//   CHECK-DAG:  %[[S:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0], [1], [2, 3]] : tensor<32x2x10x4096xf32> into tensor<32x2x40960xf32>
//   CHECK-DAG:  %[[D:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0], [1], [2, 3]] : tensor<2x32x10x4096xf32> into tensor<2x32x40960xf32>
//       CHECK:  %[[R:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]],
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel"]}
//  CHECK-SAME:     ins(%[[S]] : tensor<32x2x40960xf32>) outs(%[[D]] : tensor<2x32x40960xf32>) {
//       CHECK:   } -> tensor<2x32x40960xf32>
//       CHECK:  tensor.expand_shape %[[R]] {{\[}}[0], [1], [2, 3]] : tensor<2x32x40960xf32> into tensor<2x32x10x4096xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @uncollapsable(%arg0 : tensor<41x3x1x57xf32>, %arg1 : tensor<3x1x57x41xf32>) -> tensor<3x1x57x41xf32> {
  %0 = linalg.generic {
      indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<41x3x1x57xf32>) outs(%arg1 : tensor<3x1x57x41xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x1x57x41xf32>
  return %0 : tensor<3x1x57x41xf32>
}
// CHECK-LABEL: func @uncollapsable(
//       CHECK:   linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel"]

// -----

// CHECK-LABEL:   func.func private @collapsable_memref(
// CHECK-SAME:                                          %[[VAL_0:.*]]: memref<1x24x32x8xf32>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: memref<1x24x32x8xf32>) -> memref<1x24x32x8xf32> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x24x32x8xf32>
// CHECK:           %[[VAL_3:.*]] = memref.collapse_shape %[[VAL_0]] {{\[\[}}0], [1], [2, 3]] : memref<1x24x32x8xf32> into memref<1x24x256xf32>
// CHECK:           %[[VAL_4:.*]] = memref.collapse_shape %[[VAL_1]] {{\[\[}}0], [1], [2, 3]] : memref<1x24x32x8xf32> into memref<1x24x256xf32>
// CHECK:           %[[VAL_5:.*]] = memref.collapse_shape %[[VAL_2]] {{\[\[}}0], [1], [2, 3]] : memref<1x24x32x8xf32> into memref<1x24x256xf32>
// CHECK:           linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_3]], %[[VAL_4]] : memref<1x24x256xf32>, memref<1x24x256xf32>) outs(%[[VAL_5]] : memref<1x24x256xf32>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f32, %[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : f32
// CHECK:             linalg.yield %[[VAL_9]] : f32
// CHECK:           }
// CHECK:           return %[[VAL_2]] : memref<1x24x32x8xf32>
// CHECK:         }

func.func private @collapsable_memref(%arg0: memref<1x24x32x8xf32>, %arg1: memref<1x24x32x8xf32>) -> (memref<1x24x32x8xf32>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x24x32x8xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<1x24x32x8xf32>, memref<1x24x32x8xf32>) outs(%alloc : memref<1x24x32x8xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %0 = arith.addf %in, %in_0 : f32
    linalg.yield %0 : f32
  }
  return %alloc : memref<1x24x32x8xf32>
}

// -----

// CHECK-LABEL: func @uncollapsable_strided_memref(
//       CHECK:   linalg.generic
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel"]

func.func @uncollapsable_strided_memref(%arg0: memref<2x6x24x48xi32>, %arg1: memref<2x6x24x48xi32>) -> (memref<2x6x24x48xi32>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x6x24x48xi32>
  %subview = memref.subview %arg0[0, 0, 0, 0] [1, 3, 12, 24] [1, 1, 1, 1] : memref<2x6x24x48xi32> to memref<1x3x12x24xi32, strided<[6912, 1152, 48, 1], offset: 0>>
  %subview0 = memref.subview %arg1[0, 0, 0, 0] [1, 3, 12, 24] [1, 1, 1, 1] : memref<2x6x24x48xi32> to memref<1x3x12x24xi32, strided<[6912, 1152, 48, 1], offset: 0>>
  %subview1 = memref.subview %alloc[0, 0, 0, 0] [1, 3, 12, 24] [1, 1, 1, 1] : memref<2x6x24x48xi32> to memref<1x3x12x24xi32, strided<[6912, 1152, 48, 1], offset: 0>>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%subview, %subview0 : memref<1x3x12x24xi32, strided<[6912, 1152, 48, 1], offset: 0>>, memref<1x3x12x24xi32, strided<[6912, 1152, 48, 1], offset: 0>>) outs(%subview1 : memref<1x3x12x24xi32, strided<[6912, 1152, 48, 1], offset: 0>>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %0 = arith.addi %in, %in_0 : i32
    linalg.yield %0 : i32
  }
  return %alloc : memref<2x6x24x48xi32>
}

// -----

// CHECK-LABEL:   func.func @linalg_copy(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x2x3x4x5xf32, 1 : i64>,
// CHECK-SAME:                           %[[VAL_1:.*]]: tensor<1x2x3x4x5xf32, 3 : i64>) -> tensor<1x2x3x4x5xf32, 3 : i64> {
// CHECK:           %[[VAL_2:.*]] = tensor.collapse_shape %[[VAL_0]] {{\[\[}}0], [1], [2, 3], [4]] : tensor<1x2x3x4x5xf32, 1 : i64> into tensor<1x2x12x5xf32>
// CHECK:           %[[VAL_3:.*]] = tensor.collapse_shape %[[VAL_1]] {{\[\[}}0], [1], [2, 3], [4]] : tensor<1x2x3x4x5xf32, 3 : i64> into tensor<1x2x12x5xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.collapse_shape %[[VAL_2]] {{\[\[}}0], [1], [2, 3]] : tensor<1x2x12x5xf32> into tensor<1x2x60xf32>
// CHECK:           %[[VAL_5:.*]] = tensor.collapse_shape %[[VAL_3]] {{\[\[}}0], [1], [2, 3]] : tensor<1x2x12x5xf32> into tensor<1x2x60xf32>
// CHECK:           %[[VAL_6:.*]] = linalg.copy ins(%[[VAL_4]] : tensor<1x2x60xf32>) outs(%[[VAL_5]] : tensor<1x2x60xf32>) -> tensor<1x2x60xf32>
// CHECK:           %[[VAL_7:.*]] = tensor.expand_shape %[[VAL_6]] {{\[\[}}0], [1], [2, 3]] : tensor<1x2x60xf32> into tensor<1x2x12x5xf32>
// CHECK:           %[[VAL_8:.*]] = tensor.expand_shape %[[VAL_7]] {{\[\[}}0], [1], [2, 3], [4]] : tensor<1x2x12x5xf32> into tensor<1x2x3x4x5xf32, 3 : i64>
// CHECK:           return %[[VAL_8]] : tensor<1x2x3x4x5xf32, 3 : i64>
// CHECK:         }

func.func @linalg_copy(
    %arg0: tensor<1x2x3x4x5xf32, 1>, %arg1: tensor<1x2x3x4x5xf32, 3>) -> tensor<1x2x3x4x5xf32, 3> {
  %0 = linalg.copy ins(%arg0: tensor<1x2x3x4x5xf32, 1>) outs(%arg1: tensor<1x2x3x4x5xf32, 3>) -> tensor<1x2x3x4x5xf32, 3>
  return %0 : tensor<1x2x3x4x5xf32, 3>
}

// -----

// CHECK-LABEL:   func.func private @memref_linalg_copy(
// CHECK-SAME:                                          %[[VAL_0:.*]]: memref<1x24x32x8xf32, 1>,
// CHECK-SAME:                                          %[[VAL_1:.*]]: memref<1x24x32x8xf32, 1>) {
// CHECK:           %[[VAL_2:.*]] = memref.collapse_shape %[[VAL_0]] {{\[\[}}0], [1], [2, 3]] : memref<1x24x32x8xf32, 1> into memref<1x24x256xf32, 1>
// CHECK:           %[[VAL_3:.*]] = memref.collapse_shape %[[VAL_1]] {{\[\[}}0], [1], [2, 3]] : memref<1x24x32x8xf32, 1> into memref<1x24x256xf32, 1>
// CHECK:           linalg.copy ins(%[[VAL_2]] : memref<1x24x256xf32, 1>) outs(%[[VAL_3]] : memref<1x24x256xf32, 1>)
// CHECK:           return
// CHECK:         }

func.func private @memref_linalg_copy(%arg0: memref<1x24x32x8xf32, 1>, %arg1: memref<1x24x32x8xf32, 1>) {
  linalg.copy ins(%arg0: memref<1x24x32x8xf32, 1>) outs(%arg1: memref<1x24x32x8xf32, 1>)
  return
}
