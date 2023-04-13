// RUN: mlir-opt %s -test-linalg-data-layout-propagation -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @dynamic_elem_pack(%arg0: tensor<?x?xf32>, %dest: tensor<?x?x8x2xf32>) -> tensor<?x?x8x2xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = arith.addf %arg3, %arg3 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  %4 = tensor.pack %3
    inner_dims_pos = [0, 1]
    inner_tiles = [8, 2]
    into %dest : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
  return %4 : tensor<?x?x8x2xf32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK:      func.func @dynamic_elem_pack
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[DEST:[a-zA-Z0-9]+]]
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-DAG:    %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK-DAG:    %[[OUTER_D0:.+]] = affine.apply #[[MAP0]]()[%[[D0]]]
// CHECK-DAG:    %[[OUTER_D1:.+]] = affine.apply #[[MAP1]]()[%[[D1]]]
// CHECK:        %[[ARG0_EMPTY:.+]] = tensor.empty(%[[OUTER_D0]], %[[OUTER_D1]]) : tensor<?x?x8x2xf32>
// CHECK:        %[[PACK_ARG0:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:     inner_dims_pos = [0, 1] inner_tiles = [8, 2]
// CHECK-SAME:     into %[[ARG0_EMPTY]]
// CHECK:        %[[ELEM:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP2]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[PACK_ARG0]]
// CHECK-SAME:     outs(%[[DEST]]
// CHECK:        return %[[ELEM]] : tensor<?x?x8x2xf32>

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @elem_pack_transpose_inner_dims(%arg0: tensor<128x256xi32>, %dest: tensor<4x16x16x32xi32>) -> tensor<4x16x16x32xi32>{
  %init = tensor.empty() : tensor<128x256xi32>
  %elem = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<128x256xi32>)
      outs(%init : tensor<128x256xi32>) {
    ^bb0(%arg3: i32, %arg4: i32):
      %4 = arith.addi %arg3, %arg3 : i32
      linalg.yield %4 : i32
  } -> tensor<128x256xi32>
  %pack = tensor.pack %elem
    inner_dims_pos = [1, 0]
    inner_tiles = [16, 32]
    into %dest : tensor<128x256xi32> -> tensor<4x16x16x32xi32>
  return %pack : tensor<4x16x16x32xi32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK:      func.func @elem_pack_transpose_inner_dims
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[DEST:[a-zA-Z0-9]+]]
// CHECK:        %[[ARG0_EMPTY:.+]] = tensor.empty() : tensor<4x16x16x32xi32>
// CHECK:        %[[PACK_ARG0:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:     inner_dims_pos = [1, 0] inner_tiles = [16, 32]
// CHECK-SAME:     into %[[ARG0_EMPTY]]
// CHECK:        %[[ELEM:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[PACK_ARG0]]
// CHECK-SAME:     outs(%[[DEST]]
// CHECK:        return %[[ELEM]] : tensor<4x16x16x32xi32>

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @elem_pack_transpose_outer_dims(%arg0: tensor<128x256xi32>, %dest: tensor<16x4x32x16xi32>) -> tensor<16x4x32x16xi32>{
  %init = tensor.empty() : tensor<128x256xi32>
  %elem = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<128x256xi32>)
      outs(%init : tensor<128x256xi32>) {
    ^bb0(%arg3: i32, %arg4: i32):
      %4 = arith.addi %arg3, %arg3 : i32
      linalg.yield %4 : i32
  } -> tensor<128x256xi32>
  %pack = tensor.pack %elem
    outer_dims_perm = [1, 0]
    inner_dims_pos = [0, 1]
    inner_tiles = [32, 16]
    into %dest : tensor<128x256xi32> -> tensor<16x4x32x16xi32>
  return %pack : tensor<16x4x32x16xi32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK:      func.func @elem_pack_transpose_outer_dims
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[DEST:[a-zA-Z0-9]+]]
// CHECK:        %[[ARG0_EMPTY:.+]] = tensor.empty() : tensor<16x4x32x16xi32>
// CHECK:        %[[PACK_ARG0:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:     outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:     into %[[ARG0_EMPTY]] : tensor<128x256xi32> -> tensor<16x4x32x16xi32>
// CHECK:        %[[ELEM:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP0]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[PACK_ARG0]]
// CHECK-SAME:     outs(%[[DEST]]
// CHECK:        return %[[ELEM]] : tensor<16x4x32x16xi32>

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @elem_pack_transpose_inner_and_outer_dims(%arg0: tensor<128x256xi32>, %dest: tensor<16x4x16x32xi32>) -> tensor<16x4x16x32xi32>{
  %init = tensor.empty() : tensor<128x256xi32>
  %elem = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<128x256xi32>)
      outs(%init : tensor<128x256xi32>) {
    ^bb0(%arg3: i32, %arg4: i32):
      %4 = arith.addi %arg3, %arg3 : i32
      linalg.yield %4 : i32
  } -> tensor<128x256xi32>
  %pack = tensor.pack %elem
    outer_dims_perm = [1, 0]
    inner_dims_pos = [1, 0]
    inner_tiles = [16, 32]
    into %dest : tensor<128x256xi32> -> tensor<16x4x16x32xi32>
  return %pack : tensor<16x4x16x32xi32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK:      func.func @elem_pack_transpose_inner_and_outer_dims
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[DEST:[a-zA-Z0-9]+]]
// CHECK:        %[[ARG0_EMPTY:.+]] = tensor.empty() : tensor<16x4x16x32xi32>
// CHECK:        %[[PACK_ARG0:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:     outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 32]
// CHECK-SAME:     into %[[ARG0_EMPTY]]
// CHECK:        %[[ELEM:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP0]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[PACK_ARG0]]
// CHECK-SAME:     outs(%[[DEST]]
// CHECK:        return %[[ELEM]] : tensor<16x4x16x32xi32>

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
func.func @dynamic_broadcast_pack(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %dest: tensor<?x?x8x2xf32>) -> tensor<?x?x8x2xf32>
{
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %1 = tensor.dim %arg1, %c0 : tensor<?xf32>
  %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map1, #map2, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %4 = arith.addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  %4 = tensor.pack %3
    inner_dims_pos = [0, 1]
    inner_tiles = [8, 2]
    into %dest : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
  return %4 : tensor<?x?x8x2xf32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK:      func.func @dynamic_broadcast_pack
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[DEST:[a-zA-Z0-9]+]]
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-DAG:    %[[OUTER_D0:.+]] = affine.apply #[[MAP0]]()[%[[D0]]]
// CHECK:        %[[ARG0_EMPTY:.+]] = tensor.empty(%[[OUTER_D0]]) : tensor<?x8xf32>
// CHECK:        %[[PACK_ARG0:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:     inner_dims_pos = [0] inner_tiles = [8]
// CHECK-SAME:     into %[[ARG0_EMPTY]]
// CHECK-DAG:    %[[D1:.+]] = tensor.dim %[[ARG1]], %[[C0]]
// CHECK-DAG:    %[[OUTER_D1:.+]] = affine.apply #[[MAP1]]()[%[[D1]]]
// CHECK:        %[[ARG1_EMPTY:.+]] = tensor.empty(%[[OUTER_D1]]) : tensor<?x2xf32>
// CHECK:        %[[PACK_ARG1:.+]] = tensor.pack %[[ARG1]]
// CHECK-SAME:     inner_dims_pos = [0] inner_tiles = [2]
// CHECK-SAME:     into %[[ARG1_EMPTY]]
// CHECK:        %[[ELEM:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[PACK_ARG0]], %[[PACK_ARG0]]
// CHECK-SAME:     outs(%[[DEST]]
// CHECK:        return %[[ELEM]] : tensor<?x?x8x2xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @elem_pack_transpose_inner_and_outer_dims2(%arg0: tensor<64xf32>, %dest: tensor<1x2x56x57x32xf32>) -> tensor<1x2x56x57x32xf32> {
  %0 = tensor.empty() : tensor<1x56x57x64xf32>
  %1 = linalg.generic {
      indexing_maps = [#map, #map1],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<64xf32>)
    outs(%0 : tensor<1x56x57x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x56x57x64xf32>
  %2 = tensor.pack %1 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %dest : tensor<1x56x57x64xf32> -> tensor<1x2x56x57x32xf32>
  return %2 : tensor<1x2x56x57x32xf32>
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK:     func.func @elem_pack_transpose_inner_and_outer_dims2
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[DEST:[a-zA-Z0-9]+]]
// CHECK:       %[[ARG0_EMPTY:.+]] = tensor.empty() : tensor<2x32xf32>
// CHECK:       %[[PACKED_ARG0:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:    inner_dims_pos = [0] inner_tiles = [32]
// CHECK-SAME:  into %[[ARG0_EMPTY]]
// CHECK:       %[[RES:.+]] = linalg.generic
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:    ins(%[[PACKED_ARG0]]
// CHECK-SAME:    outs(%[[DEST]]

// -----

func.func @transpose_pack(%arg0: tensor<100x128x200x256xi32>, %arg1: tensor<100xi32>, %arg2: tensor<128xi32>, %dest: tensor<100x200x4x16x16x32xi32>) -> tensor<100x200x4x16x16x32xi32>
{
  %init_transpose = tensor.empty() : tensor<100x200x128x256xi32>
  %transpose = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0)>,
                       affine_map<(d0, d1, d2, d3) -> (d1)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0, %arg1, %arg2 : tensor<100x128x200x256xi32>, tensor<100xi32>, tensor<128xi32>)
      outs(%init_transpose : tensor<100x200x128x256xi32>) {
    ^bb0(%b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32):
      %0 = arith.addi %b0, %b1 : i32
      %1 = arith.addi %0, %b2 : i32
      linalg.yield %1 : i32
    } -> tensor<100x200x128x256xi32>
  %4 = tensor.pack %transpose
    inner_dims_pos = [3, 2]
    inner_tiles = [16, 32]
    into %dest : tensor<100x200x128x256xi32> -> tensor<100x200x4x16x16x32xi32>
  return %4 : tensor<100x200x4x16x16x32xi32>
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d5)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d3, d4, d5)>
// CHECK:     func.func @transpose_pack
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[DEST:[a-zA-Z0-9]+]]
// CHECK:       %[[ARG0_EMPTY:.+]] = tensor.empty() : tensor<100x4x200x16x16x32xi32>
// CHECK:       %[[PACKED_ARG0:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:    inner_dims_pos = [3, 1] inner_tiles = [16, 32]
// CHECK-SAME:  into %[[ARG0_EMPTY]]
// CHECK:       %[[ARG2_EMPTY:.+]] = tensor.empty() : tensor<4x32xi32>
// CHECK:       %[[PACKED_ARG2:.+]] = tensor.pack %[[ARG2]]
// CHECK-SAME:    inner_dims_pos = [0] inner_tiles = [32]
// CHECK-SAME:  into %[[ARG2_EMPTY]]
// CHECK:       %[[RES:.+]] = linalg.generic
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]]]
// CHECK-SAME:    ins(%[[PACKED_ARG0]], %[[ARG1]], %[[PACKED_ARG2]]
// CHECK-SAME:    outs(%[[DEST]]

// -----

func.func @affine_constant_expr_pack(%arg0: tensor<100x128x200x256xi32>, %arg1: tensor<100x1x1x1xi32>, %arg2: tensor<1x128x1x1xi32>, %dest: tensor<100x200x4x16x16x32xi32>) -> tensor<100x200x4x16x16x32xi32>
{
  %init_transpose = tensor.empty() : tensor<100x200x128x256xi32>
  %transpose = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, 0, 0, 0)>,
                       affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0, %arg1, %arg2 : tensor<100x128x200x256xi32>, tensor<100x1x1x1xi32>, tensor<1x128x1x1xi32>)
      outs(%init_transpose : tensor<100x200x128x256xi32>) {
    ^bb0(%b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32):
      %0 = arith.addi %b0, %b1 : i32
      %1 = arith.addi %0, %b2 : i32
      linalg.yield %1 : i32
    } -> tensor<100x200x128x256xi32>
  %4 = tensor.pack %transpose
    inner_dims_pos = [3, 2]
    inner_tiles = [16, 32]
    into %dest : tensor<100x200x128x256xi32> -> tensor<100x200x4x16x16x32xi32>
  return %4 : tensor<100x200x4x16x16x32xi32>
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, 0, 0, 0)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d1, 0, 0, d5)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d3, d4, d5)>
// CHECK:     func.func @affine_constant_expr_pack
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[DEST:[a-zA-Z0-9]+]]
// CHECK:       %[[ARG0_EMPTY:.+]] = tensor.empty() : tensor<100x4x200x16x16x32xi32>
// CHECK:       %[[PACKED_ARG0:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:    inner_dims_pos = [3, 1] inner_tiles = [16, 32]
// CHECK-SAME:  into %[[ARG0_EMPTY]]
// CHECK:       %[[ARG2_EMPTY:.+]] = tensor.empty() : tensor<1x4x1x1x32xi32>
// CHECK:       %[[PACKED_ARG2:.+]] = tensor.pack %[[ARG2]]
// CHECK-SAME:    inner_dims_pos = [1] inner_tiles = [32]
// CHECK-SAME:  into %[[ARG2_EMPTY]]
// CHECK:       %[[RES:.+]] = linalg.generic
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]]]
// CHECK-SAME:    ins(%[[PACKED_ARG0]], %[[ARG1]], %[[PACKED_ARG2]]
// CHECK-SAME:    outs(%[[DEST]]

// -----

func.func @transpose_pack_with_outer_dims(%arg0: tensor<100x128x200x256xi32>, %arg1: tensor<100xi32>, %arg2: tensor<128xi32>, %dest: tensor<200x4x16x100x16x32xi32>) -> tensor<200x4x16x100x16x32xi32>
{
  %init_transpose = tensor.empty() : tensor<100x200x128x256xi32>
  %transpose = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0)>,
                       affine_map<(d0, d1, d2, d3) -> (d1)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0, %arg1, %arg2 : tensor<100x128x200x256xi32>, tensor<100xi32>, tensor<128xi32>)
      outs(%init_transpose : tensor<100x200x128x256xi32>) {
    ^bb0(%b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32):
      %0 = arith.addi %b0, %b1 : i32
      %1 = arith.addi %0, %b2 : i32
      linalg.yield %1 : i32
    } -> tensor<100x200x128x256xi32>
  %4 = tensor.pack %transpose
    outer_dims_perm = [1, 2, 3, 0]
    inner_dims_pos = [3, 2]
    inner_tiles = [16, 32]
    into %dest : tensor<100x200x128x256xi32> -> tensor<200x4x16x100x16x32xi32>
  return %4 : tensor<200x4x16x100x16x32xi32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d5)>
// CHECK:     func.func @transpose_pack_with_outer_dims
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[DEST:[a-zA-Z0-9]+]]
// CHECK: %[[ARG0_EMPTY:.+]] = tensor.empty() : tensor<200x4x16x100x16x32xi32>
// CHECK: %[[PACKED_ARG0:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:  outer_dims_perm = [2, 1, 3, 0] inner_dims_pos = [3, 1] inner_tiles = [16, 32]
// CHECK-SAME:  into %[[ARG0_EMPTY]]
// CHECK: %[[ARG2_EMPTY:.+]] = tensor.empty() : tensor<4x32xi32>
// CHECK: %[[PACKED_ARG2:.+]] = tensor.pack %[[ARG2]]
// CHECK-SAME:  inner_dims_pos = [0] inner_tiles = [32]
// CHECK-SAME:  into %[[ARG2_EMPTY]]
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]], #[[MAP]]]
// CHECK-SAME:  ins(%[[PACKED_ARG0]], %[[ARG1]], %[[PACKED_ARG2]]
// CHECK-SAME:  outs(%[[DEST]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @elem_pack_transpose_outer_dims(%arg0: tensor<128x256xi32>, %init: tensor<128x256xi32>) -> tensor<16x4x32x16xi32>{
  %elem = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<128x256xi32>)
      outs(%init : tensor<128x256xi32>) {
    ^bb0(%arg3: i32, %arg4: i32):
      %4 = arith.addi %arg3, %arg4 : i32
      linalg.yield %4 : i32
  } -> tensor<128x256xi32>
  %empty = tensor.empty() : tensor<16x4x32x16xi32>
  %pack = tensor.pack %elem
    outer_dims_perm = [1, 0]
    inner_dims_pos = [0, 1]
    inner_tiles = [32, 16]
    into %empty : tensor<128x256xi32> -> tensor<16x4x32x16xi32>
  return %pack : tensor<16x4x32x16xi32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func.func @elem_pack_transpose_outer_dims
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]
// CHECK: %[[ARG1_EMPTY:.+]] = tensor.empty() : tensor<16x4x32x16xi32>
// CHECK: %[[PACKED_ARG1:.+]] = tensor.pack %[[ARG1]]
// CHECK-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[ARG1_EMPTY]]
// CHECK: %[[ARG0_EMPTY:.+]] = tensor.empty() : tensor<16x4x32x16xi32>
// CHECK: %[[PACKED_ARG0:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[ARG0_EMPTY]]
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME:  ins(%[[PACKED_ARG0]]
// CHECK-SAME:  outs(%[[PACKED_ARG1]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @unpack_on_output(%arg0: tensor<12x2x56x56x32xf32>) -> tensor<12x56x56x64xf32> {
  %0 = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
  %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<12x56x56x64xf32>) {
    ^bb0(%out: f32):
      %3 = arith.addf %out, %out : f32
      linalg.yield %3 : f32
  } -> tensor<12x56x56x64xf32>
  return %2 : tensor<12x56x56x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: func.func @unpack_on_output
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]
// CHECK: %[[ARG0_EMPTY_UNPACK:.+]] = tensor.empty() : tensor<12x56x56x64xf32>
// CHECK: %[[UNPACKED_ARG0:.+]] = tensor.unpack %[[ARG0]]
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:  into %[[ARG0_EMPTY_UNPACK]]
// CHECK: %[[ARG0_EMPTY_PACK:.+]] = tensor.empty() : tensor<12x2x56x56x32xf32>
// CHECK: %[[PACKED_ARG0:.+]] = tensor.pack %[[UNPACKED_ARG0]]
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:  into %[[ARG0_EMPTY_PACK]]
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]]]
// CHECK-SAME:  outs(%[[PACKED_ARG0]]
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[RES]]
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:  into %[[ARG0_EMPTY_UNPACK]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @unpack_on_input(%arg0: tensor<12x2x56x56x32xf32>, %init: tensor<12x56x56x64xf32>) -> tensor<12x56x56x64xf32> {
  %0 = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1: tensor<12x56x56x64xf32>) outs(%init : tensor<12x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %out : f32
      linalg.yield %3 : f32
  } -> tensor<12x56x56x64xf32>
  return %2 : tensor<12x56x56x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: func.func @unpack_on_input
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
// CHECK: %[[ARG0_UNPACK_EMPTY:.+]] = tensor.empty() : tensor<12x56x56x64xf32>
// CHECK: %[[UNPACKED_ARG0:.+]] = tensor.unpack %[[ARG0]] 
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] 
// CHECK-SAME:  into %[[ARG0_UNPACK_EMPTY]]
// CHECK: %[[ARG1_PACK_EMPTY:.+]] = tensor.empty() : tensor<12x2x56x56x32xf32>
// CHECK: %[[ARG1_PACK:.+]] = tensor.pack %[[ARG1]] 
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] 
// CHECK-SAME:  into %[[ARG1_PACK_EMPTY]]
// CHECK: %[[ARG0_PACK_EMPTY:.+]] = tensor.empty() : tensor<12x2x56x56x32xf32>
// CHECK: %[[ARG0_PACK:.+]] = tensor.pack %[[UNPACKED_ARG0]] 
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] 
// CHECK-SAME:  into %[[ARG0_PACK_EMPTY]]
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME:  ins(%[[ARG0_PACK]]
// CHECK-SAME:  outs(%[[ARG1_PACK]]
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[RES]] 
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] 
// CHECK-SAME:  into %[[ARG0_UNPACK_EMPTY]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @unpack_element_type_change(%arg0: tensor<12x2x56x56x32xf32>, %init: tensor<12x56x56x64xf16>) -> tensor<12x56x56x64xf16> {
  %0 = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1: tensor<12x56x56x64xf32>) outs(%init : tensor<12x56x56x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %3 = arith.truncf %in : f32 to f16
      linalg.yield %3 : f16
  } -> tensor<12x56x56x64xf16>
  return %2 : tensor<12x56x56x64xf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: func.func @unpack_element_type_change
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
// CHECK: %[[ARG0_UNPACK_EMPTY:.+]] = tensor.empty() : tensor<12x56x56x64xf32>
// CHECK: %[[UNPACKED_ARG0:.+]] = tensor.unpack %[[ARG0]]
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:  into %[[ARG0_UNPACK_EMPTY]]
// CHECK: %[[ARG1_PACK_EMPTY:.+]] = tensor.empty() : tensor<12x2x56x56x32xf16>
// CHECK: %[[ARG1_PACK:.+]] = tensor.pack %[[ARG1]]
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:  into %[[ARG1_PACK_EMPTY]]
// CHECK: %[[ARG0_PACK_EMPTY:.+]] = tensor.empty() : tensor<12x2x56x56x32xf32>
// CHECK: %[[ARG0_PACK:.+]] = tensor.pack %[[UNPACKED_ARG0]]
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:  into %[[ARG0_PACK_EMPTY]]
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME:  ins(%[[ARG0_PACK]]
// CHECK-SAME:  outs(%[[ARG1_PACK]]
// CHECK: %[[ARG0_NEW_EMPTY_UNPACK:.+]] = tensor.empty() : tensor<12x56x56x64xf16>
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[RES]]
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:  into %[[ARG0_NEW_EMPTY_UNPACK]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @forward_tensor_empty(%arg0: tensor<12x2x56x56x32xf32>) -> tensor<12x56x56x64xf32> {
  %init = tensor.empty() : tensor<12x56x56x64xf32>
  %0 = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1: tensor<12x56x56x64xf32>) outs(%init : tensor<12x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %in : f32
      linalg.yield %3 : f32
  } -> tensor<12x56x56x64xf32>
  return %2 : tensor<12x56x56x64xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: func.func @forward_tensor_empty
// CHECK-SAME:  %[[ARG0:[a-zA-Z0-9]+]]
// CHECK: %[[ARG0_UNPACK_EMPTY:.+]] = tensor.empty() : tensor<12x56x56x64xf32>
// CHECK: %[[UNPACKED_ARG0:.+]] = tensor.unpack %[[ARG0]] 
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] 
// CHECK-SAME:  into %[[ARG0_UNPACK_EMPTY]]
// CHECK: %[[DEST:.+]] = tensor.empty() : tensor<12x2x56x56x32xf32>
// CHECK: %[[ARG0_PACK_EMPTY:.+]] = tensor.empty() : tensor<12x2x56x56x32xf32>
// CHECK: %[[PACKED_ARG0:.+]] = tensor.pack %[[UNPACKED_ARG0]] 
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] 
// CHECK-SAME:  into %[[ARG0_PACK_EMPTY]]
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME:  ins(%[[PACKED_ARG0]]
// CHECK-SAME:  outs(%[[DEST]]
// CHECK: %[[UNPACKED:.+]] = tensor.unpack %[[RES]]
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] 
// CHECK-SAME:  into %[[ARG0_UNPACK_EMPTY]]

// -----

func.func @pad_valid_propagation(%arg0: tensor<1x2x56x56x32xf32>) -> tensor<1x58x58x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
  %padded = tensor.pad %1 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
    tensor.yield %cst : f32
  } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>
  return %padded : tensor<1x58x58x64xf32>
}

// CHECK: func.func @pad_valid_propagation(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x2x56x56x32xf32>)
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[PADDED:.+]] = tensor.pad %[[ARG0]] low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0]
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x58x58x64xf32>
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[PADDED]] 
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] 
// CHECK-SAME:  into %[[EMPTY]] : tensor<1x2x58x58x32xf32> -> tensor<1x58x58x64xf32>

// -----

func.func @pad_valid_propagation(%arg0: tensor<1x2x56x56x32xf32>) -> tensor<2x58x58x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
  %padded = tensor.pad %1 low[1, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
    tensor.yield %cst : f32
  } : tensor<1x56x56x64xf32> to tensor<2x58x58x64xf32>
  return %padded : tensor<2x58x58x64xf32>
}

// CHECK: func.func @pad_valid_propagation(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x2x56x56x32xf32>)
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[PADDED:.+]] = tensor.pad %[[ARG0]] low[1, 0, 1, 1, 0] high[0, 0, 1, 1, 0]
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<2x58x58x64xf32>
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[PADDED]]
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME:  into %[[EMPTY]] : tensor<2x2x58x58x32xf32> -> tensor<2x58x58x64xf32>

// -----

func.func @pad_along_unpacked_dim(%arg0: tensor<1x2x56x56x32xf32>) -> tensor<1x58x58x66xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
  %padded = tensor.pad %1 low[0, 1, 1, 1] high[0, 1, 1, 1] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
    tensor.yield %cst : f32
  } : tensor<1x56x56x64xf32> to tensor<1x58x58x66xf32>
  return %padded : tensor<1x58x58x66xf32>
}

// CHECK: func.func @pad_along_unpacked_dim(
// CHECK: %[[ARG0:.+]]: tensor<1x2x56x56x32xf32>)
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x56x56x64xf32>
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[ARG0]] 
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] 
// CHECK-SAME:  into %[[EMPTY]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
// CHECK: %[[PADDED:.+]] = tensor.pad %[[UNPACK]] low[0, 1, 1, 1] high[0, 1, 1, 1]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @would_break_dominance(%arg0: tensor<128x256xi32>) -> tensor<4x16x16x32xi32>{
  %init = tensor.empty() : tensor<128x256xi32>
  %elem = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<128x256xi32>)
      outs(%init : tensor<128x256xi32>) {
    ^bb0(%arg3: i32, %arg4: i32):
      %4 = arith.addi %arg3, %arg3 : i32
      linalg.yield %4 : i32
  } -> tensor<128x256xi32>
  %dest = bufferization.alloc_tensor() : tensor<4x16x16x32xi32>
  %pack = tensor.pack %elem
    inner_dims_pos = [1, 0]
    inner_tiles = [16, 32]
    into %dest : tensor<128x256xi32> -> tensor<4x16x16x32xi32>
  return %pack : tensor<4x16x16x32xi32>
}

// CHECK: func.func @would_break_dominance(
// CHECK-SAME: %[[ARG0:.+]]: tensor<128x256xi32>)
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<128x256xi32>
// CHECK-NEXT: %[[GEN:.+]] = linalg.generic
// CHECK-SAME:  ins(%[[ARG0]]
// CHECK-SAME:  outs(%[[EMPTY]]
// CHECK: %[[ALLOC:.+]] = bufferization.alloc_tensor() : tensor<4x16x16x32xi32>
// CHECK-NEXT: %{{.+}} = tensor.pack %[[GEN]]
// CHECK-SAME:  inner_dims_pos = [1, 0] inner_tiles = [16, 32] 
// CHECK-SAME:  into %[[ALLOC]]

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> ()>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @scalar_tensor(%arg0 : tensor<f32>) -> tensor<1x32x7x7x32xf32> {
  %empty_gen = tensor.empty() : tensor<1x7x7x1024xf32>
  %gen = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<f32>) outs(%empty_gen : tensor<1x7x7x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x7x7x1024xf32>
  %empty_pack = tensor.empty() : tensor<1x32x7x7x32xf32>
  %pack = tensor.pack %gen outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %empty_pack : tensor<1x7x7x1024xf32> -> tensor<1x32x7x7x32xf32>
  return %pack : tensor<1x32x7x7x32xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> ()>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: func.func @scalar_tensor
// CHECK-SAME: %[[ARG0:.+]]: tensor<f32>)
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x32x7x7x32xf32>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[ARG0]]
// CHECK-SAME: outs(%[[EMPTY]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @unpack_empty_inner_dims(%arg0: tensor<12x64x56x56xf32>) -> tensor<12x56x56x64xf32> {
  %init = tensor.empty() : tensor<12x56x56x64xf32>
  %0 = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [] inner_tiles = [] into %0 : tensor<12x64x56x56xf32> -> tensor<12x56x56x64xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1: tensor<12x56x56x64xf32>) outs(%init : tensor<12x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %in : f32
      linalg.yield %3 : f32
  } -> tensor<12x56x56x64xf32>
  return %2 : tensor<12x56x56x64xf32>
}

// CHECK: func.func @unpack_empty_inner_dims
// CHECK: %[[UNPACKED_ARG0:.+]] = tensor.unpack
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [] inner_tiles = [] 
// CHECK: %[[PACKED_ARG0:.+]] = tensor.pack %[[UNPACKED_ARG0]] 
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [] inner_tiles = [] 
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME:  ins(%[[PACKED_ARG0]]
// CHECK: %[[UNPACKED:.+]] = tensor.unpack %[[RES]]
// CHECK-SAME:  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [] inner_tiles = [] 

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @reduction_pack_transpose_inner_dims(%arg0: tensor<128x256x32xi32>, %dest: tensor<4x16x16x32xi32>) -> tensor<4x16x16x32xi32>{
  %init = tensor.empty() : tensor<128x256xi32>
  %elem = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%arg0 : tensor<128x256x32xi32>)
      outs(%init : tensor<128x256xi32>) {
    ^bb0(%arg3: i32, %arg4: i32):
      %4 = arith.addi %arg3, %arg4 : i32
      linalg.yield %4 : i32
  } -> tensor<128x256xi32>
  %pack = tensor.pack %elem
    inner_dims_pos = [1, 0]
    inner_tiles = [16, 32]
    into %dest : tensor<128x256xi32> -> tensor<4x16x16x32xi32>
  return %pack : tensor<4x16x16x32xi32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
// CHECK:      func.func @reduction_pack_transpose_inner_dims
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[DEST:[a-zA-Z0-9]+]]
// CHECK:        %[[ORIG_INIT:.+]] = tensor.empty() : tensor<128x256xi32>
// CHECK:        %[[INIT_EMPTY:.+]] = tensor.empty() : tensor<4x16x16x32xi32>
// CHECK:        %[[PACK_INIT:.+]] = tensor.pack %[[ORIG_INIT]]
// CHECK:        %[[ARG0_EMPTY:.+]] = tensor.empty() : tensor<4x16x32x16x32xi32>
// CHECK:        %[[PACK_ARG0:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:     inner_dims_pos = [1, 0] inner_tiles = [16, 32]
// CHECK-SAME:     into %[[ARG0_EMPTY]]
// CHECK:        %[[RED:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[PACK_ARG0]]
// CHECK-SAME:     outs(%[[PACK_INIT]]
// CHECK:        return %[[RED]] : tensor<4x16x16x32xi32>

// -----

func.func @reduction_pack_with_outer_dims(%arg0: tensor<100x128x200x256xi32>, %arg1: tensor<100xi32>, %arg2: tensor<128xi32>) -> tensor<4x16x100x16x32xi32>
{
  %init_reduction = tensor.empty() : tensor<100x128x256xi32>
  %reduction = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0)>,
                       affine_map<(d0, d1, d2, d3) -> (d1)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>],
      iterator_types = ["parallel", "parallel", "reduction", "parallel"]}
      ins(%arg0, %arg1, %arg2 : tensor<100x128x200x256xi32>, tensor<100xi32>, tensor<128xi32>)
      outs(%init_reduction : tensor<100x128x256xi32>) {
    ^bb0(%b0 : i32, %b1 : i32, %b2 : i32, %b3 : i32):
      %0 = arith.addi %b0, %b1 : i32
      %1 = arith.addi %0, %b2 : i32
      %2 = arith.addi %1, %b3 : i32
      linalg.yield %2 : i32
    } -> tensor<100x128x256xi32>
  %init_pack = tensor.empty() : tensor<4x16x100x16x32xi32>
  %4 = tensor.pack %reduction
    outer_dims_perm = [1, 2, 0]
    inner_dims_pos = [2, 1]
    inner_tiles = [16, 32]
    into %init_pack : tensor<100x128x256xi32> -> tensor<4x16x100x16x32xi32>
  return %4 : tensor<4x16x100x16x32xi32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d5)>
// CHECK:     func.func @reduction_pack_with_outer_dims
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]
// CHECK: %[[INIT:.+]] = tensor.empty() : tensor<100x128x256xi32>
// CHECK: %[[INIT_EMPTY:.+]] = tensor.empty() : tensor<4x16x100x16x32xi32>
// CHECK: %[[PACKED_INIT:.+]] = tensor.pack %[[INIT]]
// CHECK-SAME:  outer_dims_perm = [1, 2, 0] inner_dims_pos = [2, 1] inner_tiles = [16, 32]
// CHECK-SAME:  into %[[INIT_EMPTY]]
// CHECK: %[[ARG0_EMPTY:.+]] = tensor.empty() : tensor<4x16x200x100x16x32xi32>
// CHECK: %[[PACKED_ARG0:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:  outer_dims_perm = [1, 3, 2, 0] inner_dims_pos = [3, 1] inner_tiles = [16, 32]
// CHECK-SAME:  into %[[ARG0_EMPTY]]
// CHECK: %[[ARG2_EMPTY:.+]] = tensor.empty() : tensor<4x32xi32>
// CHECK: %[[PACKED_ARG2:.+]] = tensor.pack %[[ARG2]]
// CHECK-SAME:  inner_dims_pos = [0] inner_tiles = [32]
// CHECK-SAME:  into %[[ARG2_EMPTY]]
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]], #[[MAP3]]]
// CHECK-SAME:  ins(%[[PACKED_ARG0]], %[[ARG1]], %[[PACKED_ARG2]]
// CHECK-SAME:  outs(%[[PACKED_INIT]]

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d3)>
func.func @unpack_different_destination_shape(%arg0: tensor<1x1x1080x1920x16xi32>) -> tensor<16x540x960xi32>{
  %init = tensor.empty() : tensor<16x540x960xi32>
  %filter = tensor.empty() : tensor<2x2xi32>
  %empty = tensor.empty() : tensor<1x16x1080x1920xi32>
  %unpack = tensor.unpack %arg0
      inner_dims_pos = [1]
      inner_tiles = [16]
      into %empty : tensor<1x1x1080x1920x16xi32> -> tensor<1x16x1080x1920xi32>
  %pool = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%unpack, %filter : tensor<1x16x1080x1920xi32>, tensor<2x2xi32>)
      outs(%init : tensor<16x540x960xi32>) {
    ^bb0(%in: i32, %in_1: i32, %out: i32):
      %max = arith.maxui %in, %out : i32
      linalg.yield %max : i32
  } -> tensor<16x540x960xi32>
  return %pool : tensor<16x540x960xi32>
}
// CHECK-DAG:  #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5, d6)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d3, d6)>
// CHECK:      func.func @unpack_different_destination_shape
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:        %[[FILTER:.+]] = tensor.empty() : tensor<2x2xi32>
// CHECK:        %[[INIT:.+]] = tensor.empty() : tensor<1x540x960x16xi32>
// CHECK:        %[[PACK_EMPTY:.+]] = tensor.empty() : tensor<1x1x1080x1920x16xi32>
// CHECK:        %[[PACK_ARG0:.+]] = tensor.pack
// CHECK-SAME:     inner_dims_pos = [1] inner_tiles = [16]
// CHECK-SAME:     into %[[PACK_EMPTY]]
// CHECK:        %[[POOL:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "parallel"]
// CHECK-SAME:     ins(%[[PACK_ARG0]], %[[FILTER]]
// CHECK-SAME:     outs(%[[INIT]]
// CHECK:        %[[UNPACK_NEW_DEST:.+]] = tensor.empty() : tensor<16x540x960xi32>
// CHECK:        %[[UNPACK:.+]] = tensor.unpack %[[POOL]]
// CHECK-SAME:     inner_dims_pos = [0] inner_tiles = [16]
// CHECK-SAME:     into %[[UNPACK_NEW_DEST]]
// CHECK:        return %[[UNPACK]] : tensor<16x540x960xi32>
