// RUN: mlir-opt --split-input-file %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @cast(
func.func @cast(%arg0: tensor<*xf32>, %arg1 : tensor<4x4xf32>, %arg2: tensor<?x?xf32>) {
  // CHECK: tensor.cast %{{.*}} : tensor<*xf32> to tensor<?x?xf32>
  %0 = tensor.cast %arg0 : tensor<*xf32> to tensor<?x?xf32>
  // CHECK: tensor.cast %{{.*}} : tensor<4x4xf32> to tensor<*xf32>
  %1 = tensor.cast %arg1 : tensor<4x4xf32> to tensor<*xf32>
  // CHECK: tensor.cast %{{.*}} : tensor<?x?xf32> to tensor<4x?xf32>
  %2 = tensor.cast %arg2 : tensor<?x?xf32> to tensor<4x?xf32>
  // CHECK: tensor.cast %{{.*}} : tensor<4x?xf32> to tensor<?x?xf32>
  %3 = tensor.cast %2 : tensor<4x?xf32> to tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func @concat(
func.func @concat(%arg0: tensor<4x7x3xf32>, %arg1 : tensor<4x4x3xf32>, %arg2: tensor<?x?x?xf32>) {
  // CHECK: tensor.concat dim(0) %{{.*}} : (tensor<4x7x3xf32>) -> tensor<4x7x3xf32>
  %0 = tensor.concat dim(0) %arg0 : (tensor<4x7x3xf32>) -> tensor<4x7x3xf32>
  // CHECK: tensor.concat dim(1) %{{.*}} : (tensor<4x7x3xf32>, tensor<4x4x3xf32>) -> tensor<4x11x3xf32>
  %1 = tensor.concat dim(1) %arg0, %arg1 : (tensor<4x7x3xf32>, tensor<4x4x3xf32>) -> tensor<4x11x3xf32>
  // CHECK: tensor.concat dim(2) %{{.*}} : (tensor<4x7x3xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = tensor.concat dim(2) %arg0, %arg2 : (tensor<4x7x3xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK: tensor.concat dim(1) %{{.*}} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x10x?xf32>
  %3 = tensor.concat dim(1) %arg2, %arg2 : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x10x?xf32>
  // CHECK: tensor.concat dim(1) %{{.*}} : (tensor<?x?x?xf32>, tensor<4x4x3xf32>, tensor<4x7x3xf32>) -> tensor<4x?x3xf32>
  %4 = tensor.concat dim(1) %arg2, %arg1, %arg0 : (tensor<?x?x?xf32>, tensor<4x4x3xf32>, tensor<4x7x3xf32>) -> tensor<4x?x3xf32>
  return
}

// -----

// CHECK-LABEL: func @empty(
//  CHECK-SAME:             %[[sz:.*]]: index
func.func @empty(%sz: index) -> tensor<5x?x6xf32> {
  // CHECK: tensor.empty(%[[sz]]) : tensor<5x?x6xf32>
  %0 = tensor.empty(%sz) : tensor<5x?x6xf32>
  return %0 : tensor<5x?x6xf32>
}

// -----

// CHECK-LABEL: func @empty_with_encoding(
//  CHECK-SAME:             %[[sz:.*]]: index
func.func @empty_with_encoding(%sz: index) -> tensor<5x?x6xf32, "foo"> {
  // CHECK: tensor.empty(%[[sz]]) : tensor<5x?x6xf32, "foo">
  %0 = tensor.empty(%sz) : tensor<5x?x6xf32, "foo">
  return %0 : tensor<5x?x6xf32, "foo">
}

// -----

// CHECK-LABEL:   func @extract(
// CHECK-SAME:                  %[[TENSOR:.*]]: tensor<?x?x?xf32>,
// CHECK-SAME:                  %[[INDEX:.*]]: index) {
func.func @extract(%arg0: tensor<?x?x?xf32>, %arg1: index) {
  // CHECK: tensor.extract %[[TENSOR]][%[[INDEX]], %[[INDEX]], %[[INDEX]]] : tensor<?x?x?xf32>
  %0 = tensor.extract %arg0[%arg1, %arg1, %arg1] : tensor<?x?x?xf32>
  return
}

// -----

// CHECK-LABEL:   func @insert(
// CHECK-SAME:                  %[[SCALAR:.*]]: f32
// CHECK-SAME:                  %[[INDEX:.*]]: index
// CHECK-SAME:                  %[[DEST1:.*]]: tensor<?x?x?xf32>
func.func @insert(%arg0: f32, %arg1: index, %arg2: tensor<?x?x?xf32>) {
  // CHECK: tensor.insert %[[SCALAR]] into %[[DEST1]][%[[INDEX]], %[[INDEX]], %[[INDEX]]] : tensor<?x?x?xf32>
  %0 = tensor.insert %arg0 into %arg2[%arg1, %arg1, %arg1] : tensor<?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: func @tensor.from_elements() {
func.func @tensor.from_elements() {
  %c0 = "arith.constant"() {value = 0: index} : () -> index
  // CHECK: tensor.from_elements %c0 : tensor<1xindex>
  %0 = tensor.from_elements %c0 : tensor<1xindex>

  %c1 = "arith.constant"() {value = 1: index} : () -> index
  // CHECK: tensor.from_elements %c0, %c1 : tensor<2xindex>
  %1 = tensor.from_elements %c0, %c1 : tensor<2xindex>

  %c0_f32 = "arith.constant"() {value = 0.0: f32} : () -> f32
  // CHECK: [[C0_F32:%.*]] = arith.constant
  // CHECK: tensor.from_elements [[C0_F32]] : tensor<1xf32>
  %2 = tensor.from_elements %c0_f32 : tensor<1xf32>

  // CHECK: tensor.from_elements : tensor<0xindex>
  %3 = tensor.from_elements : tensor<0xindex>

  // CHECK: tensor.from_elements %c0, %c1, %c0, %c1, %c0, %c1 : tensor<2x3xindex>
  %4 = tensor.from_elements %c0, %c1, %c0, %c1, %c0, %c1 : tensor<2x3xindex>

  // CHECK: tensor.from_elements %c0 : tensor<index>
  %5 = tensor.from_elements %c0 : tensor<index>
  return
}

// -----

// CHECK-LABEL: @tensor.generate
func.func @tensor.generate(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  %tnsr = tensor.generate %m, %n {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = arith.constant 8.0 : f32
      tensor.yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

// CHECK-LABEL: func @tensor_reshape
func.func @tensor_reshape(%unranked: tensor<*xf32>, %shape1: tensor<1xi32>,
         %shape2: tensor<2xi32>, %shape3: tensor<?xi32>) -> tensor<*xf32> {
  %dyn_vec = tensor.reshape %unranked(%shape1)
               : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xf32>
  %dyn_mat = tensor.reshape %dyn_vec(%shape2)
               : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %new_unranked = tensor.reshape %dyn_mat(%shape3)
               : (tensor<?x?xf32>, tensor<?xi32>) -> tensor<*xf32>
  return %new_unranked : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @slice({{.*}}) {
func.func @slice(%t: tensor<8x16x4xf32>, %idx : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: tensor.extract_slice
  // CHECK-SAME: tensor<8x16x4xf32> to tensor<?x?x?xf32>
  %1 = tensor.extract_slice %t[%c0, %c0, %c0][%idx, %idx, %idx][%c1, %c1, %c1]
    : tensor<8x16x4xf32> to tensor<?x?x?xf32>

  // CHECK: tensor.extract_slice
  // CHECK-SAME: tensor<8x16x4xf32> to tensor<4x4x4xf32>
  %2 = tensor.extract_slice %t[0, 2, 0][4, 4, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<4x4x4xf32>

  // CHECK: tensor.extract_slice
  // CHECK-SAME: tensor<8x16x4xf32> to tensor<4x4xf32>
  %3 = tensor.extract_slice %t[0, 2, 0][4, 1, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<4x4xf32>

  return
}

// -----

// CHECK-LABEL: func @insert_slice({{.*}}) {
func.func @insert_slice(
    %t: tensor<8x16x4xf32>,
    %td: tensor<8x?x4xf32>,
    %t2: tensor<16x32x8xf32>,
    %t3: tensor<4x4xf32>,
    %idx : index,
    %sz : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: tensor.insert_slice
  // CHECK-SAME: tensor<8x16x4xf32> into tensor<16x32x8xf32>
  %1 = tensor.insert_slice %t into %t2[%c0, %c0, %c0][8, 16, 4][%c1, %c1, %c1]
    : tensor<8x16x4xf32> into tensor<16x32x8xf32>

  // CHECK: tensor.insert_slice
  // CHECK-SAME: tensor<8x16x4xf32> into tensor<16x32x8xf32>
  %2 = tensor.insert_slice %t into %t2[%c0, %idx, %c0][8, 16, 4][%c1, 1, %c1]
    : tensor<8x16x4xf32> into tensor<16x32x8xf32>

  // CHECK: tensor.insert_slice
  // CHECK-SAME: tensor<4x4xf32> into tensor<8x16x4xf32>
  %3 = tensor.insert_slice %t3 into %t[0, 2, 0][4, 1, 4][1, 1, 1]
    : tensor<4x4xf32> into tensor<8x16x4xf32>

  // CHECK: tensor.insert_slice
  // CHECK-SAME: tensor<8x?x4xf32> into tensor<8x16x4xf32>
  %4 = tensor.insert_slice %td into %t[0, %idx, 0][8, %sz, 4][1, 1, 1]
    : tensor<8x?x4xf32> into tensor<8x16x4xf32>

  return
}

// -----

func.func @tensor_reshape_zero_dim(%arg0 : tensor<1x1xf32>, %arg1 : tensor<f32>)
    -> (tensor<f32>, tensor<1x1xf32>) {
  %0 = tensor.collapse_shape %arg0 [] : tensor<1x1xf32> into tensor<f32>
  %1 = tensor.expand_shape %0 [] : tensor<f32> into tensor<1x1xf32>
  return %0, %1 : tensor<f32>, tensor<1x1xf32>
}
// CHECK-LABEL: func @tensor_reshape_zero_dim
//       CHECK:   tensor.collapse_shape %{{.*}} [] : tensor<1x1xf32> into tensor<f32>
//       CHECK:   tensor.expand_shape %{{.*}} [] : tensor<f32> into tensor<1x1xf32>

// -----

func.func @legal_collapsing_reshape_dynamic_tensor
  (%arg0: tensor<?x?x?x4x?xf32>) -> tensor<?x?x?xf32>
{
  %0 = tensor.collapse_shape %arg0 [[0], [1], [2, 3, 4]] :
    tensor<?x?x?x4x?xf32> into tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
//      CHECK: func @legal_collapsing_reshape_dynamic_tensor
//      CHECK:   tensor.collapse_shape
// CHECK-SAME:    [0], [1], [2, 3, 4]

// -----

func.func @rank(%t : tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = tensor.rank %{{.*}} : tensor<4x4x?xf32>
  %0 = "tensor.rank"(%t) : (tensor<4x4x?xf32>) -> index

  // CHECK: %{{.*}} = tensor.rank %{{.*}} : tensor<4x4x?xf32>
  %1 = tensor.rank %t : tensor<4x4x?xf32>
  return
}

// -----

func.func @pad_dynamic(%arg0: tensor<1x2x2x?xf32>, %low: index, %high: index,
                  %pad_value: f32) -> tensor<6x?x?x?xf32> {
  %0 = tensor.pad %arg0 low[2, %low, 3, 3] high[3, 3, %high, 2] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %pad_value : f32
    } : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>
  return %0 : tensor<6x?x?x?xf32>
}
// CHECK-LABEL: func @pad_dynamic
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]
//  CHECK-SAME: %[[LOW:[a-zA-Z0-9_]*]]
//  CHECK-SAME: %[[HIGH:[a-zA-Z0-9_]*]]
//       CHECK:   tensor.pad %[[ARG0]]
//  CHECK-SAME:     low[2, %[[LOW]], 3, 3]
//  CHECK-SAME:     high[3, 3, %[[HIGH]], 2]
//       CHECK:    : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>

// -----

func.func @pad_static(%arg0: tensor<3x4xf32>, %pad_value: f32) -> tensor<6x9xf32> {
  %0 = tensor.pad %arg0 low[1, 2] high[2, 3] {
    ^bb0(%arg1 : index, %arg2 : index):
      tensor.yield %pad_value : f32
    } : tensor<3x4xf32> to tensor<6x9xf32>
  return %0 : tensor<6x9xf32>
}
// CHECK-LABEL: func @pad_static
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]
//       CHECK:   tensor.pad %[[ARG0]] low[1, 2] high[2, 3]
//       CHECK:    : tensor<3x4xf32> to tensor<6x9xf32>

// -----

func.func @pad_asymmetrical(%arg0: tensor<2x3xf32>, %ub0: index, %ub1: index,
                       %pad_value: f32) -> tensor<?x?xf32> {
  %0 = tensor.pad %arg0 low[0, 0] high[%ub0, %ub1] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<2x3xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @pad_asymmetrical
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]
//  CHECK-SAME: %[[UB0:[a-zA-Z0-9_]*]]
//  CHECK-SAME: %[[UB1:[a-zA-Z0-9_]*]]
//       CHECK:   tensor.pad %[[ARG0]]
//  CHECK-SAME:     low[0, 0]
//  CHECK-SAME:     high[%[[UB0]], %[[UB1]]]
//       CHECK:    : tensor<2x3xf32> to tensor<?x?xf32>

// -----

func.func @pad_to_static_size(%arg0: tensor<?x?xf32>, %ub0: index, %ub1: index,
                         %pad_value: f32) -> tensor<2x3xf32> {
  %0 = tensor.pad %arg0 low[0, 0] high[%ub0, %ub1] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<?x?xf32> to tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
// CHECK-LABEL: func @pad_to_static_size
//  CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]
//  CHECK-SAME: %[[UB0:[a-zA-Z0-9_]*]]
//  CHECK-SAME: %[[UB1:[a-zA-Z0-9_]*]]
//       CHECK:   tensor.pad %[[ARG0]]
//  CHECK-SAME:     low[0, 0]
//  CHECK-SAME:     high[%[[UB0]], %[[UB1]]]
//       CHECK:    : tensor<?x?xf32> to tensor<2x3xf32>

// -----

// CHECK-LABEL: func @test_splat_op
// CHECK-SAME: [[S:%arg[0-9]+]]: f32
func.func @test_splat_op(%s : f32) {
  // CHECK: tensor.splat [[S]] : tensor<8xf32>
  %v = tensor.splat %s : tensor<8xf32>

  // CHECK: tensor.splat [[S]] : tensor<4xf32>
  %u = "tensor.splat"(%s) : (f32) -> tensor<4xf32>
  return
}

// CHECK-LABEL: func @test_splat_op
// CHECK-SAME: [[S:arg[0-9]+]]: f32
// CHECK-SAME: [[M:arg[0-9]+]]: index
// CHECK-SAME: [[N:arg[0-9]+]]: index
func.func @test_splat_op_dynamic(%s: f32, %m: index, %n: index) {
  // CHECK: tensor.splat %[[S]][%[[M]], %[[N]]] : tensor<?x8x?xf32>
  %v = tensor.splat %s[%m, %n] : tensor<?x8x?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @gather_scatter(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<4x5x6xf32>,
// CHECK-SAME:  %[[ARG1:.*]]: tensor<1x3x2xindex>,
// CHECK-SAME:  %[[ARG2:.*]]: tensor<1x3x2xi32>) {
func.func @gather_scatter(
    %dest : tensor<4x5x6xf32>, %indices: tensor<1x3x2xindex>, %indices_i32: tensor<1x3x2xi32>) {
  // CHECK: %[[GATHER:.*]] = tensor.gather %[[ARG0]][%[[ARG2]]] gather_dims([1, 2]) unique : (tensor<4x5x6xf32>, tensor<1x3x2xi32>) -> tensor<1x3x4x1x1xf32>
  %gathered = tensor.gather %dest[%indices_i32] gather_dims([1, 2]) unique:
    (tensor<4x5x6xf32>, tensor<1x3x2xi32>) -> tensor<1x3x4x1x1xf32>
  // CHECK: %[[GATHER0:.*]] = tensor.gather %[[ARG0]][%[[ARG1]]] gather_dims([1, 2]) unique : (tensor<4x5x6xf32>, tensor<1x3x2xindex>) -> tensor<1x3x4xf32>
  %rank_reduced_gathered = tensor.gather %dest[%indices] gather_dims([1, 2]) unique:
    (tensor<4x5x6xf32>, tensor<1x3x2xindex>) -> tensor<1x3x4xf32>

  // CHECK: %{{.*}} = tensor.scatter %[[GATHER]] into %[[ARG0]][%[[ARG1]]] scatter_dims([1, 2]) unique : (tensor<1x3x4x1x1xf32>, tensor<4x5x6xf32>, tensor<1x3x2xindex>) -> tensor<4x5x6xf32>
  %scattered = tensor.scatter %gathered into %dest[%indices]
      scatter_dims([1, 2]) unique:
    (tensor<1x3x4x1x1xf32>, tensor<4x5x6xf32>, tensor<1x3x2xindex>) -> tensor<4x5x6xf32>
  // CHECK: %{{.*}} = tensor.scatter %[[GATHER0]] into %[[ARG0]][%[[ARG2]]] scatter_dims([1, 2]) unique : (tensor<1x3x4xf32>, tensor<4x5x6xf32>, tensor<1x3x2xi32>) -> tensor<4x5x6xf32>
  %rank_reduced_scattered = tensor.scatter %rank_reduced_gathered into %dest[%indices_i32]
      scatter_dims([1, 2]) unique:
    (tensor<1x3x4xf32>, tensor<4x5x6xf32>, tensor<1x3x2xi32>) -> tensor<4x5x6xf32>
  return
}

// -----

func.func @pack_nc_to_ncnc(%source: tensor<128x256xf32>, %dest: tensor<4x16x32x16xf32>) -> tensor<128x256xf32> {
  %0 = tensor.pack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<128x256xf32> -> tensor<4x16x32x16xf32>
  %1 = tensor.empty() : tensor<128x256xf32>
  %2 = tensor.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %1 : tensor<4x16x32x16xf32> -> tensor<128x256xf32>
  return %2 : tensor<128x256xf32>
}

// CHECK-LABEL: func.func @pack_nc_to_ncnc(
// CHECK-SAME:  %[[SOURCE:.*]]: tensor<128x256xf32>,
// CHECK-SAME:  %[[DEST:.*]]: tensor<4x16x32x16xf32>)
// CHECK: %[[PACKED:.*]] = tensor.pack %[[SOURCE]] inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %[[DEST]] : tensor<128x256xf32> -> tensor<4x16x32x16xf32>
// CHECK: %[[BUFF:.*]] = tensor.empty() : tensor<128x256xf32>
// CHECK: %{{.*}} = tensor.unpack %[[PACKED]] inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %[[BUFF]] : tensor<4x16x32x16xf32> -> tensor<128x256xf32>

// -----

func.func @pack_nc_to_ncnc_with_padding(%source: tensor<13x15xf32>, %dest: tensor<2x8x8x2xf32>, %padding: f32) -> tensor<13x15xf32> {
  %0 = tensor.pack %source padding_value(%padding : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %dest : tensor<13x15xf32> -> tensor<2x8x8x2xf32>
  %1 = tensor.empty() : tensor<13x15xf32>
  %2 = tensor.unpack %0 inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %1 : tensor<2x8x8x2xf32> -> tensor<13x15xf32>
  return %2 : tensor<13x15xf32>
}

// CHECK-LABEL: func.func @pack_nc_to_ncnc_with_padding(
// CHECK-SAME:  %[[SOURCE:.*]]: tensor<13x15xf32>,
// CHECK-SAME:  %[[DEST:.*]]: tensor<2x8x8x2xf32>,
// CHECK-SAME:  %[[PADDING:.*]]: f32)
// CHECK: %[[PACKED:.*]] = tensor.pack %[[SOURCE]] padding_value(%[[PADDING]] : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %[[DEST]] : tensor<13x15xf32> -> tensor<2x8x8x2xf32>
// CHECK: %[[BUFF:.*]] = tensor.empty() : tensor<13x15xf32>
// CHECK: %{{.*}} = tensor.unpack %[[PACKED]] inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %[[BUFF]] : tensor<2x8x8x2xf32> -> tensor<13x15xf32>

// -----

func.func @pack_ck_to_kcck(%source: tensor<128x256xf32>, %dest: tensor<16x4x32x16xf32>) -> tensor<128x256xf32> {
  %0 = tensor.pack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<128x256xf32> -> tensor<16x4x32x16xf32>
  %1 = tensor.empty() : tensor<128x256xf32>
  %2 = tensor.unpack %0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %1 : tensor<16x4x32x16xf32> -> tensor<128x256xf32>
  return %2 : tensor<128x256xf32>
}

// CHECK-LABEL: func.func @pack_ck_to_kcck(
// CHECK-SAME:  %[[SOURCE:.*]]: tensor<128x256xf32>,
// CHECK-SAME:  %[[DEST:.*]]: tensor<16x4x32x16xf32>)
// CHECK: %[[PACKED:.*]] = tensor.pack %[[SOURCE]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %[[DEST]] : tensor<128x256xf32> -> tensor<16x4x32x16xf32>
// CHECK: %[[BUFF:.*]] = tensor.empty() : tensor<128x256xf32>
// CHECK: %{{.*}} = tensor.unpack %[[PACKED]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %[[BUFF]] : tensor<16x4x32x16xf32> -> tensor<128x256xf32>

// -----

func.func @pad_and_pack_fully_dynamic(%source: tensor<?x?xf32>, %dest: tensor<?x?x?x?xf32>, %pad: f32, %tile_n : index, %tile_m : index) -> tensor<?x?x?x?xf32> {
  %0 = tensor.pack %source padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [%tile_n, %tile_m] into %dest : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: func.func @pad_and_pack_fully_dynamic(
// CHECK-SAME:  %[[SOURCE:.*]]: tensor<?x?xf32>,
// CHECK-SAME:  %[[DEST:.*]]: tensor<?x?x?x?xf32>,
// CHECK-SAME:  %[[PAD:.*]]: f32,
// CHECK-SAME:  %[[TILE_N:.*]]: index,
// CHECK-SAME:  %[[TILE_M:.*]]: index)
// CHECK: %{{.*}} = tensor.pack %[[SOURCE]] padding_value(%[[PAD]] : f32) inner_dims_pos = [0, 1] inner_tiles = [%[[TILE_N]], %[[TILE_M]]] into %[[DEST]] : tensor<?x?xf32> -> tensor<?x?x?x?xf32>

// -----

func.func @pad_and_pack_partially_dynamic(%source: tensor<?x?xf32>, %dest: tensor<?x?x8x2xf32>, %pad: f32) -> tensor<?x?x8x2xf32> {
  %0 = tensor.pack %source padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %dest : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
  return %0 : tensor<?x?x8x2xf32>
}

// CHECK-LABEL: func.func @pad_and_pack_partially_dynamic(
// CHECK-SAME:  %[[SOURCE:.*]]: tensor<?x?xf32>,
// CHECK-SAME:  %[[DEST:.*]]: tensor<?x?x8x2xf32>,
// CHECK-SAME:  %[[PAD:.*]]: f32)
// CHECK: %{{.*}} = tensor.pack %[[SOURCE]] padding_value(%[[PAD]] : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %[[DEST]] : tensor<?x?xf32> -> tensor<?x?x8x2xf32>

// -----

func.func @unpack_fully_dynamic(%source: tensor<?x?x?x?xf32>, %dest: tensor<?x?xf32>, %tile_n : index, %tile_m : index) -> tensor<?x?xf32> {
  %0 = tensor.unpack %source inner_dims_pos = [0, 1] inner_tiles = [%tile_n, %tile_m] into %dest : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @unpack_fully_dynamic(
// CHECK-SAME:  %[[SOURCE:.*]]: tensor<?x?x?x?xf32>,
// CHECK-SAME:  %[[DEST:.*]]: tensor<?x?xf32>,
// CHECK-SAME:  %[[TILE_N:.*]]: index,
// CHECK-SAME:  %[[TILE_M:.*]]: index)
// CHECK: %{{.*}} = tensor.unpack %[[SOURCE]] inner_dims_pos = [0, 1] inner_tiles = [%[[TILE_N]], %[[TILE_M]]] into %[[DEST]] : tensor<?x?x?x?xf32> -> tensor<?x?xf32>

// -----

func.func @unpack_partially_dynamic(%source: tensor<?x?x8x2xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensor.unpack %source inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %dest : tensor<?x?x8x2xf32> -> tensor<?x?xf32>
  return %0: tensor<?x?xf32>
}

// CHECK-LABEL: func.func @unpack_partially_dynamic(
// CHECK-SAME:  %[[SOURCE:.*]]: tensor<?x?x8x2xf32>,
// CHECK-SAME:  %[[DEST:.*]]: tensor<?x?xf32>)
// CHECK: %{{.*}} = tensor.unpack %[[SOURCE]] inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %[[DEST]] : tensor<?x?x8x2xf32> -> tensor<?x?xf32>
