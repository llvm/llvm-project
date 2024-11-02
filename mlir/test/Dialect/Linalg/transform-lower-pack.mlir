// RUN: mlir-opt %s -transform-interpreter -cse -verify-diagnostics -split-input-file | FileCheck %s

  // CHECK-LABEL: func.func @pack(
func.func @pack(%arg0: tensor<129x47x16x16xf32>, %arg1: tensor<17x2x16x16x32x8xf32>) -> tensor<17x2x16x16x32x8xf32> {
  %cst_0 = arith.constant 0.0 : f32

  // tensor.pack is lowered to tensor.pad + tensor.expand_shape + linalg.transpose
  //      CHECK: tensor.pad {{.*}} low[0, 0, 0, 0]
  //      CHECK:   : tensor<129x47x16x16xf32> to tensor<136x64x16x16xf32>
  //      CHECK: tensor.expand_shape %{{.*}} [{{.*}}[0, 1], [2, 3], [4], [5]]
  // CHECK-SAME:   : tensor<136x64x16x16xf32> into tensor<17x8x2x32x16x16xf32>
  //      CHECK: linalg.transpose
  // CHECK-SAME:   ins(%{{.*}} : tensor<17x8x2x32x16x16xf32>)
  // CHECK-SAME:   outs(%{{.*}} : tensor<17x2x16x16x32x8xf32>)
  // CHECK-SAME:   permutation = [0, 2, 4, 5, 3, 1]
  %pack = tensor.pack %arg0 padding_value(%cst_0 : f32) inner_dims_pos = [1, 0] inner_tiles = [32, 8] into %arg1
    : tensor<129x47x16x16xf32> -> tensor<17x2x16x16x32x8xf32>
  return %pack : tensor<17x2x16x16x32x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.pack">
    transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
      -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
      transform.yield
  }
}

// -----

  // CHECK-LABEL: func.func @pack(
func.func @pack(%arg0: tensor<128x8xf32>, %arg1: tensor<8x8x16x1xf32>) -> tensor<8x8x16x1xf32> {

  // tensor.pack is lowered to tensor.pad + tensor.expand_shape + linalg.transpose
  //      CHECK: tensor.pad {{.*}} low[0, 0]
  //      CHECK:   : tensor<128x8xf32> to tensor<128x8xf32>
  //      CHECK: tensor.expand_shape %{{.*}} [{{.*}}[0, 1], [2, 3]]
  // CHECK-SAME:   : tensor<128x8xf32> into tensor<8x16x8x1xf32>
  //      CHECK: linalg.transpose
  // CHECK-SAME:   ins(%{{.*}} : tensor<8x16x8x1xf32>)
  // CHECK-SAME:   outs(%{{.*}} : tensor<8x8x16x1xf32>)
  // CHECK-SAME:   permutation = [0, 2, 1, 3]

  %pack = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %arg1
    : tensor<128x8xf32> -> tensor<8x8x16x1xf32>

  return %pack : tensor<8x8x16x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.pack">
    transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
      -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @pack_as_pad(
func.func @pack_as_pad(%arg0: tensor<129x47x16x16xf32>, %arg1: tensor<1x1x1x1x136x64x16x16xf32>) -> tensor<1x1x1x1x136x64x16x16xf32> {
  %cst_0 = arith.constant 0.0 : f32

  // tensor.pack is lowered to tensor.pad + tensor.insert_slice
  //      CHECK: %[[PAD:.*]] = tensor.pad {{.*}} low[0, 0, 0, 0]
  //      CHECK:   : tensor<129x47x16x16xf32> to tensor<136x64x16x16xf32>
  //      CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1x1x1x1x136x64x16x16xf32>
  //      CHECK: %[[RES:.*]] = tensor.insert_slice %[[PAD]] into %[[EMPTY]]
  // offsets.
  // CHECK-SAME:   [0, 0, 0, 0, 0, 0, 0, 0]
  // sizes.
  // CHECK-SAME:   [1, 1, 1, 1, 136, 64, 16, 16]
  // strides multipliers.
  // CHECK-SAME:   [1, 1, 1, 1, 1, 1, 1, 1]
  // CHECK-SAME:   : tensor<136x64x16x16xf32> into tensor<1x1x1x1x136x64x16x16xf32>
  //      CHECK: return %[[RES]]
  %pack = tensor.pack %arg0 padding_value(%cst_0 : f32) inner_dims_pos = [0, 1, 2, 3] inner_tiles = [136, 64, 16, 16] into %arg1
    : tensor<129x47x16x16xf32> -> tensor<1x1x1x1x136x64x16x16xf32>
  return %pack :  tensor<1x1x1x1x136x64x16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.pack">
    transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
      -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
      transform.yield
  }
}

// -----

// Check that we don't lower the following pack as a pad.
// Although all the outer most dimensions in the resulting shape are 1s,
// some of the original dimensions are not part of the inner_dims_pos, hence
// some transpose needs to happen.
// CHECK-LABEL: func.func @pack_not_a_pad(
func.func @pack_not_a_pad(%arg0: tensor<129x47x16x16xf32>, %arg1: tensor<1x1x16x16x136x64xf32>) -> tensor<1x1x16x16x136x64xf32> {
  %cst_0 = arith.constant 0.0 : f32

  //      CHECK: tensor.pad {{.*}} low[0, 0, 0, 0]
  //      CHECK:   : tensor<129x47x16x16xf32> to tensor<136x64x16x16xf32>
  //      CHECK: tensor.expand_shape %{{.*}} [{{.*}}[0, 1], [2, 3], [4], [5]]
  // CHECK-SAME:   : tensor<136x64x16x16xf32> into tensor<1x136x1x64x16x16xf32>
  //      CHECK: linalg.transpose
  // CHECK-SAME:   ins(%{{.*}} : tensor<1x136x1x64x16x16xf32>)
  // CHECK-SAME:   outs(%{{.*}} : tensor<1x1x16x16x136x64xf32>)
  // CHECK-SAME:   permutation = [0, 2, 4, 5, 1, 3]

  %pack = tensor.pack %arg0 padding_value(%cst_0 : f32) inner_dims_pos = [0, 1] inner_tiles = [136, 64] into %arg1
    : tensor<129x47x16x16xf32> -> tensor<1x1x16x16x136x64xf32>
  return %pack :  tensor<1x1x16x16x136x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.pack">
    transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
      -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
      transform.yield
  }
}

// -----
// CHECK-LABEL: func.func @unpack(
func.func @unpack(%arg0: tensor<17x2x16x16x32x8xf32>, %arg1: tensor<129x47x16x16xf32>) -> tensor<129x47x16x16xf32> {
  %cst_0 = arith.constant 0.0 : f32
  // CHECK-SAME: %[[ARG0:.*]]: tensor<17x2x16x16x32x8xf32>, %[[ARG1:.*]]: tensor<129x47x16x16xf32>
  //      CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<17x8x2x32x16x16xf32>
  //      CHECK: %[[TRAN:.*]] = linalg.transpose
  // CHECK-SAME:    ins(%[[ARG0]] : tensor<17x2x16x16x32x8xf32>)
  // CHECK-SAME:   outs(%[[EMPTY]] : tensor<17x8x2x32x16x16xf32>)
  // CHECK-SAME:   permutation = [0, 5, 1, 4, 2, 3]
  //      CHECK: %[[CLP:.*]] = tensor.collapse_shape %[[TRAN]] {{\[}}[0, 1], [2, 3], [4], [5]]
  // CHECK-SAME:   : tensor<17x8x2x32x16x16xf32> into tensor<136x64x16x16xf32>
  //      CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[CLP]][0, 0, 0, 0] [129, 47, 16, 16] [1, 1, 1, 1]
  // CHECK-SAME:   : tensor<136x64x16x16xf32> to tensor<129x47x16x16xf32>
  //      CHECK: linalg.copy ins(%[[SLICE]] : tensor<129x47x16x16xf32>)
  // CHECK-SAME:        outs(%[[ARG1]] : tensor<129x47x16x16xf32>)
  %unpack = tensor.unpack %arg0 inner_dims_pos = [1, 0] inner_tiles = [32, 8] into %arg1
    : tensor<17x2x16x16x32x8xf32> -> tensor<129x47x16x16xf32>
  return %unpack : tensor<129x47x16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["tensor.unpack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.unpack">
    transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">)
      -> (!transform.op<"tensor.empty">,
          !transform.op<"linalg.transpose">,
          !transform.op<"tensor.collapse_shape">,
          !transform.op<"tensor.extract_slice">)
          transform.yield
  }
}

// -----

// When an unpack is a plain 'unpad', lower it to a simple extract_slice.
// CHECK-LABEL: func.func @unpack_as_pad(
func.func @unpack_as_pad(%arg0: tensor<1x1x1x1x136x64x16x16xf32>, %arg1: tensor<129x47x16x16xf32>) -> tensor<129x47x16x16xf32> {
  %cst_0 = arith.constant 0.0 : f32

  // CHECK-SAME: %[[ARG0:[^:]*]]: tensor<1x1x1x1x136x64x16x16xf32>
  //      CHECK: %[[RES:.*]] = tensor.extract_slice %[[ARG0]]
  // offsets.
  // CHECK-SAME:   [0, 0, 0, 0, 0, 0, 0, 0]
  // sizes.
  // CHECK-SAME:   [1, 1, 1, 1, 129, 47, 16, 16]
  // strides multiplers.
  // CHECK-SAME:   [1, 1, 1, 1, 1, 1, 1, 1]
  // CHECK-SAME:   : tensor<1x1x1x1x136x64x16x16xf32> to tensor<129x47x16x16xf32>
  %pack = tensor.unpack %arg0 inner_dims_pos = [0, 1, 2, 3] inner_tiles = [136, 64, 16, 16] into %arg1
    : tensor<1x1x1x1x136x64x16x16xf32> -> tensor<129x47x16x16xf32>
  return %pack : tensor<129x47x16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["tensor.unpack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.unpack">
    transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">)
      -> (!transform.op<"tensor.empty">,
          !transform.op<"linalg.transpose">,
          !transform.op<"tensor.collapse_shape">,
          !transform.op<"tensor.extract_slice">)
          transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @pack_with_outer_dims_perm(
func.func @pack_with_outer_dims_perm(%src: tensor<100x200x128x256xi32>,
                                     %dest: tensor<200x4x16x100x16x32xi32>)
    -> tensor<200x4x16x100x16x32xi32> {
  //      CHECK: tensor.pad {{.*}} low[0, 0, 0, 0]
  //      CHECK:   : tensor<100x200x128x256xi32> to tensor<100x200x128x256xi32>
  //      CHECK: tensor.expand_shape %{{.*}} [{{.*}}[0], [1], [2, 3], [4, 5]]
  // CHECK-SAME:   : tensor<100x200x128x256xi32> into tensor<100x200x4x32x16x16xi32>
  //      CHECK: linalg.transpose
  // CHECK-SAME:   ins(%{{.*}} : tensor<100x200x4x32x16x16xi32>)
  // CHECK-SAME:   outs(%{{.*}} : tensor<200x4x16x100x16x32xi32>)
  // CHECK-SAME:   permutation = [1, 2, 4, 0, 5, 3]
  %0 = tensor.pack %src
    outer_dims_perm = [1, 2, 3, 0]
    inner_dims_pos = [3, 2]
    inner_tiles = [16, 32]
    into %dest : tensor<100x200x128x256xi32> -> tensor<200x4x16x100x16x32xi32>
  return %0 : tensor<200x4x16x100x16x32xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.pack">
    transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
      -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @pack_with_pad(
func.func @pack_with_pad(%src: tensor<4225x12xf32>, %dest: tensor<265x16x16x1xf32>)
    -> tensor<265x16x16x1xf32> {
  //      CHECK: tensor.pad {{.*}} low[0, 0]
  //      CHECK:   : tensor<4225x12xf32> to tensor<4240x16xf32>
  //      CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1], [2, 3]]
  // CHECK-SAME:   : tensor<4240x16xf32> into tensor<265x16x16x1xf32>
  //      CHECK: linalg.transpose
  // CHECK-SAME:   ins(%{{[a-zA-Z0-9]*}} : tensor<265x16x16x1xf32>)
  // CHECK-SAME:   outs(%{{[a-zA-Z0-9]*}} : tensor<265x16x16x1xf32>)
  // CHECK-SAME:   permutation = [0, 2, 1, 3]
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.pack %src
    padding_value(%cst : f32)
    inner_dims_pos = [0, 1]
    inner_tiles = [16, 1] into %dest
    : tensor<4225x12xf32> -> tensor<265x16x16x1xf32>
  return %0 : tensor<265x16x16x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.pack">
    transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
      -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @pack_with_pad_and_outer_dims_perm(
func.func @pack_with_pad_and_outer_dims_perm(%src: tensor<100x200x127x255xi32>,
                                             %dest: tensor<200x4x16x100x16x32xi32>)
    -> tensor<200x4x16x100x16x32xi32> {
  //      CHECK: tensor.pad {{.*}} low[0, 0, 0, 0]
  //      CHECK:   : tensor<100x200x127x255xi32> to tensor<100x200x128x256xi32>
  //      CHECK: tensor.expand_shape %{{.*}} [{{.*}}[0], [1], [2, 3], [4, 5]]
  // CHECK-SAME:   : tensor<100x200x128x256xi32> into tensor<100x200x4x32x16x16xi32>
  //      CHECK: linalg.transpose
  // CHECK-SAME:   ins(%{{.*}} : tensor<100x200x4x32x16x16xi32>)
  // CHECK-SAME:   outs(%{{.*}} : tensor<200x4x16x100x16x32xi32>)
  // CHECK-SAME:   permutation = [1, 2, 4, 0, 5, 3]
  %cst_0 = arith.constant 0 : i32
  %0 = tensor.pack %src
    padding_value(%cst_0 : i32)
    outer_dims_perm = [1, 2, 3, 0]
    inner_dims_pos = [3, 2]
    inner_tiles = [16, 32]
    into %dest : tensor<100x200x127x255xi32> -> tensor<200x4x16x100x16x32xi32>
  return %0 : tensor<200x4x16x100x16x32xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.pack">
    transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
      -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
      transform.yield
  }
}

// -----

// CHECK-DAG:   #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 * 16 - s1)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 * 32 - s1)>
// CHECK:       func.func @dynamic_pack_pad_transpose_inner_and_outer_dims(
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
func.func @dynamic_pack_pad_transpose_inner_and_outer_dims(%source: tensor<?x?xf32>) -> tensor<?x?x16x32xf32> {
  // CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
  // CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
  // CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[SRC]], %[[C0]]
  // CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[SRC]], %[[C1]]
  // CHECK-DAG:   %[[OUT_D0:.+]] = arith.ceildivui %[[D1]], %[[C16]] : index
  // CHECK-DAG:   %[[OUT_D1:.+]] = arith.ceildivui %[[D0]], %[[C32]] : index
  // CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[OUT_D0]], %[[OUT_D1]]) : tensor<?x?x16x32xf32>
  // CHECK-DAG:   %[[DEST_D0:.+]] = tensor.dim %[[EMPTY]], %[[C0]]
  // CHECK-DAG:   %[[DEST_D1:.+]] = tensor.dim %[[EMPTY]], %[[C1]]
  // CHECK-DAG:   %[[H1:.+]] = affine.apply #[[MAP0]]()[%[[DEST_D0]], %[[D1]]]
  // CHECK-DAG:   %[[H0:.+]] = affine.apply #[[MAP1]]()[%[[DEST_D1]], %[[D0]]]
  // CHECK:       %[[PAD:.+]] = tensor.pad %[[SRC]] low[0, 0] high[%[[H0]], %[[H1]]]
  // CHECK:         : tensor<?x?xf32> to tensor<?x?xf32>
  // CHECK:       %[[EXPAND:.+]] = tensor.expand_shape %[[PAD]] {{\[}}[0, 1], [2, 3]]
  // CHECK-SAME:   : tensor<?x?xf32> into tensor<?x32x?x16xf32>
  // CHECK:       %[[TRANSP:.+]] = linalg.transpose
  // CHECK-SAME:    ins(%[[EXPAND]] : tensor<?x32x?x16xf32>)
  // CHECK-SAME:    outs(%[[EMPTY]] : tensor<?x?x16x32xf32>)
  // CHECK-SAME:    permutation = [2, 0, 3, 1]
  // CHECK:       return %[[TRANSP]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %source, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %source, %c1 : tensor<?x?xf32>
  %padding_value = arith.constant 0.0 : f32

  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %tiled_d0 = arith.ceildivui %d0, %c32 : index
  %tiled_d1 = arith.ceildivui %d1, %c16 : index
  %init_pack = tensor.empty(%tiled_d1, %tiled_d0) : tensor<?x?x16x32xf32>
  %pack = tensor.pack %source padding_value(%padding_value : f32)
      outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 32] into %init_pack
      : tensor<?x?xf32> -> tensor<?x?x16x32xf32>
  return %pack : tensor<?x?x16x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.pack">
    transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
      -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @pack_as_pad_with_outer_dims_perm(
func.func @pack_as_pad_with_outer_dims_perm(%arg0: tensor<129x47x16x16xf32>, %arg1: tensor<1x1x1x1x136x64x16x16xf32>) -> tensor<1x1x1x1x136x64x16x16xf32> {
  %cst_0 = arith.constant 0.0 : f32

  // tensor.pack is lowered to tensor.pad + tensor.insert_slice
  //      CHECK: %[[PAD:.*]] = tensor.pad {{.*}} low[0, 0, 0, 0]
  //      CHECK:   : tensor<129x47x16x16xf32> to tensor<136x64x16x16xf32>
  //      CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1x1x1x1x136x64x16x16xf32>
  //      CHECK: %[[RES:.*]] = tensor.insert_slice %[[PAD]] into %[[EMPTY]]
  // offsets.
  // CHECK-SAME:   [0, 0, 0, 0, 0, 0, 0, 0]
  // sizes.
  // CHECK-SAME:   [1, 1, 1, 1, 136, 64, 16, 16]
  // strides multipliers.
  // CHECK-SAME:   [1, 1, 1, 1, 1, 1, 1, 1]
  // CHECK-SAME:   : tensor<136x64x16x16xf32> into tensor<1x1x1x1x136x64x16x16xf32>
  //      CHECK: return %[[RES]]
  %pack = tensor.pack %arg0
    padding_value(%cst_0 : f32)
    outer_dims_perm = [1, 2, 3, 0]
    inner_dims_pos = [0, 1, 2, 3]
    inner_tiles = [136, 64, 16, 16]
    into %arg1 : tensor<129x47x16x16xf32> -> tensor<1x1x1x1x136x64x16x16xf32>
  return %pack :  tensor<1x1x1x1x136x64x16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.pack">
    transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
      -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @pack_as_pad_with_unit_dims(
// CHECK: %[[SRC:.+]]: tensor<3x1x1x1xf32>,
// CHECK: %[[OUT:.+]]: tensor<1x1x1x1x8x1xf32>)
func.func @pack_as_pad_with_unit_dims(%arg0: tensor<3x1x1x1xf32>, %arg1: tensor<1x1x1x1x8x1xf32>) -> (tensor<1x1x1x1x8x1xf32>) {
  %zero = arith.constant 0.0 : f32

  // CHECK:      %[[PAD:.+]] = tensor.pad %[[SRC]] low[0, 0, 0, 0] high[5, 0, 0, 0] {
  // CHECK:        : tensor<3x1x1x1xf32> to tensor<8x1x1x1xf32>
  // CHECK:      %[[EXPAND:.+]] = tensor.expand_shape %[[PAD]] [{{.*}}[0, 1], [2, 3], [4], [5]]
  // CHECK-SAME:   tensor<8x1x1x1xf32> into tensor<1x8x1x1x1x1xf32>
  // CHECK:      %[[TRANSPOSED:.+]] = linalg.transpose
  // CHECK-SAME:   ins(%[[EXPAND]] : tensor<1x8x1x1x1x1xf32>)
  // CHECK-SAME:   outs(%[[OUT]] : tensor<1x1x1x1x8x1xf32>)
  // CHECK-SAME:   permutation = [0, 2, 4, 5, 1, 3]
  // CHECK:      return %[[TRANSPOSED]] : tensor<1x1x1x1x8x1xf32>
  %pack = tensor.pack %arg0
      padding_value(%zero : f32)
      inner_dims_pos = [0, 1]
      inner_tiles = [8, 1] into %arg1 : tensor<3x1x1x1xf32> -> tensor<1x1x1x1x8x1xf32>

  return %pack : tensor<1x1x1x1x8x1xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.pack">
    transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
      -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
      transform.yield
  }
}

// -----

// Check that we can lower unpack with dynamic dimensions in the destination.
// CHECK-LABEL: func.func @unpack_with_dynamic_dest(
// CHECK-SAME: %[[ARG0:.*]]: tensor<32x2x49x16x16xf32>, %[[ARG1:.*]]: tensor<32x?x?xf32>)
//      CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32x2x16x49x16xf32>
//      CHECK: %[[TRAN:.*]] = linalg.transpose
// CHECK-SAME:    ins(%[[ARG0]] : tensor<32x2x49x16x16xf32>)
// CHECK-SAME:   outs(%[[EMPTY]] : tensor<32x2x16x49x16xf32>)
// CHECK-SAME:   permutation = [0, 1, 3, 2, 4]
//      CHECK: %[[CLP:.*]] = tensor.collapse_shape %[[TRAN]] {{\[}}[0], [1, 2], [3, 4]]
// CHECK-SAME:   : tensor<32x2x16x49x16xf32> into tensor<32x32x784xf32>
//      CHECK:  %[[C1:.*]] = arith.constant 1 : index
//      CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<32x?x?xf32>
//      CHECK: %[[C2:.*]] = arith.constant 2 : index
//      CHECK: %[[DIM2:.*]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<32x?x?xf32>
//      CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[CLP]][0, 0, 0] [32, %[[DIM1]], %[[DIM2]]] [1, 1, 1]
// CHECK-SAME:   : tensor<32x32x784xf32> to tensor<32x?x?xf32>
//      CHECK: linalg.copy ins(%[[SLICE]] : tensor<32x?x?xf32>)
// CHECK-SAME:        outs(%[[ARG1]] : tensor<32x?x?xf32>)
func.func @unpack_with_dynamic_dest(%arg0: tensor<32x2x49x16x16xf32>, %arg1: tensor<32x?x?xf32>) -> tensor<32x?x?xf32> {
  %pack = tensor.unpack %arg0 inner_dims_pos = [1, 2] inner_tiles = [16, 16] into %arg1
    : tensor<32x2x49x16x16xf32> -> tensor<32x?x?xf32>
  return %pack : tensor<32x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["tensor.unpack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.unpack">
    transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">)
      -> (!transform.op<"tensor.empty">,
          !transform.op<"linalg.transpose">,
          !transform.op<"tensor.collapse_shape">,
          !transform.op<"tensor.extract_slice">)
          transform.yield
  }
}

// -----

// At the moment, we cannot lower tensor.unpack with outer_dims_perm.
func.func @diagnostic_unpack(%arg0: tensor<32x64xf32>, %arg1: tensor<2x4x32x8xf32>) -> tensor<32x64xf32> {
  // expected-note @below {{target payload op}}
  %unpack = tensor.unpack %arg1 outer_dims_perm = [1, 0] 
    inner_dims_pos = [1, 0] inner_tiles = [32, 8] into %arg0 : tensor<2x4x32x8xf32> -> tensor<32x64xf32>
  return %unpack : tensor<32x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %unpack = transform.structured.match ops{["tensor.unpack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.unpack">
    // expected-error @below {{cannot lower to transpose + collapse + extract}} 
    transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">)
      -> (!transform.op<"tensor.empty">,
          !transform.op<"linalg.transpose">,
          !transform.op<"tensor.collapse_shape">,
          !transform.op<"tensor.extract_slice">)
          transform.yield
  }
}
