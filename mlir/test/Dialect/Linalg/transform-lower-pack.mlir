// RUN: mlir-opt %s -test-transform-dialect-interpreter --split-input-file | FileCheck %s

  // CHECK-LABEL: func.func @pack(
func.func @pack(%arg0: tensor<129x47x16x16xf32>, %arg1: tensor<17x2x16x16x32x8xf32>) -> tensor<17x2x16x16x32x8xf32> {
  %cst_0 = arith.constant 0.0 : f32

  // tensor.pack is lowered to tensor.pad + tensor.expand_shape + linalg.transpose
  //      CHECK: %[[C0:.*]] = arith.constant 0 : index
  //      CHECK: tensor.pad {{.*}} low[%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
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

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.pack">
  transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
    -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
}

// -----

  // CHECK-LABEL: func.func @pack(
func.func @pack(%arg0: tensor<128x8xf32>, %arg1: tensor<8x8x16x1xf32>) -> tensor<8x8x16x1xf32> {

  // tensor.pack is lowered to tensor.pad + tensor.expand_shape + linalg.transpose
  //      CHECK: %[[C0:.*]] = arith.constant 0 : index
  //      CHECK: tensor.pad {{.*}} low[%[[C0]], %[[C0]]]
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

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.pack">
  transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
    -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
}

// -----

// CHECK-LABEL: func.func @pack_as_pad(
func.func @pack_as_pad(%arg0: tensor<129x47x16x16xf32>, %arg1: tensor<1x1x1x1x136x64x16x16xf32>) -> tensor<1x1x1x1x136x64x16x16xf32> {
  %cst_0 = arith.constant 0.0 : f32

  // tensor.pack is lowered to tensor.pad + tensor.insert_slice
  //      CHECK: %[[C0:.*]] = arith.constant 0 : index
  //      CHECK: %[[PAD:.*]] = tensor.pad {{.*}} low[%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
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

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.pack">
  transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
    -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
}

// -----

// Check that we don't lower the following pack as a pad.
// Although all the outer most dimensions in the resulting shape are 1s,
// some of the original dimensions are not part of the inner_dims_pos, hence
// some transpose needs to happen.
// CHECK-LABEL: func.func @pack_not_a_pad(
func.func @pack_not_a_pad(%arg0: tensor<129x47x16x16xf32>, %arg1: tensor<1x1x16x16x136x64xf32>) -> tensor<1x1x16x16x136x64xf32> {
  %cst_0 = arith.constant 0.0 : f32

  //      CHECK: %[[C0:.*]] = arith.constant 0 : index
  //      CHECK: tensor.pad {{.*}} low[%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
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

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.pack">
  transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
    -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
}

// -----
// CHECK-LABEL: func.func @unpack(
func.func @unpack(%arg0: tensor<17x2x16x16x32x8xf32>, %arg1: tensor<129x47x16x16xf32>) -> tensor<129x47x16x16xf32> {
  %cst_0 = arith.constant 0.0 : f32

  //      CHECK: tensor.empty() : tensor<17x8x2x32x16x16xf32>
  //      CHECK: linalg.transpose
  // CHECK-SAME:    ins(%{{.*}} : tensor<17x2x16x16x32x8xf32>)
  // CHECK-SAME:   outs(%{{.*}} : tensor<17x8x2x32x16x16xf32>)
  // CHECK-SAME:   permutation = [0, 5, 1, 4, 2, 3]
  //      CHECK: tensor.collapse_shape {{.*}}[0, 1], [2, 3], [4], [5]]
  // CHECK-SAME:   : tensor<17x8x2x32x16x16xf32> into tensor<136x64x16x16xf32>
  //      CHECK: tensor.extract_slice %{{.*}}[0, 0, 0, 0] [129, 47, 16, 16] [1, 1, 1, 1]
  // CHECK-SAME:   : tensor<136x64x16x16xf32> to tensor<129x47x16x16xf32>
  %pack = tensor.unpack %arg0 inner_dims_pos = [1, 0] inner_tiles = [32, 8] into %arg1
    : tensor<17x2x16x16x32x8xf32> -> tensor<129x47x16x16xf32>
  return %pack : tensor<129x47x16x16xf32>
}

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %unpack = transform.structured.match ops{["tensor.unpack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.unpack">
  transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">)
    -> (!transform.op<"tensor.empty">,
        !transform.op<"linalg.transpose">,
        !transform.op<"tensor.collapse_shape">,
        !transform.op<"tensor.extract_slice">)
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

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %unpack = transform.structured.match ops{["tensor.unpack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.unpack">
  transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">)
    -> (!transform.op<"tensor.empty">,
        !transform.op<"linalg.transpose">,
        !transform.op<"tensor.collapse_shape">,
        !transform.op<"tensor.extract_slice">)
}

// -----

// CHECK-LABEL: func.func @pack_with_outer_dims_perm(
func.func @pack_with_outer_dims_perm(%src: tensor<100x200x128x256xi32>,
                                     %dest: tensor<200x4x16x100x16x32xi32>)
    -> tensor<200x4x16x100x16x32xi32> {
  //      CHECK: %[[C0:.*]] = arith.constant 0 : index
  //      CHECK: tensor.pad {{.*}} low[%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
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

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.pack">
  transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
    -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
}

// -----

// CHECK-LABEL: func.func @pack_with_pad_and_outer_dims_perm(
func.func @pack_with_pad_and_outer_dims_perm(%src: tensor<100x200x127x255xi32>,
                                             %dest: tensor<200x4x16x100x16x32xi32>)
    -> tensor<200x4x16x100x16x32xi32> {
  //      CHECK: %[[C0:.*]] = arith.constant 0 : index
  //      CHECK: tensor.pad {{.*}} low[%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
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

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.pack">
  transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
    -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
}

// -----

// CHECK-LABEL: func.func @pack_as_pad_with_outer_dims_perm(
func.func @pack_as_pad_with_outer_dims_perm(%arg0: tensor<129x47x16x16xf32>, %arg1: tensor<1x1x1x1x136x64x16x16xf32>) -> tensor<1x1x1x1x136x64x16x16xf32> {
  %cst_0 = arith.constant 0.0 : f32

  // tensor.pack is lowered to tensor.pad + tensor.insert_slice
  //      CHECK: %[[C0:.*]] = arith.constant 0 : index
  //      CHECK: %[[PAD:.*]] = tensor.pad {{.*}} low[%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
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

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.pack">
  transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
    -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
}
