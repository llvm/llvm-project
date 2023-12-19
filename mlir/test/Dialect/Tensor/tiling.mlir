// RUN: mlir-opt %s -transform-interpreter -canonicalize -cse -split-input-file | FileCheck %s

//  CHECK-DAG:  #[[MAP0:.*]] = affine_map<()[s0] -> (s0 + 8)>
//  CHECK-DAG:  #[[MAP1:.*]] = affine_map<()[s0] -> (s0 + 7)>
//       CHECK: func @dynamic_pad_tensor_3_4(
//  CHECK-SAME:     %[[IN:.*]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[DIM_IN0:.*]] = tensor.dim %[[IN]], %[[C0]]
//   CHECK-DAG:   %[[DIM_IN1:.*]] = tensor.dim %[[IN]], %[[C1]]
//   CHECK-DAG:   %[[DIM0:.*]] = affine.apply #[[MAP0]]()[%[[DIM_IN0]]]
//   CHECK-DAG:   %[[DIM1:.*]] = affine.apply #[[MAP1]]()[%[[DIM_IN1]]]
//       CHECK:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[DIM0]] step %[[C2]]
//       CHECK:     scf.for {{.*}} = %[[C0]] to %[[DIM1]] step %[[C3]] iter_args(%[[INNER_OUT:.*]] =
//       CHECK:       %[[SWAP_RESULT:.*]] = scf.if
//       CHECK:         tensor.generate
//       CHECK:       else
//       CHECK:         %[[SLICE:.*]] = tensor.extract_slice %[[IN]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       CHECK:         %[[PAD:.*]] = tensor.pad %[[SLICE]]
//       CHECK:       tensor.insert_slice %[[SWAP_RESULT]] into %[[INNER_OUT]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       CHECK:   return %[[RESULT]]

func.func @dynamic_pad_tensor_3_4(%input_tensor: tensor<?x?xf32>,
                         %pad_value: f32) -> tensor<?x?xf32> {
  %0 = tensor.pad %input_tensor low[3, 4] high[5, 3] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 [2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

//   CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 + 7)>
//   CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 + 8)>
//       CHECK: func @dynamic_pad_tensor_0_3(
//  CHECK-SAME:     %[[IN:.*]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[DIM_IN1:.*]] = tensor.dim %[[IN]], %[[C1]]
//   CHECK-DAG:   %[[DIM1:.*]] = affine.apply #[[MAP0]]()[%[[DIM_IN1]]]
//   CHECK-DAG:   %[[DIM_IN0:.*]] = tensor.dim %[[IN]], %[[C0]]
//   CHECK-DAG:   %[[DIM0:.*]] = affine.apply #[[MAP1]]()[%[[DIM_IN0]]]
//       CHECK:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[DIM1]] step %[[C3]] iter_args(%[[INNER_OUT:.*]] =
//       CHECK:     %[[SWAP_RESULT:.*]] = scf.if
//       CHECK:       tensor.generate
//       CHECK:     else
//       CHECK:       %[[SLICE:.*]] = tensor.extract_slice %[[IN]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       CHECK:       %[[PAD:.*]] = tensor.pad %[[SLICE]] low[3, %{{.*}}] high[{{.*}}, {{.*}}]
//       CHECK:     tensor.insert_slice %[[SWAP_RESULT]] into %[[INNER_OUT]][0, {{.*}}] [%[[DIM0]], {{.*}}] [1, 1]
//       CHECK:   return %[[RESULT]]

func.func @dynamic_pad_tensor_0_3(%input_tensor: tensor<?x?xf32>,
                         %pad_value: f32) -> tensor<?x?xf32> {
  %0 = tensor.pad %input_tensor low[3, 4] high[5, 3] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loop = transform.structured.tile_using_for %0 [0, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func @static_pad_tensor_3_4(
//  CHECK-SAME:     %[[IN:.*]]: tensor<7x9xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C15:.*]] = arith.constant 15 : index
//   CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
//       CHECK:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[C15]] step %[[C2]]
//       CHECK:     scf.for {{.*}} = %[[C0]] to %[[C16]] step %[[C3]] iter_args(%[[INNER_OUT:.*]] =
//       CHECK:       %[[SWAP_RESULT:.*]] = scf.if
//       CHECK:         tensor.generate
//       CHECK:       else
//       CHECK:         %[[SLICE:.*]] = tensor.extract_slice %[[IN]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       CHECK:         %[[PAD:.*]] = tensor.pad %[[SLICE]]
//       CHECK:       tensor.insert_slice %[[SWAP_RESULT]] into %[[INNER_OUT]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       CHECK:   return %[[RESULT]]

func.func @static_pad_tensor_3_4(%input_tensor: tensor<7x9xf32>,
                        %pad_value: f32) -> tensor<15x16xf32> {
  %0 = tensor.pad %input_tensor low[3, 4] high[5, 3] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<7x9xf32> to tensor<15x16xf32>
  return %0 : tensor<15x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 [2, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func @static_pad_tensor_0_3(
//  CHECK-SAME:     %[[IN:.*]]: tensor<7x9xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
//       CHECK:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[C16]] step %[[C3]] iter_args(%[[INNER_OUT:.*]] =
//       CHECK:     %[[SWAP_RESULT:.*]] = scf.if
//       CHECK:       tensor.generate
//       CHECK:     else
//       CHECK:       %[[SLICE:.*]] = tensor.extract_slice %[[IN]][0, {{.*}}] [7, {{.*}}] [1, 1]
//       CHECK:       %[[PAD:.*]] = tensor.pad %[[SLICE]] low[3, %{{.*}}] high[5, {{.*}}]
//       CHECK:     %[[CAST_SWAP_RESULT:.*]] = tensor.cast %[[SWAP_RESULT]] : tensor<?x?xf32> to tensor<15x?xf32>
//       CHECK:     tensor.insert_slice %[[CAST_SWAP_RESULT]] into %[[INNER_OUT]][0, {{.*}}] [15, {{.*}}] [1, 1]
//       CHECK:   return %[[RESULT]]

func.func @static_pad_tensor_0_3(%input_tensor: tensor<7x9xf32>,
                        %pad_value: f32) -> tensor<15x16xf32> {
  %0 = tensor.pad %input_tensor low[3, 4] high[5, 3] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<7x9xf32> to tensor<15x16xf32>
  return %0 : tensor<15x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loop = transform.structured.tile_using_for %0 [0, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-LABEL: func @static_pad_tile_evenly_0_3(
//  CHECK-SAME:     %[[IN:.*]]: tensor<7x9xf32>, %[[OUT:.*]]: tensor<14x15xf32>
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C15:.*]] = arith.constant 15 : index
//       CHECK:   %[[RESULT:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C15]] step %[[C3]] iter_args(%[[INNER_OUT:.*]] =
//       CHECK:     %[[R2:.*]] = scf.if
//       CHECK:       %[[GEN:.*]] = tensor.generate
//       CHECK:       %[[cast_0:.*]] = tensor.cast %[[GEN]] : tensor<14x3xf32> to tensor<?x3xf32>
//       CHECK:       scf.yield %[[cast_0]] : tensor<?x3xf32>
//       CHECK:     else
//       CHECK:       %[[SLICE:.*]] = tensor.extract_slice %arg0[0, %{{.*}}] [7, %{{.*}}] [1, 1] : tensor<7x9xf32> to tensor<7x?xf32>
//       CHECK:       %[[PAD:.*]] = tensor.pad %[[SLICE]] low[0, 0] high[7, %{{.*}}]
//       CHECK:       %[[cast_1:.*]] = tensor.cast %[[PAD]] : tensor<14x?xf32> to tensor<?x3xf32>
//       CHECK:       scf.yield %[[cast_1]] : tensor<?x3xf32>
//       CHECK:     %[[cast:.*]] = tensor.cast %[[R2]] : tensor<?x3xf32> to tensor<14x3xf32>
//       CHECK:     %[[R3:.*]] = tensor.insert_slice %[[cast]] into %[[INNER_OUT]][0, %[[IV]]] [14, 3] [1, 1] : tensor<14x3xf32> into tensor<14x15xf32>
//       CHECK:     scf.yield %[[R3]] : tensor<14x15xf32>
//       CHECK:   return %[[RESULT]] : tensor<14x15xf32>

func.func @static_pad_tile_evenly_0_3(%input_tensor: tensor<7x9xf32>,
                             %output_tensor: tensor<14x15xf32>,
                             %pad_value: f32) -> tensor<14x15xf32> {
  %0 = tensor.pad %input_tensor low[0, 0] high[7, 6] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
    } : tensor<7x9xf32> to tensor<14x15xf32>
  return %0 : tensor<14x15xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.pad"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loop = transform.structured.tile_using_for %0 [0, 3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK:       func.func @NC_to_NCnc
// CHECK-SAME:    %[[IN:.*]]: tensor<128x256xf32>,
// CHECK-SAME:    %[[OUT:.*]]: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[RES0:.*]] = scf.for %[[N:.*]] = %[[C0]] to %[[C4]] step %[[C2]] iter_args(%[[ITER0:.*]] = %[[OUT]]) -> (tensor<4x8x32x32xf32>) {
// CHECK:           %[[RES1:.+]] = scf.for %[[C:.*]] = %[[C0]] to %[[C8]] step %[[C4]] iter_args(%[[ITER1:.*]] = %[[ITER0]]) -> (tensor<4x8x32x32xf32>) {
// CHECK-DAG:         %[[IN_N:.+]] = affine.apply #[[MAP0]](%[[N]])
// CHECK-DAG:         %[[IN_C:.+]] = affine.apply #[[MAP0]](%[[C]])
// CHECK:             %[[SUB_IN:.*]] = tensor.extract_slice %[[IN]][%[[IN_N]], %[[IN_C]]] [64, 128] [1, 1] : tensor<128x256xf32> to tensor<64x128xf32>
// CHECK:             %[[SUB_OUT:.*]] = tensor.extract_slice %[[ITER1]][%[[N]], %[[C]], 0, 0] [2, 4, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<2x4x32x32xf32>
// CHECK:             %[[SUB_RES:.*]] = tensor.pack
// CHECK-SAME:          %[[SUB_IN]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[SUB_OUT]]
// CHECK:             %[[INSERT:.*]] = tensor.insert_slice %[[SUB_RES]] into %[[ITER1]]
// CHECK:             scf.yield %[[INSERT]] : tensor<4x8x32x32xf32>
// CHECK:           }
// CHECK:           scf.yield %[[RES1:.*]] : tensor<4x8x32x32xf32>
// CHECK:         }
// CHECK:         return %[[RES0:.*]] : tensor<4x8x32x32xf32>
// CHECK:       }
func.func @NC_to_NCnc(%arg0: tensor<128x256xf32>, %arg1: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  %0 = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1 : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
  return %0 : tensor<4x8x32x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 [2, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK:       #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK:       func.func @KC_to_CKkc
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK:         scf.for %[[C:.+]] = %[[C0]] to %[[C32]] step %[[C2]]
// CHECK-DAG:         %[[IN_C:.+]] = affine.apply #[[MAP0]](%[[C]])
// CHECK:             %[[INPUT_SLICE:.+]] = tensor.extract_slice %[[IN]]
// CHECK-SAME:          [0, %[[IN_C]]] [128, 16]
// CHECK:             %[[OUTPUT_SLICE:.+]] = tensor.extract_slice %{{.+}}[%[[C]], 0, 0, 0] [2, 4, 32, 8]
// CHECK:             tensor.pack
// CHECK-SAME:          %[[INPUT_SLICE]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8]
// CHECK-SAME:          into %[[OUTPUT_SLICE]]
func.func @KC_to_CKkc(%arg0: tensor<128x256xf32>, %arg1: tensor<32x4x32x8xf32>) -> tensor<32x4x32x8xf32> {
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<128x256xf32> -> tensor<32x4x32x8xf32>
  return %0 : tensor<32x4x32x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 [2, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-DAG:     #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-DAG:     #[[MAP1:.+]] = affine_map<(d0) -> (d0 * -2 + 15, 8)>
// CHECK:         func.func @pad_and_pack_static(
// CHECK-SAME:      %[[IN:.*]]: tensor<13x15xf32>,
// CHECK-SAME:      %[[OUT:.*]]: tensor<2x8x8x2xf32>,
// CHECK-SAME:      %[[PAD:.*]]: f32) -> tensor<2x8x8x2xf32> {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[RES0:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[C8]] step %[[C4]] iter_args(%[[ITER1:.*]] = %[[OUT]]) -> (tensor<2x8x8x2xf32>) {
// CHECK-DAG:         %[[IN_J:.*]] = affine.apply #[[MAP0]](%[[J]])
// CHECK-DAG:         %[[IN_J_SZ:.*]] = affine.min #[[MAP1]](%[[J]])
// CHECK:             %[[SUB_IN:.*]] = tensor.extract_slice %[[IN]][0, %[[IN_J]]] [13, %[[IN_J_SZ]]] [1, 1]
// CHECK:             %[[SUB_OUT:.*]] = tensor.extract_slice %[[ITER1]][0, %[[J]], 0, 0] [2, 4, 8, 2] [1, 1, 1, 1]
// CHECK:             %[[SUB_RES:.*]] = tensor.pack
// CHECK-SAME:          %[[SUB_IN]] padding_value(%[[PAD]] : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2]
// CHECK-SAME:          into %[[SUB_OUT]]
// CHECK:             %[[INSERT:.*]] = tensor.insert_slice %[[SUB_RES]] into %[[ITER1]]
// CHECK:             scf.yield %[[INSERT]] : tensor<2x8x8x2xf32>
// CHECK:           }
// CHECK:           return %[[RES0:.*]] : tensor<2x8x8x2xf32>
// CHECK:         }
func.func @pad_and_pack_static(%input: tensor<13x15xf32>, %output: tensor<2x8x8x2xf32>, %pad: f32) -> tensor<2x8x8x2xf32> {
  %0 = tensor.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<13x15xf32> -> tensor<2x8x8x2xf32>
  return %0 : tensor<2x8x8x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 [2, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-DAG:     #[[MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
// CHECK-DAG:     #[[MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
// CHECK-DAG:     #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-DAG:     #[[MAP3:.+]] = affine_map<(d0, d1)[s0] -> (d1 * -8 + s0, d0 * 8)>
// CHECK-DAG:     #[[MAP4:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-DAG:     #[[MAP5:.+]] = affine_map<(d0, d1)[s0] -> (d1 * -2 + s0, d0 * 2)>
// CHECK:         func.func @pad_and_pack_partially_dynamic(
// CHECK-SAME:      %[[IN:.*]]: tensor<?x?xf32>,
// CHECK-SAME:      %[[OUT:.*]]: tensor<?x?x8x2xf32>,
// CHECK-SAME:      %[[PAD:.*]]: f32) -> tensor<?x?x8x2xf32> {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[OUT_D0:.*]] = tensor.dim %[[OUT]], %[[C0]] : tensor<?x?x8x2xf32>
// CHECK-DAG:       %[[OUT_D1:.*]] = tensor.dim %[[OUT]], %[[C1]] : tensor<?x?x8x2xf32>
// CHECK:           %[[RES0:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[OUT_D0]] step %[[C2]] iter_args(%[[ITER0:.*]] = %[[OUT]]) -> (tensor<?x?x8x2xf32>) {
// CHECK-DAG:         %[[OUT_I_SZ:.*]] = affine.min #[[MAP0]](%[[I]])[%[[OUT_D0]]]
// CHECK:             %[[RES1:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[OUT_D1]] step %[[C4]] iter_args(%[[ITER1:.*]] = %[[ITER0]]) -> (tensor<?x?x8x2xf32>) {
// CHECK-DAG:           %[[OUT_J_SZ:.*]] = affine.min #[[MAP1]](%[[J]])[%[[OUT_D1]]]
// CHECK-DAG:           %[[IN_I:.*]] = affine.apply #[[MAP2]](%[[I]])
// CHECK-DAG:           %[[IN_I_SZ:.*]] = affine.min #[[MAP3]]
// CHECK-DAG:           %[[IN_J:.*]] = affine.apply #[[MAP4]](%[[J]])
// CHECK-DAG:           %[[IN_J_SZ:.*]] = affine.min #[[MAP5]]
// CHECK:               %[[SUB_IN:.*]] = tensor.extract_slice %[[IN]][%[[IN_I]], %[[IN_J]]] [%[[IN_I_SZ]], %[[IN_J_SZ]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
// CHECK:               %[[SUB_OUT:.*]] = tensor.extract_slice %[[ITER1]][%[[I]], %[[J]], 0, 0] [%[[OUT_I_SZ]], %[[OUT_J_SZ]], 8, 2] [1, 1, 1, 1] : tensor<?x?x8x2xf32> to tensor<?x?x8x2xf32>
// CHECK:               %[[SUB_RES:.*]] = tensor.pack
// CHECK-SAME:            %[[SUB_IN]] padding_value(%[[PAD]] : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2]
// CHECK-SAME:            into %[[SUB_OUT]]
// CHECK:               %[[INSERT:.*]] = tensor.insert_slice %[[SUB_RES]] into %[[ITER1]]
// CHECK:               scf.yield %[[INSERT]] : tensor<?x?x8x2xf32>
// CHECK:             }
// CHECK:             scf.yield %[[RES1:.*]] : tensor<?x?x8x2xf32>
// CHECK:           }
// CHECK:           return %[[VAL_34:.*]] : tensor<?x?x8x2xf32>
// CHECK:         }
func.func @pad_and_pack_partially_dynamic(%input: tensor<?x?xf32>, %output: tensor<?x?x8x2xf32>, %pad: f32) -> tensor<?x?x8x2xf32> {
  %0 = tensor.pack %input padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %output : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
  return %0 : tensor<?x?x8x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 [2, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-DAG:     #[[MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
// CHECK-DAG:     #[[MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
// CHECK-DAG:     #[[MAP2:.+]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG:     #[[MAP3:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s0, -(d1 * s0) + s1)>
// CHECK:         func.func @pad_and_pack_fully_dynamic(
// CHECK-SAME:      %[[IN:.*]]: tensor<?x?xf32>,
// CHECK-SAME:      %[[OUT:.*]]: tensor<?x?x?x?xf32>,
// CHECK-SAME:      %[[PAD:.*]]: f32,
// CHECK-SAME:      %[[TILE_0:.*]]: index,
// CHECK-SAME:      %[[TILE_1:.*]]: index) -> tensor<?x?x?x?xf32> {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:       %[[OUT_D0:.*]] = tensor.dim %[[OUT]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK-DAG:       %[[OUT_D1:.*]] = tensor.dim %[[OUT]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:           %[[RES0:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[OUT_D0]] step %[[C2]] iter_args(%[[ITER0:.*]] = %[[OUT]]) -> (tensor<?x?x?x?xf32>) {
// CHECK:             %[[OUT_I_SZ:.*]] = affine.min #[[MAP0]](%[[I]])[%[[OUT_D0]]]
// CHECK:             %[[RES1:.*]] = scf.for %[[J:.*]] = %[[C0]] to %[[OUT_D1]] step %[[C4]] iter_args(%[[ITER1:.*]] = %[[ITER0]]) -> (tensor<?x?x?x?xf32>) {
// CHECK:               %[[OUT_J_SZ:.*]] = affine.min #[[MAP1]](%[[J]])[%[[OUT_D1]]]
// CHECK:               %[[IN_D0:.*]] = tensor.dim %[[IN]], %[[C0]]
// CHECK:               %[[IN_D1:.*]] = tensor.dim %[[IN]], %[[C1]]
// CHECK:               %[[IN_I:.*]] = affine.apply #[[MAP2]](%[[I]])[%[[TILE_0]]]
// CHECK:               %[[IN_I_SZ:.*]] = affine.min #[[MAP3]](%[[OUT_I_SZ]], %[[I]])[%[[TILE_0]], %[[IN_D0]]]
// CHECK:               %[[IN_J:.*]] = affine.apply #[[MAP2]](%[[J]])[%[[TILE_1]]]
// CHECK:               %[[IN_J_SZ:.*]] = affine.min #[[MAP3]](%[[OUT_J_SZ]], %[[J]])[%[[TILE_1]], %[[IN_D1]]]
// CHECK:               %[[SUB_IN:.*]] = tensor.extract_slice %[[IN]][%[[IN_I]], %[[IN_J]]] [%[[IN_I_SZ]], %[[IN_J_SZ]]] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
// CHECK:               %[[OUT_D2:.+]] = tensor.dim %[[ITER1]], %[[C2]]
// CHECK:               %[[OUT_D3:.+]] = tensor.dim %[[ITER1]], %[[C3]]
// CHECK:               %[[SUB_OUT:.*]] = tensor.extract_slice %[[ITER1]][%[[I]], %[[J]], 0, 0] [%[[OUT_I_SZ]], %[[OUT_J_SZ]], %[[OUT_D2]], %[[OUT_D3]]] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
// CHECK:               %[[PACK:.*]] = tensor.pack
// CHECK-SAME:            %[[SUB_IN]] padding_value(%[[PAD]] : f32) inner_dims_pos = [0, 1] inner_tiles = [%[[TILE_0]], %[[TILE_1]]]
// CHECK-SAME:            into %[[SUB_OUT]]
// CHECK:               %[[INSERT:.*]] = tensor.insert_slice %[[PACK]] into %[[ITER1]]
// CHECK:               scf.yield %[[INSERT]] : tensor<?x?x?x?xf32>
// CHECK:             }
// CHECK:             scf.yield %[[RES1:.*]] : tensor<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return %[[RES0:.*]] : tensor<?x?x?x?xf32>
// CHECK:         }
func.func @pad_and_pack_fully_dynamic(%source: tensor<?x?xf32>, %dest: tensor<?x?x?x?xf32>, %pad: f32, %tile_n : index, %tile_m : index) -> tensor<?x?x?x?xf32> {
  %0 = tensor.pack %source padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [%tile_n, %tile_m] into %dest : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 [2, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> ((d0 + 1) floordiv 32 - d0 floordiv 32 + 1)>
// CHECK-DAG:   #[[MAP4:.+]] = affine_map<(d0) -> (d0 floordiv 16)>
// CHECK-DAG:   #[[MAP5:.+]] = affine_map<(d0) -> (d0 mod 16)>
// CHECK-DAG:   #[[MAP6:.+]] = affine_map<(d0) -> ((d0 + 3) floordiv 16 - d0 floordiv 16 + 1)>
// CHECK:       func.func @NCnc_to_NC
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG:     %[[C256:.*]] = arith.constant 256 : index
// CHECK:         %{{.+}} = scf.for %[[I:.+]] = %[[C0]] to %[[C256]] step %[[C2]]
// CHECK:           %{{.+}} = scf.for %[[J:.+]] = %[[C0]] to %[[C128]] step %[[C4]]
// CHECK-DAG:         %[[IN_I:.+]] = affine.apply #[[MAP0]](%[[I]])
// CHECK-DAG:         %[[OFFSET_I:.+]] = affine.apply #[[MAP1]](%[[I]])
// CHECK-DAG:         %[[IN_I_SZ:.+]] = affine.apply #[[MAP2]](%[[I]])
// CHECK-DAG:         %[[IN_J:.+]] = affine.apply #[[MAP4]](%[[J]])
// CHECK-DAG:         %[[OFFSET_J:.+]] = affine.apply #[[MAP5]](%[[J]])
// CHECK-DAG:         %[[IN_J_SZ:.+]] = affine.apply #[[MAP6]](%[[J]])
// CHECK:             %[[SLICE:.+]] = tensor.extract_slice %[[IN]]
// CHECK-SAME:          [%[[IN_I]], %[[IN_J]], 0, 0] [%[[IN_I_SZ]], %[[IN_J_SZ]], 32, 16]
// CHECK-SAME:        : tensor<8x8x32x16xf32> to tensor<?x?x32x16xf32>
// CHECK:             %[[EMPTY:.+]] = tensor.empty
// CHECK:             %[[UNPACK:.+]] = tensor.unpack
// CHECK-SAME:          %[[SLICE]] inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:          into %[[EMPTY]]
// CHECK:             %[[UNPACK_SLICE:.+]] = tensor.extract_slice %[[UNPACK]]
// CHECK-SAME:          [%[[OFFSET_I]], %[[OFFSET_J]]] [2, 4]
// CHECK:             %[[RES:.+]] = tensor.insert_slice %[[UNPACK_SLICE]]
// CHECK-SAME:          into %{{.+}}[%[[I]], %[[J]]] [2, 4]
// CHECK:             scf.yield %[[RES]]
func.func @NCnc_to_NC(%source: tensor<8x8x32x16xf32>, %dest: tensor<256x128xf32>) -> tensor<256x128xf32> {
  %0 = tensor.unpack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<8x8x32x16xf32> -> tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 [2, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 32)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> ((d0 + 1) floordiv 32 - d0 floordiv 32 + 1)>
// CHECK-DAG:   #[[MAP4:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-DAG:   #[[MAP5:.+]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK-DAG:   #[[MAP6:.+]] = affine_map<(d0) -> ((d0 + 3) floordiv 8 - d0 floordiv 8 + 1)>
// CHECK:       func.func @CKkc_to_KC
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C128:.*]] = arith.constant 128 : index
// CHECK-DAG:     %[[C256:.*]] = arith.constant 256 : index
// CHECK:         %{{.+}} = scf.for %[[K:.+]] = %[[C0]] to %[[C128]] step %[[C2]]
// CHECK:           %{{.+}} = scf.for %[[C:.+]] = %[[C0]] to %[[C256]] step %[[C4]]
// CHECK-DAG:         %[[IN_K:.+]] = affine.apply #[[MAP0]](%[[K]])
// CHECK-DAG:         %[[OFFSET_K:.+]] = affine.apply #[[MAP1]](%[[K]])
// CHECK-DAG:         %[[IN_K_SZ:.+]] = affine.apply #[[MAP2]](%[[K]])
// CHECK-DAG:         %[[IN_C:.+]] = affine.apply #[[MAP4]](%[[C]])
// CHECK-DAG:         %[[OFFSET_C:.+]] = affine.apply #[[MAP5]](%[[C]])
// CHECK-DAG:         %[[IN_C_SZ:.+]] = affine.apply #[[MAP6]](%[[C]])
// CHECK:             %[[IN_SLICE:.+]] = tensor.extract_slice %[[IN]]
// CHECK:               [%[[IN_C]], %[[IN_K]], 0, 0] [%[[IN_C_SZ]], %[[IN_K_SZ]], 32, 8]
// CHECK:             %[[EMPTY:.+]] = tensor.empty
// CHECK:             %[[UNPACK:.+]] = tensor.unpack
// CHECK-SAME:          %[[IN_SLICE]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8]
// CHECK-SAME:          into %[[EMPTY]]
// CHECK:             %[[UNPACK_SLICE:.+]] = tensor.extract_slice %[[UNPACK]]
// CHECK-SAME:          [%[[OFFSET_K]], %[[OFFSET_C]]] [2, 4]
// CHECK:             %[[RES:.+]] = tensor.insert_slice %[[UNPACK_SLICE]]
// CHECK-SAME:          into %{{.+}}[%[[K]], %[[C]]] [2, 4]
// CHECK:             scf.yield %[[RES]]
func.func @CKkc_to_KC(%source: tensor<32x4x32x8xf32>, %dest: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = tensor.unpack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %dest : tensor<32x4x32x8xf32> -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 [2, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 floordiv 2)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 floordiv 4)>
// CHECK:       func.func @perfect_CKkc_to_KC
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[C128:.*]] = arith.constant 128 : index
// CHECK:         %{{.+}} = scf.for %[[K:.+]] = %[[C0]] to %[[C8]] step %[[C2]]
// CHECK:           %{{.+}} = scf.for %[[C:.+]] = %[[C0]] to %[[C128]] step %[[C4]]
// CHECK-DAG:         %[[IN_K:.+]] = affine.apply #[[MAP0]](%[[K]])
// CHECK-DAG:         %[[IN_C:.+]] = affine.apply #[[MAP1]](%[[C]])
// CHECK:             %[[IN_SLICE:.+]] = tensor.extract_slice %[[IN]]
// CHECK:               [%[[IN_C]], %[[IN_K]], 0, 0] [1, 1, 2, 4]
// CHECK:             %[[ITER_SLICE:.+]] = tensor.extract_slice %{{.+}}[%[[K]], %[[C]]] [2, 4]
// CHECK:             %[[UNPACK:.+]] = tensor.unpack
// CHECK-SAME:          %[[IN_SLICE]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [2, 4]
// CHECK-SAME:          into %[[ITER_SLICE]]
// CHECK:             %[[RES:.+]] = tensor.insert_slice %[[UNPACK]]
// CHECK-SAME:          into %{{.+}}[%[[K]], %[[C]]] [2, 4]
// CHECK:             scf.yield %[[RES]]
func.func @perfect_CKkc_to_KC(%source: tensor<32x4x2x4xf32>, %dest: tensor<8x128xf32>) -> tensor<8x128xf32> {
  %0 = tensor.unpack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [2, 4] into %dest : tensor<32x4x2x4xf32> -> tensor<8x128xf32>
  return %0 : tensor<8x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 [2, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 2)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0) -> (d0 ceildiv 2)>
// CHECK:       func.func @dynamic_perfect_CKkc_to_KC
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[DIM_0:.+]] = tensor.dim %[[OUT]], %[[C0]]
// CHECK-DAG:     %[[DIM_1:.+]] = tensor.dim %[[OUT]], %[[C1]]
// CHECK:         %{{.+}} = scf.for %[[K:.+]] = %[[C0]] to %[[DIM_0]] step %[[C2]]
// CHECK-DAG:       %[[OUT_K_SZ:.+]] = affine.min #[[MAP0]](%[[K]])[%[[DIM_0]]]
// CHECK:           %{{.+}} = scf.for %[[C:.+]] = %[[C0]] to %[[DIM_1]] step %[[C4]]
// CHECK-DAG:         %[[OUT_C_SZ:.+]] = affine.min #[[MAP1]](%[[C]])[%[[DIM_1]]]
// CHECK-DAG:         %[[IN_K:.+]] = affine.apply #[[MAP2]](%[[K]])
// CHECK-DAG:         %[[IN_C:.+]] = affine.apply #[[MAP2]](%[[C]])
// CHECK-DAG:         %[[IN_C_SZ:.+]] = affine.apply #[[MAP3]](%[[OUT_C_SZ]])
// CHECK:             %[[IN_SLICE:.+]] = tensor.extract_slice %[[IN]]
// CHECK:               [%[[IN_C]], %[[IN_K]], 0, 0] [%[[IN_C_SZ]], 1, 2, 2]
// CHECK:             %[[ITER_SLICE:.+]] = tensor.extract_slice %{{.+}}[%[[K]], %[[C]]] [%[[OUT_K_SZ]], %[[OUT_C_SZ]]]
// CHECK:             %[[UNPACK:.+]] = tensor.unpack
// CHECK-SAME:          %[[IN_SLICE]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [2, 2]
// CHECK-SAME:          into %[[ITER_SLICE]]
// CHECK:             %[[RES:.+]] = tensor.insert_slice %[[UNPACK]]
// CHECK-SAME:          into %{{.+}}[%[[K]], %[[C]]] [%[[OUT_K_SZ]], %[[OUT_C_SZ]]]
// CHECK:             scf.yield %[[RES]]

func.func @dynamic_perfect_CKkc_to_KC(%source: tensor<?x?x2x2xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tensor.unpack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %dest : tensor<?x?x2x2xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 [2, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 floordiv 2)>
// CHECK: func.func @perfect_NKPQk_to_NPQK(
// CHECK-SAME:  %[[SOURCE:.+]]: tensor<1x4x6x6x2xf32>,
// CHECK-SAME:  %{{.+}}: tensor<1x6x6x8xf32>)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %{{.+}} = scf.for %[[P:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:   %{{.+}} = scf.for %[[Q:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:     %{{.+}} = scf.for %[[K:.+]] = %[[C0]] to %[[C8]] step %[[C4]]
// CHECK:       %[[K_SZ:.+]] = affine.apply #[[MAP]](%[[K]])
// CHECK:       %[[SLICE_SOURCE:.+]] = tensor.extract_slice %[[SOURCE]][0, %[[K_SZ]], %[[P]], %[[Q]], 0]
// CHECK:       %[[SLICE_DEST:.+]] = tensor.extract_slice %{{.+}}[0, %[[P]], %[[Q]], %[[K]]]
// CHECK:       %[[UNPACK:.+]] = tensor.unpack
// CHECK-SAME:    %[[SLICE_SOURCE]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2]
// CHECK-SAME:    into %[[SLICE_DEST]]
// CHECK:       %[[RES:.+]] = tensor.insert_slice %[[UNPACK]]
// CHECK-SAME:    into %{{.+}}[0, %[[P]], %[[Q]], %[[K]]]
// CHECK:       scf.yield %[[RES]]

func.func @perfect_NKPQk_to_NPQK(%source: tensor<1x4x6x6x2xf32>, %dest: tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32> {
  %0 = tensor.unpack %source outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2] into %dest : tensor<1x4x6x6x2xf32> -> tensor<1x6x6x8xf32>
  return %0 : tensor<1x6x6x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:4 = transform.structured.tile_using_for %0 [1, 1, 1, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

func.func private @get_dynamic_tile_size() -> index

// CHECK-LABEL: func.func @fully_dynamic_unpack
// CHECK-SAME:    %[[SRC:[0-9a-zA-Z]+]]
// CHECK-SAME:    %[[DST:[0-9a-zA-Z]+]]
// CHECK:         %[[INNER_TS:.+]] = call @get_dynamic_tile_size() : () -> index
// CHECK:         %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[DST]])
// CHECK:           %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]])
// CHECK:             %[[SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK:             %[[EMPTY:.+]] = tensor.empty
// CHECK:             %[[UNPACK:.+]] = tensor.unpack %[[SLICE]]
// CHECK-SAME:          inner_dims_pos = [1, 0] inner_tiles = [%[[INNER_TS]], %[[INNER_TS]]] into %[[EMPTY]]
func.func @fully_dynamic_unpack(%source: tensor<?x?x?x?xf32>, %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = func.call @get_dynamic_tile_size() : () -> index
  %1 = tensor.unpack %source inner_dims_pos = [1, 0] inner_tiles = [%0, %0] into %dest : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.unpack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:2 = transform.structured.tile_using_for %0 [4, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}

// -----

// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK: func.func @perfect_NPQK_to_NKPQk
// CHECK-SAME:  %[[SOURCE:.+]]: tensor<1x6x6x8xf32>,
// CHECK-SAME:  %{{.+}}: tensor<1x4x6x6x2xf32>)
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : index
// CHECK: %{{.+}} = scf.for %[[ARG2:.+]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK:   %{{.+}} = scf.for %[[ARG4:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:     %{{.+}} = scf.for %[[ARG6:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:       %[[APPLY:.+]] = affine.apply #[[MAP1]](%[[ARG2]])
// CHECK:       %[[SLICE_SOURCE:.+]] = tensor.extract_slice %[[SOURCE]][0, %[[ARG4]], %[[ARG6]], %[[APPLY]]]
// CHECK:       %[[SLICE_DEST:.+]] = tensor.extract_slice %{{.+}}[0, %[[ARG2]], %[[ARG4]], %[[ARG6]], 0]
// CHECK:       %[[PACK:.+]] = tensor.pack
// CHECK-SAME:    %[[SLICE_SOURCE]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2]
// CHECK-SAME:    into %[[SLICE_DEST]]
// CHECK:       %[[RES:.+]] = tensor.insert_slice %[[PACK]]
// CHECK-SAME:    into %{{.+}}[0, %[[ARG2]], %[[ARG4]], %[[ARG6]], 0]
// CHECK:       scf.yield %[[RES]]

func.func @perfect_NPQK_to_NKPQk(%source: tensor<1x6x6x8xf32>, %dest: tensor<1x4x6x6x2xf32>) -> tensor<1x4x6x6x2xf32> {
  %0 = tensor.pack %source outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2] into %dest : tensor<1x6x6x8xf32> -> tensor<1x4x6x6x2xf32>
  return %0 : tensor<1x4x6x6x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      %1, %loops:4 = transform.structured.tile_using_for %0 [1, 1, 1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield
  }
}
