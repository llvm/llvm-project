// RUN: mlir-opt %s -test-transform-dialect-interpreter -canonicalize -cse -split-input-file | FileCheck %s

// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (d0 * -32 + 128, 64)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 * -32 + 256, 128)>
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
// CHECK-DAG:         %[[IN_N_SZ:.*]] = affine.min #[[MAP1]]
// CHECK-DAG:         %[[IN_C:.+]] = affine.apply #[[MAP0]](%[[C]])
// CHECK-DAG:         %[[IN_C_SZ:.*]] = affine.min #[[MAP2]]
// CHECK:             %[[SUB_IN:.*]] = tensor.extract_slice %[[IN]][%[[IN_N]], %[[IN_C]]] [%[[IN_N_SZ]], %[[IN_C_SZ]]] [1, 1] : tensor<128x256xf32> to tensor<?x?xf32>
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

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg1
    %1, %loops:2 = transform.structured.tile_to_scf_for %0 [2, 4]
}

// -----

// CHECK:       #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK:       #[[MAP1:.+]] = affine_map<(d0) -> (d0 * -8 + 256, 16)>
// CHECK:       func.func @KC_to_CKkc
// CHECK-SAME:    %[[IN:[A-Za-z0-9]+]]:
// CHECK-SAME:    %[[OUT:[A-Za-z0-9]+]]:
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK:         scf.for %[[C:.+]] = %[[C0]] to %[[C32]] step %[[C2]]
// CHECK-DAG:         %[[IN_C:.+]] = affine.apply #[[MAP0]](%[[C]])
// CHECK-DAG:         %[[IN_C_SZ:.+]] = affine.min #[[MAP1]](%[[C]])
// CHECK:             %[[INPUT_SLICE:.+]] = tensor.extract_slice %[[IN]]
// CHECK-SAME:          [0, %[[IN_C]]] [128, %[[IN_C_SZ]]]
// CHECK:             %[[OUTPUT_SLICE:.+]] = tensor.extract_slice %{{.+}}[%[[C]], 0, 0, 0] [2, 4, 32, 8]
// CHECK:             tensor.pack
// CHECK-SAME:          %[[INPUT_SLICE]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8]
// CHECK-SAME:          into %[[OUTPUT_SLICE]]
func.func @KC_to_CKkc(%arg0: tensor<128x256xf32>, %arg1: tensor<32x4x32x8xf32>) -> tensor<32x4x32x8xf32> {
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<128x256xf32> -> tensor<32x4x32x8xf32>
  return %0 : tensor<32x4x32x8xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg1
    %1, %loops:2 = transform.structured.tile_to_scf_for %0 [2, 4]
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

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg1
    %1, %loops:2 = transform.structured.tile_to_scf_for %0 [2, 4]
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

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg1
    %1, %loops:2 = transform.structured.tile_to_scf_for %0 [2, 4]
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
// CHECK:               %[[OUT_D2:.+]] = tensor.dim %[[OUT]], %[[C2]]
// CHECK:               %[[OUT_D3:.+]] = tensor.dim %[[OUT]], %[[C3]]
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

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg1
    %1, %loops:2 = transform.structured.tile_to_scf_for %0 [2, 4]
}
