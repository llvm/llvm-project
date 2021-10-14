// RUN: mlir-opt %s -linalg-tile="tile-sizes=2,3" -cse -split-input-file | \
// RUN: FileCheck %s -check-prefix=TILE2
// RUN: mlir-opt %s -linalg-tile="tile-sizes=0,3" -resolve-shaped-type-result-dims -cse -split-input-file | \
// RUN: FileCheck %s -check-prefix=TILE1

//  TILE2-DAG:  #[[MAP0:.*]] = affine_map<()[s0] -> (s0 + 8)>
//  TILE2-DAG:  #[[MAP1:.*]] = affine_map<()[s0] -> (s0 + 7)>
//       TILE2: func @dynamic_pad_tensor(
//  TILE2-SAME:     %[[IN:.*]]: tensor<?x?xf32>
//   TILE2-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   TILE2-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   TILE2-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   TILE2-DAG:   %[[C3:.*]] = arith.constant 3 : index
//       TILE2:   %[[DIM_IN0:.*]] = tensor.dim %[[IN]], %[[C0]]
//       TILE2:   %[[DIM0:.*]] = affine.apply #[[MAP0]]()[%[[DIM_IN0]]]
//       TILE2:   %[[DIM_IN1:.*]] = tensor.dim %[[IN]], %[[C1]]
//       TILE2:   %[[DIM1:.*]] = affine.apply #[[MAP1]]()[%[[DIM_IN1]]]
//       TILE2:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[DIM0]] step %[[C2]]
//       TILE2:     scf.for {{.*}} = %[[C0]] to %[[DIM1]] step %[[C3]] iter_args(%[[INNER_OUT:.*]] =
//       TILE2:       %[[SWAP_RESULT:.*]] = scf.if
//       TILE2:         tensor.generate
//       TILE2:       else
//       TILE2:         %[[SLICE:.*]] = tensor.extract_slice %[[IN]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       TILE2:         %[[PAD:.*]] = linalg.pad_tensor %[[SLICE]]
//       TILE2:       tensor.insert_slice %[[SWAP_RESULT]] into %[[INNER_OUT]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       TILE2:   return %[[RESULT]]

//   TILE1-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 + 7)>
//   TILE1-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 + 8)>
//       TILE1: func @dynamic_pad_tensor(
//  TILE1-SAME:     %[[IN:.*]]: tensor<?x?xf32>
//   TILE1-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   TILE1-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   TILE1-DAG:   %[[C3:.*]] = arith.constant 3 : index
//       TILE1:   %[[DIM_IN1:.*]] = tensor.dim %[[IN]], %[[C1]]
//       TILE1:   %[[DIM1:.*]] = affine.apply #[[MAP0]]()[%[[DIM_IN1]]]
//       TILE1:   %[[DIM_IN0:.*]] = tensor.dim %[[IN]], %[[C0]]
//       TILE1:   %[[DIM0:.*]] = affine.apply #[[MAP1]]()[%[[DIM_IN0]]]
//       TILE1:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[DIM1]] step %[[C3]] iter_args(%[[INNER_OUT:.*]] =
//       TILE1:     %[[SWAP_RESULT:.*]] = scf.if
//       TILE1:       tensor.generate
//       TILE1:     else
//       TILE1:       %[[SLICE:.*]] = tensor.extract_slice %[[IN]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       TILE1:       %[[PAD:.*]] = linalg.pad_tensor %[[SLICE]] low[3, %{{.*}}] high[{{.*}}, {{.*}}]
//       TILE1:     tensor.insert_slice %[[SWAP_RESULT]] into %[[INNER_OUT]][0, {{.*}}] [%[[DIM0]], {{.*}}] [1, 1]
//       TILE1:   return %[[RESULT]]

func @dynamic_pad_tensor(%input_tensor: tensor<?x?xf32>,
                         %pad_value: f32) -> tensor<?x?xf32> {
  %0 = linalg.pad_tensor %input_tensor low[3, 4] high[5, 3] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad_value : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// TILE2-LABEL: func @static_pad_tensor(
//  TILE2-SAME:     %[[IN:.*]]: tensor<7x9xf32>
//   TILE2-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   TILE2-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   TILE2-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   TILE2-DAG:   %[[C15:.*]] = arith.constant 15 : index
//   TILE2-DAG:   %[[C16:.*]] = arith.constant 16 : index
//       TILE2:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[C15]] step %[[C2]]
//       TILE2:     scf.for {{.*}} = %[[C0]] to %[[C16]] step %[[C3]] iter_args(%[[INNER_OUT:.*]] =
//       TILE2:       %[[SWAP_RESULT:.*]] = scf.if
//       TILE2:         tensor.generate
//       TILE2:       else
//       TILE2:         %[[SLICE:.*]] = tensor.extract_slice %[[IN]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       TILE2:         %[[PAD:.*]] = linalg.pad_tensor %[[SLICE]]
//       TILE2:       tensor.insert_slice %[[SWAP_RESULT]] into %[[INNER_OUT]][{{.*}}, {{.*}}] [{{.*}}, {{.*}}] [1, 1]
//       TILE2:   return %[[RESULT]]


// TILE1-LABEL: func @static_pad_tensor(
//  TILE1-SAME:     %[[IN:.*]]: tensor<7x9xf32>
//   TILE1-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   TILE1-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   TILE1-DAG:   %[[C16:.*]] = arith.constant 16 : index
//       TILE1:   %[[RESULT:.*]] = scf.for {{.*}} = %[[C0]] to %[[C16]] step %[[C3]] iter_args(%[[INNER_OUT:.*]] =
//       TILE1:     %[[SWAP_RESULT:.*]] = scf.if
//       TILE1:       tensor.generate
//       TILE1:     else
//       TILE1:       %[[SLICE:.*]] = tensor.extract_slice %[[IN]][0, {{.*}}] [7, {{.*}}] [1, 1]
//       TILE1:       %[[PAD:.*]] = linalg.pad_tensor %[[SLICE]] low[3, %{{.*}}] high[5, {{.*}}]
//       TILE1:     tensor.insert_slice %[[SWAP_RESULT]] into %[[INNER_OUT]][0, {{.*}}] [15, {{.*}}] [1, 1]
//       TILE1:   return %[[RESULT]]

func @static_pad_tensor(%input_tensor: tensor<7x9xf32>,
                        %pad_value: f32) -> tensor<15x16xf32> {
  %0 = linalg.pad_tensor %input_tensor low[3, 4] high[5, 3] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad_value : f32
    } : tensor<7x9xf32> to tensor<15x16xf32>
  return %0 : tensor<15x16xf32>
}

// -----

// TILE1-LABEL: func @static_pad_tile_evenly(
//  TILE1-SAME:     %[[IN:.*]]: tensor<7x9xf32>, %[[OUT:.*]]: tensor<14x15xf32>
//   TILE1-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   TILE1-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   TILE1-DAG:   %[[C15:.*]] = arith.constant 15 : index
//       TILE1:   %[[RESULT:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C15]] step %[[C3]] iter_args(%[[INNER_OUT:.*]] =
//       TILE1:     %[[R2:.*]] = scf.if
//       TILE1:       %[[GEN:.*]] = tensor.generate
//       TILE1:       scf.yield %[[GEN]] : tensor<14x3xf32>
//       TILE1:     else
//       TILE1:       %[[SLICE:.*]] = tensor.extract_slice %arg0[0, %{{.*}}] [7, %{{.*}}] [1, 1] : tensor<7x9xf32> to tensor<7x?xf32>
//       TILE1:       %[[PAD:.*]] = linalg.pad_tensor %[[SLICE]] low[0, 0] high[7, %{{.*}}]
//       TILE1:       scf.yield %[[PAD]] : tensor<14x3xf32>
//       TILE1:     %[[R3:.*]] = tensor.insert_slice %[[R2]] into %[[INNER_OUT]][0, %[[IV]]] [14, 3] [1, 1] : tensor<14x3xf32> into tensor<14x15xf32>
//       TILE1:     scf.yield %[[R3]] : tensor<14x15xf32>
//       TILE1:   return %[[RESULT]] : tensor<14x15xf32>
func @static_pad_tile_evenly(%input_tensor: tensor<7x9xf32>,
                             %output_tensor: tensor<14x15xf32>,
                             %pad_value: f32) -> tensor<14x15xf32> {
  %0 = linalg.pad_tensor %input_tensor low[0, 0] high[7, 6] {
    ^bb0(%arg1: index, %arg2: index):
      linalg.yield %pad_value : f32
    } : tensor<7x9xf32> to tensor<14x15xf32>
  return %0 : tensor<14x15xf32>
}
