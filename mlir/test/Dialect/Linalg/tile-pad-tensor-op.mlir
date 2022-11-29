// RUN: mlir-opt %s -test-transform-dialect-interpreter -canonicalize -cse -split-input-file

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

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1
    %1, %loops:2 = transform.structured.tile_to_scf_for %0 [2, 3]
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

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1
    %1, %loop = transform.structured.tile_to_scf_for %0 [0, 3]
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

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1
    %1, %loops:2 = transform.structured.tile_to_scf_for %0 [2, 3]
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

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1
    %1, %loop = transform.structured.tile_to_scf_for %0 [0, 3]
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
//       CHECK:       %[[cast_0:.*]] = tensor.cast %[[GEN]] : tensor<14x3xf32> to tensor<?x?xf32>
//       CHECK:       scf.yield %[[cast_0]] : tensor<?x?xf32>
//       CHECK:     else
//       CHECK:       %[[SLICE:.*]] = tensor.extract_slice %arg0[0, %{{.*}}] [7, %{{.*}}] [1, 1] : tensor<7x9xf32> to tensor<7x?xf32>
//       CHECK:       %[[PAD:.*]] = tensor.pad %[[SLICE]] low[0, 0] high[7, %{{.*}}]
//       CHECK:       %[[cast_1:.*]] = tensor.cast %[[PAD]] : tensor<14x?xf32> to tensor<?x?xf32>
//       CHECK:       scf.yield %[[cast_1]] : tensor<?x?xf32>
//       CHECK:     %[[cast:.*]] = tensor.cast %[[R2]] : tensor<?x?xf32> to tensor<14x3xf32>
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

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1
    %1, %loop = transform.structured.tile_to_scf_for %0 [0, 3]
}
