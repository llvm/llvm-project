// RUN: mlir-opt -test-linalg-pad-fusion -split-input-file %s | FileCheck %s

func @dynamic_pad_fusion(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index,
    %arg3 : index, %arg4 : index, %arg5 : f32) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0 : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
    ^bb0(%arg6 : f32, %arg7 : f32):
      %1 = arith.mulf %arg6, %arg6 : f32
      linalg.yield %1 : f32
    } -> tensor<?x?xf32>
  %1 = tensor.pad %0 low [%arg1, %arg2] high [%arg3, %arg4] {
    ^bb0(%arg6: index, %arg7 : index):
      tensor.yield %arg5 : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s2 + s0 + s1)>
//      CHECK: func @dynamic_pad_fusion
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG5:[a-zA-Z0-9]+]]: f32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[SOURCE:.+]] = linalg.generic
//  CHECK-DAG:   %[[SOURCE_D0:.+]] = tensor.dim %[[SOURCE]], %[[C0]]
//  CHECK-DAG:   %[[TARGET_D0:.+]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG3]], %[[SOURCE_D0]]]
//  CHECK-DAG:   %[[SOURCE_D1:.+]] = tensor.dim %[[SOURCE]], %[[C1]]
//  CHECK-DAG:   %[[TARGET_D1:.+]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[ARG4]], %[[SOURCE_D1]]]
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[TARGET_D0]], %[[TARGET_D1]]] 
//      CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[ARG5]]{{.*}}outs(%[[INIT]]
//  CHECK-DAG:   %[[SIZE_D0:.+]] = tensor.dim %[[SOURCE]], %[[C0]]
//  CHECK-DAG:   %[[SIZE_D1:.+]] = tensor.dim %[[SOURCE]], %[[C1]]
//      CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[FILL]]
// CHECK-SAME:       [%[[ARG1]], %[[ARG2]]] [%[[SIZE_D0]], %[[SIZE_D1]]] [1, 1]
//      CHECK:   %[[SOURCE:.+]] = linalg.generic
// CHECK-SAME:       outs(%[[SLICE]] : tensor<?x?xf32>)
//      CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[SOURCE]] into %[[FILL]]
// CHECK-SAME:       [%[[ARG1]], %[[ARG2]]] [%[[SIZE_D0]], %[[SIZE_D1]]] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func @mixed_pad_fusion(%arg0 : tensor<?x42xf32>, %arg1 : index, %arg2 : index,
    %arg3 : f32) -> tensor<49x?xf32> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x42xf32>
  %init = linalg.init_tensor [42, %d0] : tensor<42x?xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>],
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0 : tensor<?x42xf32>) outs(%init : tensor<42x?xf32>) {
    ^bb0(%arg4 : f32, %arg5 : f32):
      %1 = arith.mulf %arg4, %arg4 : f32
      linalg.yield %1 : f32
    } -> tensor<42x?xf32>
  %1 = tensor.pad %0 low [3, %arg1] high [4, %arg2] {
    ^bb0(%arg4: index, %arg5 : index):
      tensor.yield %arg3 : f32
    } : tensor<42x?xf32> to tensor<49x?xf32>
  return %1 : tensor<49x?xf32>
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s2 + s0 + s1)>
//      CHECK: func @mixed_pad_fusion
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x42xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: index
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: f32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[SOURCE:.+]] = linalg.generic
//  CHECK-DAG:   %[[SOURCE_D1:.+]] = tensor.dim %[[SOURCE]], %[[C1]]
//  CHECK-DAG:   %[[TARGET_D1:.+]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]], %[[SOURCE_D1]]]
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [49, %[[TARGET_D1]]] 
//      CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[ARG3]]{{.*}}outs(%[[INIT]]
//  CHECK-DAG:   %[[SIZE_D1:.+]] = tensor.dim %[[SOURCE]], %[[C1]]
//      CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[FILL]]
// CHECK-SAME:       [3, %[[ARG1]]] [42, %[[SIZE_D1]]] [1, 1]
//      CHECK:   %[[SOURCE:.+]] = linalg.generic
// CHECK-SAME:       outs(%[[SLICE]] : tensor<42x?xf32>)
//      CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[SOURCE]] into %[[FILL]]
// CHECK-SAME:       [3, %[[ARG1]]] [42, %[[SIZE_D1]]] [1, 1]
//      CHECK:   return %[[RESULT]]
