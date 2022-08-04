// RUN: mlir-opt %s -test-linalg-greedy-fusion -split-input-file | FileCheck %s

func.func @matmul_tensors(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %t0 = linalg.matmul ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>

  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %t0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %t0, %c1 : tensor<?x?xf32>
  %2 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %3 = scf.for %arg3 = %c0 to %0 step %c2 iter_args(%arg4 = %arg2) -> (tensor<?x?xf32>) {
    %4 = scf.for %arg5 = %c0 to %2 step %c3 iter_args(%arg6 = %arg4) -> (tensor<?x?xf32>) {
      %5 = scf.for %arg7 = %c0 to %1 step %c4 iter_args(%arg8 = %arg6) -> (tensor<?x?xf32>) {
        %6 = tensor.extract_slice %t0[%arg3, %arg7][%c2, 4][1, 1] : tensor<?x?xf32> to tensor<?x4xf32>
        %7 = tensor.extract_slice %arg1[%arg7, %arg5][4, %c3][1, 1] : tensor<?x?xf32> to tensor<4x?xf32>
        %8 = tensor.extract_slice %arg8[%arg3, %arg5][%c2, %c3][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %9 = linalg.matmul ins(%6, %7 : tensor<?x4xf32>, tensor<4x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %10 = tensor.insert_slice %9 into %arg8[%arg3, %arg5] [%c2, %c3] [1, 1]  : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %10 : tensor<?x?xf32>
      }
      scf.yield %5 : tensor<?x?xf32>
    }
    scf.yield %4 : tensor<?x?xf32>
  }
  return %3 : tensor<?x?xf32>
}

//       CHECK: func @matmul_tensors(
//  CHECK-SAME: %[[A:[0-9a-z]*]]: tensor<?x?xf32>
//  CHECK-SAME: %[[B:[0-9a-z]*]]: tensor<?x?xf32>
//  CHECK-SAME: %[[C:[0-9a-z]*]]: tensor<?x?xf32>

//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[dA1:.*]] = tensor.dim %[[A]], %[[C1]] : tensor<?x?xf32>
//   CHECK-DAG: %[[dB0:.*]] = tensor.dim %[[B]], %[[C0]] : tensor<?x?xf32>
//   CHECK-DAG: %[[dB1:.*]] = tensor.dim %[[B]], %[[C1]] : tensor<?x?xf32>
//       CHECK: scf.for %[[I:[0-9a-z]*]]
//       CHECK:   %[[stA:.*]] = tensor.extract_slice %[[A]][%[[I]], 0] [2, %[[dA1]]] [1, 1]  : tensor<?x?xf32> to tensor<2x?xf32>
//       CHECK:   scf.for %[[J:[0-9a-z]*]]
//  CHECK-NEXT:     scf.for %[[K:[0-9a-z]*]] {{.*}} iter_args(%[[RES:[0-9a-z]*]]
//   CHECK-DAG:       %[[stB1:.*]] = tensor.extract_slice %[[B]][%[[K]], %[[J]]] [4, 3] [1, 1]  : tensor<?x?xf32> to tensor<4x3xf32>
//   CHECK-DAG:       %[[stF:.*]] = tensor.extract_slice %[[RES]][%[[I]], %[[J]]] [2, 3] [1, 1]  : tensor<?x?xf32> to tensor<2x3xf32>
//
// slices of the producing matmul.
//   CHECK-DAG:       %[[stB2:.*]] = tensor.extract_slice %[[B]][0, %[[K]]] [%[[dB0]], 4] [1, 1]  : tensor<?x?xf32> to tensor<?x4xf32>
//   CHECK-DAG:       %[[stC:.*]] = tensor.extract_slice %[[C]][%[[I]], %[[K]]] [2, 4] [1, 1]  : tensor<?x?xf32> to tensor<2x4xf32>
//       CHECK:       %[[stD:.*]] = linalg.matmul ins(%[[stA]], %[[stB2]] : tensor<2x?xf32>, tensor<?x4xf32>) outs(%[[stC]] : tensor<2x4xf32>)  -> tensor<2x4xf32>
//  CHECK-NEXT:       %[[stG:.*]] = linalg.matmul ins(%[[stD]], %[[stB1]] : tensor<2x4xf32>, tensor<4x3xf32>) outs(%[[stF]] : tensor<2x3xf32>)  -> tensor<2x3xf32>
//  CHECK-NEXT:       tensor.insert_slice %[[stG]] into %[[RES]][%[[I]], %[[J]]]

// -----

func.func @conv_tensors_static(%input: tensor<1x225x225x3xf32>, %filter: tensor<3x3x3x32xf32>, %elementwise: tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32> {
  %c112 = arith.constant 112 : index
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

  %init = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>

  %conv = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%input, %filter : tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>)
    outs(%fill : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>

  %for0 = scf.for %iv0 = %c0 to %c112 step %c8 iter_args(%arg0 = %fill) -> tensor<1x112x112x32xf32> {
    %for1 = scf.for %iv1 = %c0 to %c112 step %c16 iter_args(%arg1 = %arg0) -> tensor<1x112x112x32xf32> {
      %for2 = scf.for %iv2 = %c0 to %c32 step %c4 iter_args(%arg2 = %arg1) -> tensor<1x112x112x32xf32> {
        %0 = tensor.extract_slice %conv[0, %iv0, %iv1, %iv2][1, 8, 16, 4][1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x8x16x4xf32>
        %1 = tensor.extract_slice %elementwise[0, %iv0, %iv1, %iv2][1, 8, 16, 4][1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x8x16x4xf32>
        %2 = tensor.extract_slice %arg2[0, %iv0, %iv1, %iv2][1, 8, 16, 4][1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x8x16x4xf32>
        %add = linalg.generic
          {
            indexing_maps = [
              affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
              affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
              affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
            iterator_types = ["parallel", "parallel", "parallel", "parallel"]
          }
          ins(%0, %1 : tensor<1x8x16x4xf32>, tensor<1x8x16x4xf32>) outs(%2 : tensor<1x8x16x4xf32>) {
        ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
          %result = arith.addf %arg3, %arg4 : f32
          linalg.yield %result : f32
        } -> tensor<1x8x16x4xf32>

        %insert = tensor.insert_slice %add into %arg2[0, %iv0, %iv1, %iv2] [1, 8, 16, 4] [1, 1, 1, 1]  : tensor<1x8x16x4xf32> into tensor<1x112x112x32xf32>
        scf.yield %insert : tensor<1x112x112x32xf32>
      }
      scf.yield %for2 : tensor<1x112x112x32xf32>
    }
    scf.yield %for1 : tensor<1x112x112x32xf32>
  }
  return %for0 : tensor<1x112x112x32xf32>
}

//      CHECK: #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 2)>
//      CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//      CHECK: func @conv_tensors_static
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x225x225x3xf32>, %[[FILTER:.+]]: tensor<3x3x3x32xf32>, %[[ELEM:.+]]: tensor<1x112x112x32xf32>)

//      CHECK: %[[INIT:.+]] = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
// CHECK-NEXT: %[[FILL:.+]] = linalg.fill ins(%cst : f32) outs(%[[INIT]] : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>

// CHECK-NEXT: scf.for %[[IV0:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ARG0:.+]] = %[[FILL]])
// CHECK-NEXT:   %[[OFFSET_H:.+]] = affine.apply #[[MAP0]](%[[IV0]])
// CHECK-NEXT:   scf.for %[[IV1:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ARG1:.+]] = %[[ARG0]])
// CHECK-NEXT:     %[[OFFSET_W:.+]] = affine.apply #[[MAP0]](%[[IV1]])
// CHECK-NEXT:     %[[ST_INPUT:.+]] = tensor.extract_slice %arg0[0, %[[OFFSET_H]], %[[OFFSET_W]], 0] [1, 17, 33, 3] [1, 1, 1, 1] : tensor<1x225x225x3xf32> to tensor<1x17x33x3xf32>
// CHECK-NEXT:     scf.for %[[IV2:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ARG2:.+]] = %[[ARG1]])
// CHECK-NEXT:       %[[ST_ELEM:.+]] = tensor.extract_slice %[[ELEM]][0, %[[IV0]], %[[IV1]], %[[IV2]]] [1, 8, 16, 4] [1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x8x16x4xf32>
// CHECK-NEXT:       %[[ST_ARG2:.+]] = tensor.extract_slice %[[ARG2]][0, %[[IV0]], %[[IV1]], %[[IV2]]] [1, 8, 16, 4] [1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x8x16x4xf32>
// CHECK-NEXT:       %[[ST_FILTER:.+]] = tensor.extract_slice %[[FILTER]][0, 0, 0, %[[IV2]]] [3, 3, 3, 4] [1, 1, 1, 1] : tensor<3x3x3x32xf32> to tensor<3x3x3x4xf32>
// CHECK-NEXT:       %[[ST_FILL:.+]] = tensor.extract_slice %[[FILL]][0, %[[IV0]], %[[IV1]], %[[IV2]]] [1, 8, 16, 4] [1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x8x16x4xf32>
// CHECK-NEXT:       %[[ST_CONV:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:         ins(%[[ST_INPUT]], %[[ST_FILTER]] : tensor<1x17x33x3xf32>, tensor<3x3x3x4xf32>)
// CHECK-SAME:         outs(%[[ST_FILL]] : tensor<1x8x16x4xf32>)
// CHECK-NEXT:       %[[ADD:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[ST_CONV]], %[[ST_ELEM]] : tensor<1x8x16x4xf32>, tensor<1x8x16x4xf32>)
// CHECK-SAME:         outs(%[[ST_ARG2]] : tensor<1x8x16x4xf32>)
//      CHECK:       tensor.insert_slice %[[ADD]] into %[[ARG2]][0, %[[IV0]], %[[IV1]], %[[IV2]]] [1, 8, 16, 4]

// -----

func.func @conv_tensors_dynamic(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?x?xf32>, %elementwise: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  %n = tensor.dim %elementwise, %c0 : tensor<?x?x?x?xf32>
  %oh = tensor.dim %elementwise, %c1 : tensor<?x?x?x?xf32>
  %ow = tensor.dim %elementwise, %c2 : tensor<?x?x?x?xf32>
  %oc = tensor.dim %elementwise, %c3 : tensor<?x?x?x?xf32>

  %init = linalg.init_tensor [%n, %oh, %ow, %oc] : tensor<?x?x?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

  %conv = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%input, %filter : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs(%fill : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

  %for0 = scf.for %iv0 = %c0 to %n step %c8 iter_args(%arg0 = %fill) -> tensor<?x?x?x?xf32> {
    %for1 = scf.for %iv1 = %c0 to %oh step %c16 iter_args(%arg1 = %arg0) -> tensor<?x?x?x?xf32> {
      %for2 = scf.for %iv2 = %c0 to %ow step %c4 iter_args(%arg2 = %arg1) -> tensor<?x?x?x?xf32> {
        %for3 = scf.for %iv3 = %c0 to %oc step %c2 iter_args(%arg3 = %arg2) -> tensor<?x?x?x?xf32> {
          %n_size = affine.min affine_map<(d0)[s0] -> (8, -d0 + s0)>(%iv0)[%n]
          %oh_size = affine.min affine_map<(d0)[s0] -> (16, -d0 + s0)>(%iv1)[%oh]
          %ow_size = affine.min affine_map<(d0)[s0] -> (4, -d0 + s0)>(%iv2)[%ow]
          %oc_size = affine.min affine_map<(d0)[s0] -> (2, -d0 + s0)>(%iv2)[%oc]
          %0 = tensor.extract_slice %conv[%iv0, %iv1, %iv2, %iv3][%n_size, %oh_size, %ow_size, %oc_size][1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
          %1 = tensor.extract_slice %elementwise[%iv0, %iv1, %iv2, %iv3][%n_size, %oh_size, %ow_size, %oc_size][1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
          %2 = tensor.extract_slice %arg3[%iv0, %iv1, %iv2, %iv3][%n_size, %oh_size, %ow_size, %oc_size][1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
          %add = linalg.generic
            {
              indexing_maps = [
                affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
              iterator_types = ["parallel", "parallel", "parallel", "parallel"]
            }
            ins(%0, %1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%2 : tensor<?x?x?x?xf32>) {
          ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
            %result = arith.addf %arg4, %arg5 : f32
            linalg.yield %result : f32
          } -> tensor<?x?x?x?xf32>

          %insert = tensor.insert_slice %add into %arg3[%iv0, %iv1, %iv2, %iv3] [%n_size, %oh_size, %ow_size, %oc_size] [1, 1, 1, 1]  : tensor<?x?x?x?xf32> into tensor<?x?x?x?xf32>
          scf.yield %insert : tensor<?x?x?x?xf32>
        }
        scf.yield %for3 : tensor<?x?x?x?xf32>
      }
      scf.yield %for2 : tensor<?x?x?x?xf32>
    }
    scf.yield %for1 : tensor<?x?x?x?xf32>
  }
  return %for0 : tensor<?x?x?x?xf32>
}

// CHECK: #[[BOUND8_MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 8)>
// CHECK: #[[BOUND8_MAP_2:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s1, -d0 + s0, 8)>
// CHECK: #[[BOUND16_MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
// CHECK: #[[X2_MAP:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK: #[[INPUT_BOUND:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * -2 + s0 * 2 + s1 - 2, d1 * 2 + s1 - 2)>
// CHECK: #[[BOUND16_MAP_2:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s1, -d0 + s0, 16)>
// CHECK: #[[BOUND4_MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
// CHECK: #[[BOUND2_MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 2)>
// CHECK: #[[BOUND4_MAP_2:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s1, -d0 + s0, 4)>
// CHECK: #[[BOUND2_MAP_2:.+]] = affine_map<(d0, d1)[s0, s1] -> (-d0 + s0, -d1 + s1, 2)>

//      CHECK: func @conv_tensors_dynamic
// CHECK-SAME: (%[[INPUT]]: tensor<?x?x?x?xf32>, %[[FILTER]]: tensor<?x?x?x?xf32>, %[[ELEM]]: tensor<?x?x?x?xf32>)

//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index

//  CHECK-DAG:   %[[ELEM_N:.+]] = tensor.dim %[[ELEM]], %[[C0]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[ELEM_OH:.+]] = tensor.dim %[[ELEM]], %[[C1]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[ELEM_OW:.+]] = tensor.dim %[[ELEM]], %[[C2]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[ELEM_OC:.+]] = tensor.dim %[[ELEM]], %[[C3]] : tensor<?x?x?x?xf32>

//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[ELEM_N]], %[[ELEM_OH]], %[[ELEM_OW]], %[[ELEM_OC]]] : tensor<?x?x?x?xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill ins(%cst : f32) outs(%[[INIT]] : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

//  CHECK-DAG:   %[[FILTER_H:.+]] = tensor.dim %[[FILTER]], %[[C0]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[FILTER_W:.+]] = tensor.dim %[[FILTER]], %[[C1]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[FILTER_IC:.+]] = tensor.dim %[[FILTER]], %[[C2]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[FILTER_OC:.+]] = tensor.dim %[[FILTER]], %[[C3]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[INPUT_N:.+]] = tensor.dim %[[INPUT]], %[[C0]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[INPUT_C:.+]] = tensor.dim %[[INPUT]], %[[C3]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[FILL_H:.+]] = tensor.dim %[[FILL]], %[[C1]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[FILL_W:.+]] = tensor.dim %[[FILL]], %[[C2]] : tensor<?x?x?x?xf32>

//      CHECK:   scf.for %[[IV0:.+]] = %{{.+}} to %[[ELEM_N]] step %{{.+}} iter_args(%{{.+}} = %[[FILL]])
// CHECK-NEXT:     %[[SIZE_ELEM_N:.+]] = affine.min #[[BOUND8_MAP]](%[[IV0]])[%[[ELEM_N]]]
// CHECK-NEXT:     %[[SIZE_INPUT_N:.+]] = affine.min #[[BOUND8_MAP_2]](%[[IV0]])[%[[INPUT_N]], %[[ELEM_N]]]
// CHECK-NEXT:     scf.for %[[IV1:.+]] = %{{.+}} to %[[ELEM_OH]]
// CHECK-NEXT:       %[[SIZE_ELEM_OH:.+]] = affine.min #[[BOUND16_MAP]](%[[IV1]])[%[[ELEM_OH]]]
// CHECK-NEXT:       %[[OFFSET_OH:.+]] = affine.apply #[[X2_MAP]](%[[IV1]])
// CHECK-NEXT:       %[[SIZE_INPUT_H:.+]] = affine.min #[[INPUT_BOUND]](%[[IV1]], %[[SIZE_ELEM_OH]])[%[[FILL_H]], %[[FILTER_H]]]
// CHECK-NEXT:       %[[SIZE_ELEM_OH_2:.+]] = affine.min #[[BOUND16_MAP_2]](%[[IV1]])[%[[FILL_H]], %[[ELEM_OH]]]
// CHECK-NEXT:       scf.for %[[IV2:.+]] = %{{.+}} to %[[ELEM_OW]]
// CHECK-NEXT:         %[[SIZE_ELEM_OW:.+]] = affine.min #[[BOUND4_MAP]](%[[IV2]])[%[[ELEM_OW]]]
// CHECK-NEXT:         %[[SIZE_ELEM_OC:.+]] = affine.min #[[BOUND2_MAP]](%[[IV2]])[%[[ELEM_OC]]]
// CHECK-NEXT:         %[[OFFSET_OW:.+]] = affine.apply #[[X2_MAP]](%[[IV2]])
// CHECK-NEXT:         %[[SIZE_INPUT_W:.+]] = affine.min #[[INPUT_BOUND]](%[[IV2]], %[[SIZE_ELEM_OW]])[%[[FILL_W]], %[[FILTER_W]]]
// CHECK-NEXT:         %[[SIZE_ELEM_OW_2:.+]] = affine.min #[[BOUND4_MAP_2]](%[[IV2]])[%[[FILL_W]], %[[ELEM_OW]]]
// CHECK-NEXT:         %[[ST_INPUT:.+]] = tensor.extract_slice %[[INPUT]][%[[IV0]], %[[OFFSET_OH]], %[[OFFSET_OW]], 0]
// CHECK-SAME:               [%[[SIZE_INPUT_N]], %[[SIZE_INPUT_H]], %[[SIZE_INPUT_W]], %[[INPUT_C]]]
// CHECK-NEXT:         scf.for %[[IV3:.+]] = %{{.+}} to %[[ELEM_OC]] step %{{.+}} iter_args(%[[ARG:[a-z0-9]+]]
// CHECK-NEXT:           %[[ST_ELEM:.+]] = tensor.extract_slice %[[ELEM]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
// CHECK-SAME:                 [%[[SIZE_ELEM_N]], %[[SIZE_ELEM_OH]], %[[SIZE_ELEM_OW]], %[[SIZE_ELEM_OC]]]
// CHECK-NEXT:           %[[ST_ARG:.+]] = tensor.extract_slice %[[ARG]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
// CHECK-SAME:                 [%[[SIZE_ELEM_N]], %[[SIZE_ELEM_OH]], %[[SIZE_ELEM_OW]], %[[SIZE_ELEM_OC]]]
// CHECK-NEXT:           %[[SIZE_ELEM_OC_2:.+]] = affine.min #[[BOUND2_MAP_2]](%[[IV3]], %[[IV2]])[%[[FILTER_OC]], %[[ELEM_OC]]]
// CHECK-NEXT:           %[[ST_FILTER:.+]] = tensor.extract_slice %[[FILTER]][0, 0, 0, %[[IV3]]]
// CHECK-SAME:                 [%[[FILTER_H]], %[[FILTER_W]], %[[FILTER_IC]], %[[SIZE_ELEM_OC_2]]]
// CHECK-NEXT:           %[[ST_FILL:.+]] = tensor.extract_slice %[[FILL]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
// CHECK-SAME:                 [%[[SIZE_INPUT_N]], %[[SIZE_ELEM_OH_2]], %[[SIZE_ELEM_OW_2]], %[[SIZE_ELEM_OC_2]]]
// CHECK-NEXT:           %[[ST_CONV:.+]] = linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:                 ins(%[[ST_INPUT]], %[[ST_FILTER]] : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
// CHECK-SAME:                 outs(%[[ST_FILL]] : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK-NEXT:           %[[ST_ADD:.+]] = linalg.generic
// CHECK-SAME:                 ins(%[[ST_CONV]], %[[ST_ELEM]] : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
// CHECK-SAME:                 outs(%[[ST_ARG]] : tensor<?x?x?x?xf32>)
//      CHECK:           tensor.insert_slice %[[ST_ADD]] into %[[ARG]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
// CHECK-SAME:                 [%[[SIZE_ELEM_N]], %[[SIZE_ELEM_OH]], %[[SIZE_ELEM_OW]], %[[SIZE_ELEM_OC]]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
//     CHECK: func @pad_generic_static
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG:   %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:   %[[C128:.*]] = arith.constant 128 : index
//     CHECK:   scf.for %{{.*}} = %[[C0]] to %[[C64]] step %[[C16]]
//     CHECK:     %[[CMPI1:.*]] = arith.cmpi eq
//     CHECK:     scf.for %{{.*}} = %[[C0]] to %[[C128]] step %[[C32]]
//     CHECK:       %[[CMPI2:.*]] = arith.cmpi eq
//     CHECK:       %[[HASZERO:.*]] = arith.ori %[[CMPI2]], %[[CMPI1]] : i1
//     CHECK:       scf.if %[[HASZERO]]
//     CHECK:         tensor.generate
//     CHECK:       else
//     CHECK:         tensor.extract_slice
//     CHECK:         tensor.pad
//     CHECK:       tensor.extract_slice
//     CHECK:       tensor.extract_slice
//     CHECK:       linalg.generic
//     CHECK:       tensor.insert_slice
func.func @pad_generic_static(%small_input: tensor<58x1xf32>, %large_input: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %zero = arith.constant 0.0 : f32

  %d0 = tensor.dim %large_input, %c0 : tensor<64x128xf32>
  %d1 = tensor.dim %large_input, %c1 : tensor<64x128xf32>

  %pad = tensor.pad %small_input low[4, 60] high[2, 67] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %zero : f32
  } : tensor<58x1xf32> to tensor<64x128xf32>

  %fill = linalg.fill ins(%zero : f32) outs(%large_input : tensor<64x128xf32>) -> tensor<64x128xf32>

  %for0 = scf.for %iv0 = %c0 to %d0 step %c16 iter_args(%arg0 = %fill) -> tensor<64x128xf32> {
    %for1 = scf.for %iv1 = %c0 to %d1 step %c32 iter_args(%arg1 = %arg0) -> tensor<64x128xf32> {
      %0 = tensor.extract_slice %pad[%iv0, %iv1][16, 32][1, 1] : tensor<64x128xf32> to tensor<16x32xf32>
      %1 = tensor.extract_slice %large_input[%iv0, %iv1][16, 32][1, 1] : tensor<64x128xf32> to tensor<16x32xf32>
      %2 = tensor.extract_slice %arg1[%iv0, %iv1][16, 32][1, 1] : tensor<64x128xf32> to tensor<16x32xf32>

      %add = linalg.generic
        {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%0, %1 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%2 : tensor<16x32xf32>) {
      ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
        %result = arith.addf %arg4, %arg5 : f32
        linalg.yield %result : f32
      } -> tensor<16x32xf32>

      %insert = tensor.insert_slice %add into %arg1[%iv0, %iv1] [16, 32] [1, 1]  : tensor<16x32xf32> into tensor<64x128xf32>
      scf.yield %insert : tensor<64x128xf32>
    }
    scf.yield %for1 : tensor<64x128xf32>
  }
  return %for0 : tensor<64x128xf32>
}
