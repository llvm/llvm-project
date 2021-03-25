// RUN: mlir-opt %s -test-linalg-greedy-fusion -split-input-file | FileCheck %s

func @matmul_tensors(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %t0 = linalg.matmul ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>

  %c4 = constant 4 : index
  %c2 = constant 2 : index
  %c0 = constant 0 : index
  %c3 = constant 3 : index
  %c1 = constant 1 : index
  %0 = memref.dim %t0, %c0 : tensor<?x?xf32>
  %1 = memref.dim %t0, %c1 : tensor<?x?xf32>
  %2 = memref.dim %arg1, %c1 : tensor<?x?xf32>
  %3 = scf.for %arg3 = %c0 to %0 step %c2 iter_args(%arg4 = %arg2) -> (tensor<?x?xf32>) {
    %4 = scf.for %arg5 = %c0 to %2 step %c3 iter_args(%arg6 = %arg4) -> (tensor<?x?xf32>) {
      %5 = scf.for %arg7 = %c0 to %1 step %c4 iter_args(%arg8 = %arg6) -> (tensor<?x?xf32>) {
        %6 = subtensor %t0[%arg3, %arg7][%c2, 4][1, 1] : tensor<?x?xf32> to tensor<?x4xf32>
        %7 = subtensor %arg1[%arg7, %arg5][4, %c3][1, 1] : tensor<?x?xf32> to tensor<4x?xf32>
        %8 = subtensor %arg8[%arg3, %arg5][%c2, %c3][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %9 = linalg.matmul ins(%6, %7 : tensor<?x4xf32>, tensor<4x?xf32>) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %10 = subtensor_insert %9 into %arg8[%arg3, %arg5] [%c2, %c3] [1, 1]  : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %10 : tensor<?x?xf32>
      }
      scf.yield %5 : tensor<?x?xf32>
    }
    scf.yield %4 : tensor<?x?xf32>
  }
  return %3 : tensor<?x?xf32>
}

//       CHECK: #[[BOUND2_MAP:.+]] = affine_map<(d0)[s0] -> (2, -d0 + s0)>
//       CHECK: #[[BOUND4_MAP:.+]] = affine_map<(d0)[s0] -> (4, -d0 + s0)>

//       CHECK: func @matmul_tensors(
//  CHECK-SAME: %[[A:[0-9a-z]*]]: tensor<?x?xf32>
//  CHECK-SAME: %[[B:[0-9a-z]*]]: tensor<?x?xf32>
//  CHECK-SAME: %[[C:[0-9a-z]*]]: tensor<?x?xf32>

//   CHECK-DAG: %[[C0:.*]] = constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = constant 1 : index
//   CHECK-DAG: %[[dA0:.*]] = memref.dim %[[A]], %[[C0]] : tensor<?x?xf32>
//   CHECK-DAG: %[[dA1:.*]] = memref.dim %[[A]], %[[C1]] : tensor<?x?xf32>
//   CHECK-DAG: %[[dB0:.*]] = memref.dim %[[B]], %[[C0]] : tensor<?x?xf32>
//   CHECK-DAG: %[[dB1:.*]] = memref.dim %[[B]], %[[C1]] : tensor<?x?xf32>
//   CHECK-DAG: %[[dC0:.*]] = memref.dim %[[C]], %[[C0]] : tensor<?x?xf32>
//   CHECK-DAG: %[[dC1:.*]] = memref.dim %[[C]], %[[C1]] : tensor<?x?xf32>
//       CHECK: scf.for %[[I:[0-9a-z]*]]
//       CHECK:   %[[sizeA0:.*]] = affine.min #[[BOUND2_MAP]](%[[I]])[%[[dA0]]]
//       CHECK:   %[[stA:.*]] = subtensor %[[A]][%[[I]], 0] [%[[sizeA0]], %[[dA1]]] [1, 1]  : tensor<?x?xf32> to tensor<?x?xf32>
//       CHECK:   %[[sizeC0:.*]] = affine.min #[[BOUND2_MAP]](%[[I]])[%[[dC0]]]
//  CHECK-NEXT:   scf.for %[[J:[0-9a-z]*]]
//  CHECK-NEXT:     scf.for %[[K:[0-9a-z]*]] {{.*}} iter_args(%[[RES:[0-9a-z]*]]
//   CHECK-DAG:       %[[stB1:.*]] = subtensor %[[B]][%[[K]], %[[J]]] [4, 3] [1, 1]  : tensor<?x?xf32> to tensor<4x3xf32>
//   CHECK-DAG:       %[[stF:.*]] = subtensor %[[RES]][%[[I]], %[[J]]] [2, 3] [1, 1]  : tensor<?x?xf32> to tensor<2x3xf32>
//
// subtensors of the producing matmul.
//       CHECK:       %[[sizeB1:.*]] = affine.min #[[BOUND4_MAP]](%[[K]])[%[[dB1]]]
//       CHECK:       %[[stB2:.*]] = subtensor %[[B]][0, %[[K]]] [%[[dB0]], %[[sizeB1]]] [1, 1]  : tensor<?x?xf32> to tensor<?x?xf32>
//       CHECK:       %[[sizeC1:.*]] = affine.min #[[BOUND4_MAP]](%[[K]])[%[[dC1]]]
//       CHECK:       %[[stC:.*]] = subtensor %[[C]][%[[I]], %[[K]]] [%[[sizeC0]], %[[sizeC1]]] [1, 1]  : tensor<?x?xf32> to tensor<?x?xf32>
//       CHECK:       %[[stD:.*]] = linalg.matmul ins(%[[stA]], %[[stB2]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[stC]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
//       CHECK:       %[[CAST:.*]] = tensor.cast %[[stD]] : tensor<?x?xf32> to tensor<?x4xf32>
//  CHECK-NEXT:       %[[stG:.*]] = linalg.matmul ins(%[[CAST]], %[[stB1]] : tensor<?x4xf32>, tensor<4x3xf32>) outs(%[[stF]] : tensor<2x3xf32>)  -> tensor<2x3xf32>
//  CHECK-NEXT:       subtensor_insert %[[stG]] into %[[RES]][%[[I]], %[[J]]]

// -----

func @conv_tensors_static(%input: tensor<1x225x225x32xf32>, %filter: tensor<3x3x3x32xf32>, %elementwise: tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32> {
  %c112 = constant 112 : index
  %c32 = constant 32 : index
  %c16 = constant 16 : index
  %c8 = constant 8 : index
  %c4 = constant 4 : index
  %c0 = constant 0 : index
  %cst = constant 0.0 : f32

  %init = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
  %fill = linalg.fill(%init, %cst) : tensor<1x112x112x32xf32>, f32 -> tensor<1x112x112x32xf32>

  %conv = linalg.conv_2d_input_nhwc_filter_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%input, %filter : tensor<1x225x225x32xf32>, tensor<3x3x3x32xf32>)
    outs(%fill : tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>

  %for0 = scf.for %iv0 = %c0 to %c112 step %c8 iter_args(%arg0 = %fill) -> tensor<1x112x112x32xf32> {
    %for1 = scf.for %iv1 = %c0 to %c112 step %c16 iter_args(%arg1 = %arg0) -> tensor<1x112x112x32xf32> {
      %for2 = scf.for %iv2 = %c0 to %c32 step %c4 iter_args(%arg2 = %arg1) -> tensor<1x112x112x32xf32> {
        %0 = subtensor %conv[0, %iv0, %iv1, %iv2][1, 8, 16, 4][1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x8x16x4xf32>
        %1 = subtensor %elementwise[0, %iv0, %iv1, %iv2][1, 8, 16, 4][1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x8x16x4xf32>
        %2 = subtensor %arg2[0, %iv0, %iv1, %iv2][1, 8, 16, 4][1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x8x16x4xf32>
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
          %result = addf %arg3, %arg4 : f32
          linalg.yield %result : f32
        } -> tensor<1x8x16x4xf32>

        %insert = subtensor_insert %add into %arg2[0, %iv0, %iv1, %iv2] [1, 8, 16, 4] [1, 1, 1, 1]  : tensor<1x8x16x4xf32> into tensor<1x112x112x32xf32>
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
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x225x225x32xf32>, %[[FILTER:.+]]: tensor<3x3x3x32xf32>, %[[ELEM:.+]]: tensor<1x112x112x32xf32>)

//      CHECK: %[[INIT:.+]] = linalg.init_tensor [1, 112, 112, 32] : tensor<1x112x112x32xf32>
// CHECK-NEXT: %[[FILL:.+]] = linalg.fill(%[[INIT]], %cst) : tensor<1x112x112x32xf32>, f32 -> tensor<1x112x112x32xf32>

// CHECK-NEXT: scf.for %[[IV0:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ARG0:.+]] = %[[FILL]])
// CHECK-NEXT:   %[[OFFSET_H:.+]] = affine.apply #[[MAP0]](%[[IV0]])
// CHECK-NEXT:   scf.for %[[IV1:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ARG1:.+]] = %[[ARG0]])
// CHECK-NEXT:     %[[OFFSET_W:.+]] = affine.apply #[[MAP0]](%[[IV1]])
// CHECK-NEXT:     %[[ST_INPUT:.+]] = subtensor %arg0[0, %[[OFFSET_H]], %[[OFFSET_W]], 0] [1, 17, 33, 32] [1, 1, 1, 1] : tensor<1x225x225x32xf32> to tensor<1x17x33x32xf32>
// CHECK-NEXT:     scf.for %[[IV2:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ARG2:.+]] = %[[ARG1]])
// CHECK-NEXT:       %[[ST_ELEM:.+]] = subtensor %[[ELEM]][0, %[[IV0]], %[[IV1]], %[[IV2]]] [1, 8, 16, 4] [1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x8x16x4xf32>
// CHECK-NEXT:       %[[ST_ARG2:.+]] = subtensor %[[ARG2]][0, %[[IV0]], %[[IV1]], %[[IV2]]] [1, 8, 16, 4] [1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x8x16x4xf32>
// CHECK-NEXT:       %[[ST_FILTER:.+]] = subtensor %[[FILTER]][0, 0, 0, %[[IV2]]] [3, 3, 3, 4] [1, 1, 1, 1] : tensor<3x3x3x32xf32> to tensor<3x3x3x4xf32>
// CHECK-NEXT:       %[[ST_FILL:.+]] = subtensor %[[FILL]][0, %[[IV0]], %[[IV1]], %[[IV2]]] [1, 8, 16, 4] [1, 1, 1, 1] : tensor<1x112x112x32xf32> to tensor<1x8x16x4xf32>
// CHECK-NEXT:       %[[ST_CONV:.+]] = linalg.conv_2d_input_nhwc_filter_hwcf
// CHECK-SAME:         ins(%[[ST_INPUT]], %[[ST_FILTER]] : tensor<1x17x33x32xf32>, tensor<3x3x3x4xf32>)
// CHECK-SAME:         outs(%[[ST_FILL]] : tensor<1x8x16x4xf32>)
// CHECK-NEXT:       %[[ADD:.+]] = linalg.generic
// CHECK-SAME:         ins(%[[ST_CONV]], %[[ST_ELEM]] : tensor<1x8x16x4xf32>, tensor<1x8x16x4xf32>)
// CHECK-SAME:         outs(%[[ST_ARG2]] : tensor<1x8x16x4xf32>)
//      CHECK:       subtensor_insert %[[ADD]] into %[[ARG2]][0, %[[IV0]], %[[IV1]], %[[IV2]]] [1, 8, 16, 4]

// -----

#bound4_map = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#bound8_map = affine_map<(d0)[s0] -> (8, -d0 + s0)>
#bound16_map = affine_map<(d0)[s0] -> (16, -d0 + s0)>

func @conv_tensors_dynamic(%input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?x?xf32>, %elementwise: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %cst = constant 0.0 : f32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c8 = constant 8 : index
  %c16 = constant 16 : index

  %n = memref.dim %elementwise, %c0 : tensor<?x?x?x?xf32>
  %oh = memref.dim %elementwise, %c1 : tensor<?x?x?x?xf32>
  %ow = memref.dim %elementwise, %c2 : tensor<?x?x?x?xf32>
  %oc = memref.dim %elementwise, %c3 : tensor<?x?x?x?xf32>

  %init = linalg.init_tensor [%n, %oh, %ow, %oc] : tensor<?x?x?x?xf32>
  %fill = linalg.fill(%init, %cst) : tensor<?x?x?x?xf32>, f32 -> tensor<?x?x?x?xf32>

  %conv = linalg.conv_2d_input_nhwc_filter_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%input, %filter : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
    outs(%fill : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

  %for0 = scf.for %iv0 = %c0 to %oh step %c8 iter_args(%arg0 = %fill) -> tensor<?x?x?x?xf32> {
    %for1 = scf.for %iv1 = %c0 to %ow step %c16 iter_args(%arg1 = %arg0) -> tensor<?x?x?x?xf32> {
      %for2 = scf.for %iv2 = %c0 to %oc step %c4 iter_args(%arg2 = %arg1) -> tensor<?x?x?x?xf32> {
        %for3 = scf.for %iv3 = %c0 to %oc step %c2 iter_args(%arg3 = %arg2) -> tensor<?x?x?x?xf32> {
          %n_size = affine.min #bound8_map(%iv0)[%n]
          %oh_size = affine.min #bound16_map(%iv1)[%oh]
          %ow_size = affine.min #bound4_map(%iv2)[%ow]
          %oc_size = affine.min #bound4_map(%iv2)[%oc]
          %0 = subtensor %conv[%iv0, %iv1, %iv2, %iv3][%n_size, %oh_size, %ow_size, %oc_size][1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
          %1 = subtensor %elementwise[%iv0, %iv1, %iv2, %iv3][%n_size, %oh_size, %ow_size, %oc_size][1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
          %2 = subtensor %arg3[%iv0, %iv1, %iv2, %iv3][%n_size, %oh_size, %ow_size, %oc_size][1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x?x?x?xf32>
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
            %result = addf %arg4, %arg5 : f32
            linalg.yield %result : f32
          } -> tensor<?x?x?x?xf32>

          %insert = subtensor_insert %add into %arg3[%iv0, %iv1, %iv2, %iv3] [%n_size, %oh_size, %ow_size, %oc_size] [1, 1, 1, 1]  : tensor<?x?x?x?xf32> into tensor<?x?x?x?xf32>
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

// -----

// CHECK: #[[BOUND8_MAP:.+]] = affine_map<(d0)[s0] -> (8, -d0 + s0)>
// CHECK: #[[BOUND8_MAP_2:.+]] = affine_map<(d0)[s0, s1] -> (-d0 + s0, 8, -d0 + s1)>
// CHECK: #[[BOUND8_MAP_3:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 8)>
// CHECK: #[[BOUND16_MAP:.+]] = affine_map<(d0)[s0] -> (16, -d0 + s0)>
// CHECK: #[[X2_MAP:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK: #[[INPUT_BOUND:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * 2 + s0 - 2, d1 * -2 + s1)>
// CHECK: #[[BOUND16_MAP_2:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
// CHECK: #[[BOUND4_MAP:.+]] = affine_map<(d0)[s0] -> (4, -d0 + s0)>
// CHECK: #[[BOUND4_MAP_2:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
// CHECK: #[[BOUND4_MAP_3:.+]] = affine_map<(d0, d1)[s0, s1] -> (-d0 + s0, 4, -d1 + s1)>
// CHECK: #[[BOUND4_MAP_4:.+]] = affine_map<(d0, d1)[s0] -> (-d0 + s0, 4, -d1 + s0)>

//      CHECK: func @conv_tensors_dynamic
// CHECK-SAME: (%[[INPUT]]: tensor<?x?x?x?xf32>, %[[FILTER]]: tensor<?x?x?x?xf32>, %[[ELEM]]: tensor<?x?x?x?xf32>)

//  CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//  CHECK-DAG:   %[[C3:.+]] = constant 3 : index

//  CHECK-DAG:   %[[ELEM_N:.+]] = memref.dim %[[ELEM]], %[[C0]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[ELEM_OH:.+]] = memref.dim %[[ELEM]], %[[C1]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[ELEM_OW:.+]] = memref.dim %[[ELEM]], %[[C2]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[ELEM_OC:.+]] = memref.dim %[[ELEM]], %[[C3]] : tensor<?x?x?x?xf32>

//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[ELEM_N]], %[[ELEM_OH]], %[[ELEM_OW]], %[[ELEM_OC]]] : tensor<?x?x?x?xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill(%[[INIT]], %cst) : tensor<?x?x?x?xf32>, f32 -> tensor<?x?x?x?xf32>

//  CHECK-DAG:   %[[FILTER_H:.+]] = memref.dim %[[FILTER]], %[[C0]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[FILTER_W:.+]] = memref.dim %[[FILTER]], %[[C1]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[INPUT_N:.+]] = memref.dim %[[INPUT]], %[[C0]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[INPUT_H:.+]] = memref.dim %[[INPUT]], %[[C1]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[INPUT_W:.+]] = memref.dim %[[INPUT]], %[[C2]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[INPUT_C:.+]] = memref.dim %[[INPUT]], %[[C3]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[FILTER_IC:.+]] = memref.dim %[[FILTER]], %[[C2]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[FILTER_OC:.+]] = memref.dim %[[FILTER]], %[[C3]] : tensor<?x?x?x?xf32>

//      CHECK:   scf.for %[[IV0:.+]] = %{{.+}} to %[[ELEM_OH]] step %{{.+}} iter_args(%{{.+}} = %[[FILL]])
// CHECK-NEXT:     %[[SIZE_ELEM_N:.+]] = affine.min #[[BOUND8_MAP]](%[[IV0]])[%[[ELEM_N]]]
// CHECK-NEXT:     %[[SIZE_INPUT_N:.+]] = affine.min #[[BOUND8_MAP_2]](%[[IV0]])[%[[INPUT_N]], %[[ELEM_N]]]
// CHECK-NEXT:     %[[SIZE_ELEM_N_2:.+]] = affine.min #[[BOUND8_MAP_3]](%[[IV0]])[%[[ELEM_N]]]
// CHECK-NEXT:     scf.for %[[IV1:.+]] = %{{.+}} to %[[ELEM_OW]]
// CHECK-NEXT:       %[[SIZE_ELEM_OH:.+]] = affine.min #[[BOUND16_MAP]](%[[IV1]])[%[[ELEM_OH]]]
// CHECK-NEXT:       %[[OFFSET_OH:.+]] = affine.apply #[[X2_MAP]](%[[IV1]])
// CHECK-NEXT:       %[[SIZE_INPUT_H:.+]] = affine.min #[[INPUT_BOUND]](%[[SIZE_ELEM_OH]], %[[IV1]])[%[[FILTER_H]], %[[INPUT_H]]]
// CHECK-NEXT:       %[[SIZE_ELEM_OH_2:.+]] = affine.min #[[BOUND16_MAP_2]](%[[IV1]])[%[[ELEM_OH]]]
// CHECK-NEXT:       scf.for %[[IV2:.+]] = %{{.+}} to %[[ELEM_OC]]
// CHECK-NEXT:         %[[SIZE_ELEM_OW:.+]] = affine.min #[[BOUND4_MAP]](%[[IV2]])[%[[ELEM_OW]]]
// CHECK-NEXT:         %[[SIZE_ELEM_OC:.+]] = affine.min #[[BOUND4_MAP]](%[[IV2]])[%[[ELEM_OC]]]
// CHECK-NEXT:         %[[OFFSET_OW:.+]] = affine.apply #[[X2_MAP]](%[[IV2]])
// CHECK-NEXT:         %[[SIZE_INPUT_W:.+]] = affine.min #[[INPUT_BOUND]](%[[SIZE_ELEM_OW]], %[[IV2]])[%[[FILTER_W]], %[[INPUT_W]]]
// CHECK-NEXT:         %[[ST_INPUT:.+]] = subtensor %[[INPUT]][%[[IV0]], %[[OFFSET_OH]], %[[OFFSET_OW]], 0]
// CHECK-SAME:               [%[[SIZE_INPUT_N]], %[[SIZE_INPUT_H]], %[[SIZE_INPUT_W]], %[[INPUT_C]]]
// CHECK-NEXT:         %[[SIZE_ELEM_OW_2:.+]] = affine.min #[[BOUND4_MAP_2]](%[[IV2]])[%[[ELEM_OW]]]
// CHECK-NEXT:         scf.for %[[IV3:.+]] = %{{.+}} to %[[ELEM_OC]] step %{{.+}} iter_args(%[[ARG:[a-z0-9]+]]
// CHECK-NEXT:           %[[ST_ELEM:.+]] = subtensor %[[ELEM]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
// CHECK-SAME:                 [%[[SIZE_ELEM_N]], %[[SIZE_ELEM_OH]], %[[SIZE_ELEM_OW]], %[[SIZE_ELEM_OC]]]
// CHECK-NEXT:           %[[ST_ARG:.+]] = subtensor %[[ARG]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
// CHECK-SAME:                 [%[[SIZE_ELEM_N]], %[[SIZE_ELEM_OH]], %[[SIZE_ELEM_OW]], %[[SIZE_ELEM_OC]]]
// CHECK-NEXT:           %[[SIZE_ELEM_OC_2:.+]] = affine.min #[[BOUND4_MAP_3]](%[[IV3]], %[[IV2]])[%[[FILTER_OC]], %[[ELEM_OC]]]
// CHECK-NEXT:           %[[ST_FILTER:.+]] = subtensor %[[FILTER]][0, 0, 0, %[[IV3]]]
// CHECK-SAME:                 [%[[FILTER_H]], %[[FILTER_W]], %[[FILTER_IC]], %[[SIZE_ELEM_OC_2]]]
// CHECK-NEXT:           %[[SIZE_ELEM_OC_3:.+]] = affine.min #[[BOUND4_MAP_4]](%[[IV3]], %[[IV2]])[%[[ELEM_OC]]]
// CHECK-NEXT:           %[[ST_FILL:.+]] = subtensor %[[FILL]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
// CHECK-SAME:                 [%[[SIZE_ELEM_N_2]], %[[SIZE_ELEM_OH_2]], %[[SIZE_ELEM_OW_2]], %[[SIZE_ELEM_OC_3]]]
// CHECK-NEXT:           %[[ST_CONV:.+]] = linalg.conv_2d_input_nhwc_filter_hwcf
// CHECK-SAME:                 ins(%[[ST_INPUT]], %[[ST_FILTER]] : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
// CHECK-SAME:                 outs(%[[ST_FILL]] : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK-NEXT:           %[[ST_ADD:.+]] = linalg.generic
// CHECK-SAME:                 ins(%[[ST_CONV]], %[[ST_ELEM]] : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
// CHECK-SAME:                 outs(%[[ST_ARG]] : tensor<?x?x?x?xf32>)
//      CHECK:           subtensor_insert %[[ST_ADD]] into %[[ARG]][%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
// CHECK-SAME:                 [%[[SIZE_ELEM_N]], %[[SIZE_ELEM_OH]], %[[SIZE_ELEM_OW]], %[[SIZE_ELEM_OC]]]
