// RUN: mlir-opt -split-input-file -test-linalg-transform-patterns=test-linalg-to-vector-patterns %s | FileCheck %s

func @conv1d_nwc_4x2x8_memref(%input: memref<4x6x3xf32>, %filter: memref<1x3x8xf32>, %output: memref<4x2x8xf32>) {
  linalg.conv_1d_nwc_wcf
    {dilations = dense<1> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x6x3xf32>, memref<1x3x8xf32>)
    outs(%output : memref<4x2x8xf32>)
  return
}

// CHECK: #[[INPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[FILTER_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK: #[[OUTPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

//      CHECK: func @conv1d_nwc_4x2x8_memref
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x6x3xf32>, %[[FILTER:.+]]: memref<1x3x8xf32>, %[[OUTPUT:.+]]: memref<4x2x8xf32>)

//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

/// Read the whole data in one shot.
//  CHECK-DAG:   %[[V_INPUT_R:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//  CHECK-DAG:  %[[V_FILTER_R:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//  CHECK-DAG:  %[[V_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]

//      CHECK:    %[[V_FILTER:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<1x3x8xf32>
/// w == 0, kw == 0
//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x1x3xf32>
//      CHECK:  %[[V_OUTPUT_0:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 1, 8], strides = [1, 1, 1]} : vector<4x2x8xf32> to vector<4x1x8xf32>
/// w == 1, kw == 0
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 3, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x1x3xf32>
//      CHECK:  %[[V_OUTPUT_1:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 1, 0], sizes = [4, 1, 8], strides = [1, 1, 1]} : vector<4x2x8xf32> to vector<4x1x8xf32>

/// w == 0, kw == 0
//      CHECK:   %[[CONTRACT_0:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT_0]], %[[V_FILTER]], %[[V_OUTPUT_0]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>

/// w == 1, kw == 0
//      CHECK:   %[[CONTRACT_1:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT_1]], %[[V_FILTER]], %[[V_OUTPUT_1]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>

/// w == 0, kw == 0
//      CHECK:   %[[RES_0:.+]] = vector.insert_strided_slice %[[CONTRACT_0]], %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<4x1x8xf32> into vector<4x2x8xf32>
/// w == 1, kw == 0
//      CHECK:   %[[RES_1:.+]] = vector.insert_strided_slice %[[CONTRACT_1]], %[[RES_0]]
// CHECK-SAME:     {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<4x1x8xf32> into vector<4x2x8xf32>

// Write the result back in one shot.
//      CHECK:   vector.transfer_write %[[RES_1]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

// -----

func @conv1d_nwc_4x2x8_memref(%input: memref<4x6x3xf32>, %filter: memref<2x3x8xf32>, %output: memref<4x2x8xf32>) {
  linalg.conv_1d_nwc_wcf
    {dilations = dense<2> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x6x3xf32>, memref<2x3x8xf32>)
    outs(%output : memref<4x2x8xf32>)
  return
}

// CHECK: #[[INPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[FILTER_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK: #[[OUTPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

//      CHECK: func @conv1d_nwc_4x2x8_memref
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x6x3xf32>, %[[FILTER:.+]]: memref<2x3x8xf32>, %[[OUTPUT:.+]]: memref<4x2x8xf32>)

//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

/// Read the whole data in one shot.
//  CHECK-DAG:   %[[V_INPUT_R:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//  CHECK-DAG:   %[[V_FILTER_R:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//  CHECK-DAG:   %[[V_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]


/// w == 0, kw == 0
//      CHECK:  %[[V_FILTER_0:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<2x3x8xf32>
//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
//      CHECK:  %[[V_OUTPUT_0:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 1, 8], strides = [1, 1, 1]} : vector<4x2x8xf32> to vector<4x1x8xf32>
/// w == 1, kw == 0
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 3, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
//      CHECK:  %[[V_OUTPUT_1:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 1, 0], sizes = [4, 1, 8], strides = [1, 1, 1]} : vector<4x2x8xf32> to vector<4x1x8xf32>

/// w == 0, kw == 1
//      CHECK:  %[[V_FILTER_1:.+]] = vector.extract %[[V_FILTER_R]][1] : vector<2x3x8xf32>
//      CHECK:   %[[V_INPUT_2:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 2, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
/// w == 1, kw == 0
//      CHECK:   %[[V_INPUT_3:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 5, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>

/// w == 0, kw == 0
//      CHECK:   %[[CONTRACT_0:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT_0]], %[[V_FILTER_0]], %[[V_OUTPUT_0]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
/// w == 1, kw == 0
//      CHECK:   %[[CONTRACT_1:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT_1]], %[[V_FILTER_0]], %[[V_OUTPUT_1]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
/// w == 1, kw == 1
//      CHECK:   %[[CONTRACT_2:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT_2]], %[[V_FILTER_1]], %[[CONTRACT_0]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
/// w == 1, kw == 1
//      CHECK:   %[[CONTRACT_3:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT_3]], %[[V_FILTER_1]], %[[CONTRACT_1]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>

/// w == 0, kw == 0
//      CHECK:   %[[RES_0:.+]] = vector.insert_strided_slice %[[CONTRACT_2]], %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<4x1x8xf32> into vector<4x2x8xf32>
/// w == 1, kw == 0
//      CHECK:   %[[RES_1:.+]] = vector.insert_strided_slice %[[CONTRACT_3]], %[[RES_0]]
// CHECK-SAME:     {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<4x1x8xf32> into vector<4x2x8xf32>

// Write the result back in one shot.
//      CHECK:   vector.transfer_write %[[RES_1]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

// -----

func @conv1d_nwc_4x2x8_memref(%input: memref<4x6x3xf32>, %filter: memref<2x3x8xf32>, %output: memref<4x2x8xf32>) {
  linalg.conv_1d_nwc_wcf
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<4x6x3xf32>, memref<2x3x8xf32>)
    outs(%output : memref<4x2x8xf32>)
  return
}

// CHECK: #[[INPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[FILTER_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK: #[[OUTPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

//      CHECK: func @conv1d_nwc_4x2x8_memref
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x6x3xf32>, %[[FILTER:.+]]: memref<2x3x8xf32>, %[[OUTPUT:.+]]: memref<4x2x8xf32>)

//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

/// Read the whole data in one shot.
//  CHECK-DAG:   %[[V_INPUT_R:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//  CHECK-DAG:  %[[V_FILTER_R:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//  CHECK-DAG:  %[[V_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]

/// w == 0, kw == 0
//      CHECK:  %[[V_FILTER_0:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<2x3x8xf32>
//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 2, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x2x3xf32>
/// w == 0, kw == 1
//      CHECK:  %[[V_FILTER_1:.+]] = vector.extract %[[V_FILTER_R]][1] : vector<2x3x8xf32>
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 2, 0], sizes = [4, 2, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x2x3xf32>

/// w == 0, kw == 0
//      CHECK:   %[[CONTRACT_0:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT_0]], %[[V_FILTER_0]], %[[V_OUTPUT_R]]
// CHECK-SAME:     : vector<4x2x3xf32>, vector<3x8xf32> into vector<4x2x8xf32>
/// w == 0, kw == 1
//      CHECK:   %[[CONTRACT_1:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT_1]], %[[V_FILTER_1]], %[[CONTRACT_0]]
// CHECK-SAME:     : vector<4x2x3xf32>, vector<3x8xf32> into vector<4x2x8xf32>

// Write the result back in one shot.
//      CHECK:   vector.transfer_write %[[CONTRACT_1]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

// -----

func @depthwise_conv1d_nwc_wc_3x5x4_memref(%input: memref<3x5x4xf32>, %filter: memref<2x4xf32>, %output: memref<3x2x4xf32>) {
  linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<3x5x4xf32>, memref<2x4xf32>)
    outs(%output : memref<3x2x4xf32>)
  return
}

//       CHECK: func @depthwise_conv1d_nwc_wc_3x5x4_memref
//  CHECK-SAME:   (%[[INPUT:[0-9a-z]+]]: memref<3x5x4xf32>, %[[FILTER:[0-9a-z]+]]: memref<2x4xf32>, %[[OUTPUT:[0-9a-z]+]]: memref<3x2x4xf32>)

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

/// Read the whole data in one shot.
//      CHECK:   %[[V_INPUT_R:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]]
//      CHECK:  %[[V_FILTER_R:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]]]
//      CHECK:  %[[V_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

/// w == 0, kw == 0
//      CHECK:  %[[V_FILTER_0:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<2x4xf32>
//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x4xf32> to vector<3x2x4xf32>
/// w == 0, kw == 1
//      CHECK:  %[[V_FILTER_1:.+]] = vector.extract %[[V_FILTER_R]][1] : vector<2x4xf32>
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 2, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x4xf32> to vector<3x2x4xf32>

/// w == 0, kw = 0
//      CHECK:  %[[B_FILTER_0:.*]] = vector.broadcast %[[V_FILTER_0]] : vector<4xf32> to vector<3x2x4xf32>
//      CHECK:  %[[FMA_0:.*]] = vector.fma %[[V_INPUT_0]], %[[B_FILTER_0]], %[[V_OUTPUT_R]] : vector<3x2x4xf32>

/// w == 0, kw = 1
//      CHECK:  %[[B_FILTER_1:.*]] = vector.broadcast %[[V_FILTER_1]] : vector<4xf32> to vector<3x2x4xf32>
//      CHECK:  %[[FMA_1:.*]] = vector.fma %[[V_INPUT_1]], %[[B_FILTER_1]], %[[FMA_0]] : vector<3x2x4xf32>

// Write the result back in one shot.
//      CHECK:   vector.transfer_write %[[FMA_1]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]
