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
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//  CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

/// w == 0, kw == 0
//      CHECK:   %[[V_FILTER:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//      CHECK:   %[[V_INPUT0:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//      CHECK:   %[[V_OUTPUT_0:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//      CHECK:   %[[CONTRACT0:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT0]], %[[V_FILTER]], %[[V_OUTPUT_0]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
//      CHECK:   vector.transfer_write %[[CONTRACT0]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

/// w == 1, kw == 0
//      CHECK:   %[[V_INPUT3:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C3]], %[[C0]]], %[[F0]]
//      CHECK:   %[[V_OUTPUT_1:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C1]], %[[C0]]], %[[F0]]
//      CHECK:   %[[CONTRACT1:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT3]], %[[V_FILTER]], %[[V_OUTPUT_1]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
//      CHECK:   vector.transfer_write %[[CONTRACT1]], %[[OUTPUT]][%[[C0]], %[[C1]], %[[C0]]]

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
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//  CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//  CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

/// w == 0, kw == 0
//      CHECK:   %[[V_FILTER_A:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//      CHECK:   %[[V_INPUT0_A:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//      CHECK:   %[[V_OUTPUT_0_A:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//      CHECK:   %[[CONTRACT0_A:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT0_A]], %[[V_FILTER_A]], %[[V_OUTPUT_0_A]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
//      CHECK:   vector.transfer_write %[[CONTRACT0_A]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

/// w == 0, kw == 1
//      CHECK:   %[[V_INPUT3_A:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C3]], %[[C0]]], %[[F0]]
//      CHECK:   %[[V_OUTPUT_1_A:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C1]], %[[C0]]], %[[F0]]
//      CHECK:   %[[CONTRACT1_A:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT3_A]], %[[V_FILTER_A]], %[[V_OUTPUT_1_A]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
//      CHECK:   vector.transfer_write %[[CONTRACT1_A]], %[[OUTPUT]][%[[C0]], %[[C1]], %[[C0]]]

/// w == 1, kw == 0
//      CHECK:   %[[V_FILTER_B:.+]]   = vector.transfer_read %[[FILTER]][%[[C1]], %[[C0]], %[[C0]]], %[[F0]]
//      CHECK:   %[[V_INPUT0_B:.+]]   = vector.transfer_read  %[[INPUT]][%[[C0]], %[[C2]], %[[C0]]], %[[F0]]
//      CHECK:   %[[V_OUTPUT_0_B:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//      CHECK:   %[[CONTRACT0_B:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT0_B]], %[[V_FILTER_B]], %[[V_OUTPUT_0_B]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
//      CHECK:   vector.transfer_write %[[CONTRACT0_B]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

/// w == 1, kw == 1
//      CHECK:     %[[V_INPUT3_B:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C5]], %[[C0]]], %[[F0]]
//      CHECK:   %[[V_OUTPUT_1_B:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C1]], %[[C0]]], %[[F0]]
//      CHECK:   %[[CONTRACT1_B:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT3_B]], %[[V_FILTER_B]], %[[V_OUTPUT_1_B]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
//      CHECK:   vector.transfer_write %[[CONTRACT1_B]], %[[OUTPUT]][%[[C0]], %[[C1]], %[[C0]]]

// -----



// CHECK: #[[INPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[FILTER_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK: #[[OUTPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

//      CHECK: func @conv1d_nwc_4x2x8_memref
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x6x3xf32>, %[[FILTER:.+]]: memref<2x3x8xf32>, %[[OUTPUT:.+]]: memref<4x2x8xf32>)
func @conv1d_nwc_4x2x8_memref(%input: memref<4x6x3xf32>, %filter: memref<2x3x8xf32>, %output: memref<4x2x8xf32>) {
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

/// w == 0, kw == 0
//      CHECK:   %[[V_FILTER_000:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]{{.*}} vector<3x8xf32>
//      CHECK:   %[[V_INPUT_000:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]{{.*}} vector<4x2x3xf32>
//      CHECK:   %[[V_OUTPUT_0:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]{{.*}} vector<4x2x8xf32>
//      CHECK:   %[[CONTRACT0:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT_000]], %[[V_FILTER_000]], %[[V_OUTPUT_0]]
// CHECK-SAME:     : vector<4x2x3xf32>, vector<3x8xf32> into vector<4x2x8xf32>
//      CHECK:   vector.transfer_write %[[CONTRACT0]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

/// w == 0, kw == 1
//      CHECK:   %[[V_FILTER_100:.+]] = vector.transfer_read %[[FILTER]][%[[C1]], %[[C0]], %[[C0]]], %[[F0]]{{.*}} vector<3x8xf32>
//      CHECK:   %[[V_INPUT_020:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C2]], %[[C0]]], %[[F0]]{{.*}} vector<4x2x3xf32>
//      CHECK:   %[[V_OUTPUT_1:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]{{.*}} vector<4x2x8xf32>
//      CHECK:   %[[CONTRACT1:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME:     %[[V_INPUT_020]], %[[V_FILTER_100]], %[[V_OUTPUT_1]]
// CHECK-SAME:     : vector<4x2x3xf32>, vector<3x8xf32> into vector<4x2x8xf32>
//      CHECK:   vector.transfer_write %[[CONTRACT1]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]
  linalg.conv_1d_nwc_wcf
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<4x6x3xf32>, memref<2x3x8xf32>)
    outs(%output : memref<4x2x8xf32>)
  return
}
