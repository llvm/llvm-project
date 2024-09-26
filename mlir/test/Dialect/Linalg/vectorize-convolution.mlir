// RUN: mlir-opt -split-input-file -test-linalg-transform-patterns=test-linalg-to-vector-patterns %s | FileCheck %s

func.func @conv1d_nwc_4x2x8_memref(%input: memref<4x6x3xf32>, %filter: memref<1x3x8xf32>, %output: memref<4x2x8xf32>) {
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

//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x1x3xf32>
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 3, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x1x3xf32>

//      CHECK:    %[[V_FILTER:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<3x8xf32> from vector<1x3x8xf32>

//      CHECK:  %[[V_OUTPUT_0:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 1, 8], strides = [1, 1, 1]} : vector<4x2x8xf32> to vector<4x1x8xf32>
//      CHECK:  %[[V_OUTPUT_1:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 1, 0], sizes = [4, 1, 8], strides = [1, 1, 1]} : vector<4x2x8xf32> to vector<4x1x8xf32>

/// w == 0, kw == 0
//      CHECK:   %[[CONTRACT_0:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:       kind = #vector.kind<add>
// CHECK-SAME:     %[[V_INPUT_0]], %[[V_FILTER]], %[[V_OUTPUT_0]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>

/// w == 1, kw == 0
//      CHECK:   %[[CONTRACT_1:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:       kind = #vector.kind<add>
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

// This test is same as above but for i1 type with the only difference being that
// the combining kind for `vector.contract` is `OR`.
func.func @conv1d_nwc_4x2x8_memref_i1(%input: memref<4x6x3xi1>, %filter: memref<1x3x8xi1>, %output: memref<4x2x8xi1>) {
  linalg.conv_1d_nwc_wcf
    {dilations = dense<1> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x6x3xi1>, memref<1x3x8xi1>)
    outs(%output : memref<4x2x8xi1>)
  return
}
// CHECK: #[[INPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[FILTER_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK: #[[OUTPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

//      CHECK: func @conv1d_nwc_4x2x8_memref_i1
/// w == 0, kw == 0
//      CHECK:   %[[CONTRACT_0:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:       kind = #vector.kind<or>
// CHECK-SAME:       : vector<4x1x3xi1>, vector<3x8xi1> into vector<4x1x8xi1>

/// w == 1, kw == 0
//      CHECK:   %[[CONTRACT_1:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:       kind = #vector.kind<or>
// CHECK-SAME:       : vector<4x1x3xi1>, vector<3x8xi1> into vector<4x1x8xi1>

// -----

// The i8i8i32 case is similar to f32 case, so checking one case is enough for
// test coverage.
func.func @conv1d_nwc_4x2x8_i8i8i32_memref(%input: memref<4x6x3xi8>, %filter: memref<1x3x8xi8>, %output: memref<4x2x8xi32>) {
  linalg.conv_1d_nwc_wcf
    {dilations = dense<1> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x6x3xi8>, memref<1x3x8xi8>)
    outs(%output : memref<4x2x8xi32>)
  return
}

// CHECK: #[[INPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[FILTER_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK: #[[OUTPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

//      CHECK: func @conv1d_nwc_4x2x8_i8i8i32_memref
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x6x3xi8>, %[[FILTER:.+]]: memref<1x3x8xi8>, %[[OUTPUT:.+]]: memref<4x2x8xi32>)

//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C0_I8:.+]] = arith.constant 0 : i8
//  CHECK-DAG:   %[[C0_I32:.+]] = arith.constant 0 : i32

/// Read the whole data in one shot.
//  CHECK-DAG:   %[[V_INPUT_R:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[C0_I8]]
//  CHECK-DAG:  %[[V_FILTER_R:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]], %[[C0]]], %[[C0_I8]]
//  CHECK-DAG:  %[[V_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[C0_I32]]

//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xi8> to vector<4x1x3xi8>
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 3, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xi8> to vector<4x1x3xi8>

//      CHECK:    %[[V_FILTER:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<3x8xi8> from vector<1x3x8xi8>

//      CHECK:  %[[V_OUTPUT_0:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 1, 8], strides = [1, 1, 1]} : vector<4x2x8xi32> to vector<4x1x8xi32>
//      CHECK:  %[[V_OUTPUT_1:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 1, 0], sizes = [4, 1, 8], strides = [1, 1, 1]} : vector<4x2x8xi32> to vector<4x1x8xi32>

/// w == 0, kw == 0
//      CHECK:   %[[EXT_LHS_0:.+]] = arith.extsi %[[V_INPUT_0]] : vector<4x1x3xi8> to vector<4x1x3xi32>
//      CHECK:   %[[EXT_RHS_0:.+]] = arith.extsi %[[V_FILTER]] : vector<3x8xi8> to vector<3x8xi32>
//      CHECK:   %[[CONTRACT_0:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     %[[EXT_LHS_0]], %[[EXT_RHS_0]], %[[V_OUTPUT_0]]
// CHECK-SAME:     : vector<4x1x3xi32>, vector<3x8xi32> into vector<4x1x8xi32>

/// w == 1, kw == 0
//      CHECK:   %[[EXT_LHS_1:.+]] = arith.extsi %[[V_INPUT_1]] : vector<4x1x3xi8> to vector<4x1x3xi32>
//      CHECK:   %[[EXT_RHS_1:.+]] = arith.extsi %[[V_FILTER]] : vector<3x8xi8> to vector<3x8xi32>
//      CHECK:   %[[CONTRACT_1:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     %[[EXT_LHS_1]], %[[EXT_RHS_1]], %[[V_OUTPUT_1]]
// CHECK-SAME:     : vector<4x1x3xi32>, vector<3x8xi32> into vector<4x1x8xi32>

/// w == 0, kw == 0
//      CHECK:   %[[RES_0:.+]] = vector.insert_strided_slice %[[CONTRACT_0]], %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<4x1x8xi32> into vector<4x2x8xi32>
/// w == 1, kw == 0
//      CHECK:   %[[RES_1:.+]] = vector.insert_strided_slice %[[CONTRACT_1]], %[[RES_0]]
// CHECK-SAME:     {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<4x1x8xi32> into vector<4x2x8xi32>

// Write the result back in one shot.
//      CHECK:   vector.transfer_write %[[RES_1]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

// -----

func.func @conv1d_nwc_4x2x8_memref(%input: memref<4x6x3xf32>, %filter: memref<2x3x8xf32>, %output: memref<4x2x8xf32>) {
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

//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 3, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
//      CHECK:   %[[V_INPUT_2:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 2, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
//      CHECK:   %[[V_INPUT_3:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 5, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>

//      CHECK:  %[[V_FILTER_0:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<3x8xf32> from vector<2x3x8xf32>
//      CHECK:  %[[V_FILTER_1:.+]] = vector.extract %[[V_FILTER_R]][1] : vector<3x8xf32> from vector<2x3x8xf32>

//      CHECK:  %[[V_OUTPUT_0:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 1, 8], strides = [1, 1, 1]} : vector<4x2x8xf32> to vector<4x1x8xf32>
//      CHECK:  %[[V_OUTPUT_1:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 1, 0], sizes = [4, 1, 8], strides = [1, 1, 1]} : vector<4x2x8xf32> to vector<4x1x8xf32>

/// w == 0, kw == 0
//      CHECK:   %[[CONTRACT_0:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     %[[V_INPUT_0]], %[[V_FILTER_0]], %[[V_OUTPUT_0]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
/// w == 1, kw == 0
//      CHECK:   %[[CONTRACT_1:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     %[[V_INPUT_1]], %[[V_FILTER_0]], %[[V_OUTPUT_1]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
/// w == 1, kw == 1
//      CHECK:   %[[CONTRACT_2:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     %[[V_INPUT_2]], %[[V_FILTER_1]], %[[CONTRACT_0]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
/// w == 1, kw == 1
//      CHECK:   %[[CONTRACT_3:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
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

func.func @conv1d_nwc_4x2x8_memref(%input: memref<4x6x3xf32>, %filter: memref<2x3x8xf32>, %output: memref<4x2x8xf32>) {
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

//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 2, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x2x3xf32>
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 2, 0], sizes = [4, 2, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x2x3xf32>

//      CHECK:  %[[V_FILTER_0:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<3x8xf32> from vector<2x3x8xf32>
//      CHECK:  %[[V_FILTER_1:.+]] = vector.extract %[[V_FILTER_R]][1] : vector<3x8xf32> from vector<2x3x8xf32>

/// w == 0, kw == 0
//      CHECK:   %[[CONTRACT_0:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     %[[V_INPUT_0]], %[[V_FILTER_0]], %[[V_OUTPUT_R]]
// CHECK-SAME:     : vector<4x2x3xf32>, vector<3x8xf32> into vector<4x2x8xf32>
/// w == 0, kw == 1
//      CHECK:   %[[CONTRACT_1:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     %[[V_INPUT_1]], %[[V_FILTER_1]], %[[CONTRACT_0]]
// CHECK-SAME:     : vector<4x2x3xf32>, vector<3x8xf32> into vector<4x2x8xf32>

// Write the result back in one shot.
//      CHECK:   vector.transfer_write %[[CONTRACT_1]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

// -----

func.func @conv1d_ncw_4x8x2_memref(%input: memref<4x3x6xf32>, %filter: memref<8x3x1xf32>, %output: memref<4x8x2xf32>) {
  linalg.conv_1d_ncw_fcw
    {dilations = dense<1> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x3x6xf32>, memref<8x3x1xf32>)
    outs(%output : memref<4x8x2xf32>)
  return
}

// CHECK: #[[INPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[FILTER_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK: #[[OUTPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

//      CHECK: func @conv1d_ncw_4x8x2_memref
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x3x6xf32>, %[[FILTER:.+]]: memref<8x3x1xf32>, %[[OUTPUT:.+]]: memref<4x8x2xf32>)

//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

/// Read the whole data in one shot.
//  CHECK-DAG:   %[[V_NWC_INPUT_R:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//  CHECK-DAG:  %[[V_NWC_FILTER_R:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//  CHECK-DAG:  %[[V_NWC_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]

/// Transpose result to nwc format.
//  CHECK-DAG:   %[[V_INPUT_R:.+]] = vector.transpose %[[V_NWC_INPUT_R]], [0, 2, 1]
//  CHECK-DAG:  %[[V_FILTER_R:.+]] = vector.transpose %[[V_NWC_FILTER_R]], [2, 1, 0]
//  CHECK-DAG:  %[[V_OUTPUT_R:.+]] = vector.transpose %[[V_NWC_OUTPUT_R]], [0, 2, 1]

//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x1x3xf32>
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 3, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x1x3xf32>

//      CHECK:    %[[V_FILTER:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<3x8xf32> from vector<1x3x8xf32>

//      CHECK:  %[[V_OUTPUT_0:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 1, 8], strides = [1, 1, 1]} : vector<4x2x8xf32> to vector<4x1x8xf32>
//      CHECK:  %[[V_OUTPUT_1:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 1, 0], sizes = [4, 1, 8], strides = [1, 1, 1]} : vector<4x2x8xf32> to vector<4x1x8xf32>

/// w == 0, kw == 0
//      CHECK:   %[[CONTRACT_0:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:       kind = #vector.kind<add>
// CHECK-SAME:     %[[V_INPUT_0]], %[[V_FILTER]], %[[V_OUTPUT_0]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>

/// w == 1, kw == 0
//      CHECK:   %[[CONTRACT_1:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:       kind = #vector.kind<add>
// CHECK-SAME:     %[[V_INPUT_1]], %[[V_FILTER]], %[[V_OUTPUT_1]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>

/// w == 0, kw == 0
//      CHECK:   %[[RES_0:.+]] = vector.insert_strided_slice %[[CONTRACT_0]], %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<4x1x8xf32> into vector<4x2x8xf32>
/// w == 1, kw == 0
//      CHECK:   %[[RES_1:.+]] = vector.insert_strided_slice %[[CONTRACT_1]], %[[RES_0]]
// CHECK-SAME:     {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<4x1x8xf32> into vector<4x2x8xf32>

/// Transpose result to ncw format.
//  CHECK:  %[[RES_2:.+]] = vector.transpose %[[RES_1]], [0, 2, 1]

// Write the result back in one shot.
//      CHECK:   vector.transfer_write %[[RES_2]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

// -----

// This test is same as above but for i1 type with the only difference being that
// the combining kind for `vector.contract` is `OR`.
func.func @conv1d_ncw_4x8x2_memref_i1(%input: memref<4x3x6xi1>, %filter: memref<8x3x1xi1>, %output: memref<4x8x2xi1>) {
  linalg.conv_1d_ncw_fcw
    {dilations = dense<1> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x3x6xi1>, memref<8x3x1xi1>)
    outs(%output : memref<4x8x2xi1>)
  return
}

// CHECK: #[[INPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[FILTER_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK: #[[OUTPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

//      CHECK: func @conv1d_ncw_4x8x2_memref_i1
/// w == 0, kw == 0
//      CHECK:   vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:       kind = #vector.kind<or>
// CHECK-SAME:       : vector<4x1x3xi1>, vector<3x8xi1> into vector<4x1x8xi1>

/// w == 1, kw == 0
//      CHECK:   vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:       kind = #vector.kind<or>
// CHECK-SAME:       : vector<4x1x3xi1>, vector<3x8xi1> into vector<4x1x8xi1>

// -----

func.func @conv1d_ncw_4x8x2_memref(%input: memref<4x3x6xf32>, %filter: memref<8x3x2xf32>, %output: memref<4x8x2xf32>) {
  linalg.conv_1d_ncw_fcw
    {dilations = dense<2> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x3x6xf32>, memref<8x3x2xf32>)
    outs(%output : memref<4x8x2xf32>)
  return
}

// CHECK: #[[INPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[FILTER_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK: #[[OUTPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

//      CHECK: func @conv1d_ncw_4x8x2_memref
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x3x6xf32>, %[[FILTER:.+]]: memref<8x3x2xf32>, %[[OUTPUT:.+]]: memref<4x8x2xf32>)

//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

/// Read the whole data in one shot.
//  CHECK-DAG:   %[[V_NWC_INPUT_R:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//  CHECK-DAG:   %[[V_NWC_FILTER_R:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//  CHECK-DAG:   %[[V_NWC_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]

/// Transpose result to nwc format.
//  CHECK-DAG:   %[[V_INPUT_R:.+]] = vector.transpose %[[V_NWC_INPUT_R]], [0, 2, 1]
//  CHECK-DAG:  %[[V_FILTER_R:.+]] = vector.transpose %[[V_NWC_FILTER_R]], [2, 1, 0]
//  CHECK-DAG:  %[[V_OUTPUT_R:.+]] = vector.transpose %[[V_NWC_OUTPUT_R]], [0, 2, 1]

//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 3, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
//      CHECK:   %[[V_INPUT_2:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 2, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
//      CHECK:   %[[V_INPUT_3:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 5, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>

//      CHECK:  %[[V_FILTER_0:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<3x8xf32> from vector<2x3x8xf32>
//      CHECK:  %[[V_FILTER_1:.+]] = vector.extract %[[V_FILTER_R]][1] : vector<3x8xf32> from vector<2x3x8xf32>

//      CHECK:  %[[V_OUTPUT_0:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 1, 8], strides = [1, 1, 1]} : vector<4x2x8xf32> to vector<4x1x8xf32>
//      CHECK:  %[[V_OUTPUT_1:.+]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 1, 0], sizes = [4, 1, 8], strides = [1, 1, 1]} : vector<4x2x8xf32> to vector<4x1x8xf32>

/// w == 0, kw == 0
//      CHECK:   %[[CONTRACT_0:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     %[[V_INPUT_0]], %[[V_FILTER_0]], %[[V_OUTPUT_0]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
/// w == 1, kw == 0
//      CHECK:   %[[CONTRACT_1:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     %[[V_INPUT_1]], %[[V_FILTER_0]], %[[V_OUTPUT_1]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
/// w == 1, kw == 1
//      CHECK:   %[[CONTRACT_2:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     %[[V_INPUT_2]], %[[V_FILTER_1]], %[[CONTRACT_0]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>
/// w == 1, kw == 1
//      CHECK:   %[[CONTRACT_3:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     %[[V_INPUT_3]], %[[V_FILTER_1]], %[[CONTRACT_1]]
// CHECK-SAME:     : vector<4x1x3xf32>, vector<3x8xf32> into vector<4x1x8xf32>

/// w == 0, kw == 0
//      CHECK:   %[[RES_0:.+]] = vector.insert_strided_slice %[[CONTRACT_2]], %[[V_OUTPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<4x1x8xf32> into vector<4x2x8xf32>
/// w == 1, kw == 0
//      CHECK:   %[[RES_1:.+]] = vector.insert_strided_slice %[[CONTRACT_3]], %[[RES_0]]
// CHECK-SAME:     {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<4x1x8xf32> into vector<4x2x8xf32>

/// Transpose result to ncw format.
//  CHECK:  %[[RES_2:.+]] = vector.transpose %[[RES_1]], [0, 2, 1]

// Write the result back in one shot.
//      CHECK:   vector.transfer_write %[[RES_2]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

// -----

func.func @conv1d_ncw_4x8x2_memref(%input: memref<4x3x6xf32>, %filter: memref<8x3x2xf32>, %output: memref<4x8x2xf32>) {
  linalg.conv_1d_ncw_fcw
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<4x3x6xf32>, memref<8x3x2xf32>)
    outs(%output : memref<4x8x2xf32>)
  return
}

// CHECK: #[[INPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[FILTER_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK: #[[OUTPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

//      CHECK: func @conv1d_ncw_4x8x2_memref
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x3x6xf32>, %[[FILTER:.+]]: memref<8x3x2xf32>, %[[OUTPUT:.+]]: memref<4x8x2xf32>)

//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

/// Read the whole data in one shot.
//  CHECK-DAG:   %[[V_NWC_INPUT_R:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//  CHECK-DAG:  %[[V_NWC_FILTER_R:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]
//  CHECK-DAG:  %[[V_NWC_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]], %[[F0]]

/// Transpose result to nwc format.
//  CHECK-DAG:   %[[V_INPUT_R:.+]] = vector.transpose %[[V_NWC_INPUT_R]], [0, 2, 1]
//  CHECK-DAG:  %[[V_FILTER_R:.+]] = vector.transpose %[[V_NWC_FILTER_R]], [2, 1, 0]
//  CHECK-DAG:  %[[V_OUTPUT_R:.+]] = vector.transpose %[[V_NWC_OUTPUT_R]], [0, 2, 1]

//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [4, 2, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x2x3xf32>
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 2, 0], sizes = [4, 2, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x2x3xf32>

//      CHECK:  %[[V_FILTER_0:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<3x8xf32> from vector<2x3x8xf32>
//      CHECK:  %[[V_FILTER_1:.+]] = vector.extract %[[V_FILTER_R]][1] : vector<3x8xf32> from vector<2x3x8xf32>

/// w == 0, kw == 0
//      CHECK:   %[[CONTRACT_0:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     %[[V_INPUT_0]], %[[V_FILTER_0]], %[[V_OUTPUT_R]]
// CHECK-SAME:     : vector<4x2x3xf32>, vector<3x8xf32> into vector<4x2x8xf32>
/// w == 0, kw == 1
//      CHECK:   %[[CONTRACT_1:.+]] = vector.contract {
// CHECK-SAME:       indexing_maps = [#[[INPUT_MAP]], #[[FILTER_MAP]], #[[OUTPUT_MAP]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     %[[V_INPUT_1]], %[[V_FILTER_1]], %[[CONTRACT_0]]
// CHECK-SAME:     : vector<4x2x3xf32>, vector<3x8xf32> into vector<4x2x8xf32>

/// Transpose result to ncw format.
//  CHECK:  %[[RES:.+]] = vector.transpose %[[CONTRACT_1]], [0, 2, 1]

// Write the result back in one shot.
//      CHECK:   vector.transfer_write %[[RES]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]


// -----

func.func @conv1d_8_tensor(%input: tensor<11xf32>, %filter: tensor<4xf32>, %output: tensor<8xf32>) -> tensor<8xf32> {
  %0 = linalg.conv_1d ins(%input, %filter : tensor<11xf32>, tensor<4xf32>)
                     outs(%output : tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

//      CHECK: func @conv1d_8_tensor
// CHECK-SAME: (%[[INPUT:.+]]: tensor<11xf32>, %[[FILTER:.+]]: tensor<4xf32>, %[[OUTPUT:.+]]: tensor<8xf32>)

//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

/// Read the whole data in one shot.
//  CHECK-DAG:   %[[V_INPUT_R:.+]] = vector.transfer_read %[[INPUT]][%[[C0]]], %[[F0]]
//  CHECK-DAG:  %[[V_FILTER_R:.+]] = vector.transfer_read %[[FILTER]][%[[C0]]], %[[F0]]
//  CHECK-DAG:  %[[V_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]]], %[[F0]]

//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0], sizes = [8], strides = [1]} : vector<11xf32> to vector<8xf32>
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [1], sizes = [8], strides = [1]} : vector<11xf32> to vector<8xf32>
//      CHECK:   %[[V_INPUT_2:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [2], sizes = [8], strides = [1]} : vector<11xf32> to vector<8xf32>
//      CHECK:   %[[V_INPUT_3:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [3], sizes = [8], strides = [1]} : vector<11xf32> to vector<8xf32>

//      CHECK:  %[[V_FILTER_0:.+]] = vector.extract %[[V_FILTER_R]][0] : f32 from vector<4xf32>
//      CHECK:  %[[V_FILTER_1:.+]] = vector.extract %[[V_FILTER_R]][1] : f32 from vector<4xf32>
//      CHECK:  %[[V_FILTER_2:.+]] = vector.extract %[[V_FILTER_R]][2] : f32 from vector<4xf32>
//      CHECK:  %[[V_FILTER_3:.+]] = vector.extract %[[V_FILTER_R]][3] : f32 from vector<4xf32>

/// w == 0, kw == 0
//      CHECK:   %[[RES_0:.+]] = vector.outerproduct
// CHECK-SAME:     %[[V_INPUT_0]], %[[V_FILTER_0]], %[[V_OUTPUT_R]] {kind = #vector.kind<add>}
// CHECK-SAME:     : vector<8xf32>, f32
/// w == 1, kw == 1
//      CHECK:   %[[RES_1:.+]] = vector.outerproduct
// CHECK-SAME:     %[[V_INPUT_1]], %[[V_FILTER_1]], %[[RES_0]] {kind = #vector.kind<add>}
// CHECK-SAME:     : vector<8xf32>, f32
/// w == 2, kw == 2
//      CHECK:   %[[RES_2:.+]] = vector.outerproduct
// CHECK-SAME:     %[[V_INPUT_2]], %[[V_FILTER_2]], %[[RES_1]] {kind = #vector.kind<add>}
// CHECK-SAME:     : vector<8xf32>, f32
/// w == 3, kw == 3
//      CHECK:   %[[RES_3:.+]] = vector.outerproduct
// CHECK-SAME:     %[[V_INPUT_3]], %[[V_FILTER_3]], %[[RES_2]] {kind = #vector.kind<add>}
// CHECK-SAME:     : vector<8xf32>, f32

// Write the result back in one shot.
//      CHECK:   vector.transfer_write %[[RES_3]], %[[OUTPUT]][%[[C0]]]

// -----

func.func @depthwise_conv1d_nwc_wc_3x5x4xf32_memref(%input: memref<3x5x4xf32>, %filter: memref<2x4xf32>, %output: memref<3x2x4xf32>) {
  linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<3x5x4xf32>, memref<2x4xf32>)
    outs(%output : memref<3x2x4xf32>)
  return
}

//       CHECK: func @depthwise_conv1d_nwc_wc_3x5x4xf32_memref
//  CHECK-SAME:   (%[[INPUT:[0-9a-z]+]]: memref<3x5x4xf32>, %[[FILTER:[0-9a-z]+]]: memref<2x4xf32>, %[[OUTPUT:[0-9a-z]+]]: memref<3x2x4xf32>)

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

/// Read the whole data in one shot.
//      CHECK:   %[[V_INPUT_R:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]]
//      CHECK:  %[[V_FILTER_R:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]]]
//      CHECK:  %[[V_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x4xf32> to vector<3x2x4xf32>
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 2, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x4xf32> to vector<3x2x4xf32>

//      CHECK:  %[[V_FILTER_0:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<4xf32> from vector<2x4xf32>
//      CHECK:  %[[V_FILTER_1:.+]] = vector.extract %[[V_FILTER_R]][1] : vector<4xf32> from vector<2x4xf32>

/// w == 0, kw = 0
//      CHECK:  %[[B_FILTER_0:.*]] = vector.broadcast %[[V_FILTER_0]] : vector<4xf32> to vector<3x2x4xf32>
//      CHECK:  %[[FMA_0:.*]] = vector.fma %[[V_INPUT_0]], %[[B_FILTER_0]], %[[V_OUTPUT_R]] : vector<3x2x4xf32>

/// w == 0, kw = 1
//      CHECK:  %[[B_FILTER_1:.*]] = vector.broadcast %[[V_FILTER_1]] : vector<4xf32> to vector<3x2x4xf32>
//      CHECK:  %[[FMA_1:.*]] = vector.fma %[[V_INPUT_1]], %[[B_FILTER_1]], %[[FMA_0]] : vector<3x2x4xf32>

// Write the result back in one shot.
//      CHECK:   vector.transfer_write %[[FMA_1]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]


// -----

func.func @depthwise_conv1d_nwc_wc_3x5x4xi8_memref(%input: memref<3x5x4xi8>, %filter: memref<2x4xi8>, %output: memref<3x2x4xi32>) {
  linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<3x5x4xi8>, memref<2x4xi8>)
    outs(%output : memref<3x2x4xi32>)
  return
}

//       CHECK: func @depthwise_conv1d_nwc_wc_3x5x4xi8_memref
//  CHECK-SAME:   (%[[INPUT:[0-9a-z]+]]: memref<3x5x4xi8>, %[[FILTER:[0-9a-z]+]]: memref<2x4xi8>, %[[OUTPUT:[0-9a-z]+]]: memref<3x2x4xi32>)

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index

/// Read the whole data in one shot.
//      CHECK:   %[[V_INPUT_R:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]]
//      CHECK:  %[[V_FILTER_R:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]]]
//      CHECK:  %[[V_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x4xi8> to vector<3x2x4xi8>
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 2, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x4xi8> to vector<3x2x4xi8>

//      CHECK:  %[[V_FILTER_0:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<4xi8> from vector<2x4xi8>
//      CHECK:  %[[V_FILTER_1:.+]] = vector.extract %[[V_FILTER_R]][1] : vector<4xi8> from vector<2x4xi8>

/// w == 0, kw =
//      CHECK:  %[[EXT_INPUT_0:.*]] = arith.extsi %[[V_INPUT_0]] : vector<3x2x4xi8> to vector<3x2x4xi32>
//      CHECK:  %[[B_FILTER_0:.*]] = vector.broadcast %[[V_FILTER_0]] : vector<4xi8> to vector<3x2x4xi8>
//      CHECK:  %[[EXT_FILTER_0:.*]] = arith.extsi %[[B_FILTER_0]] : vector<3x2x4xi8> to vector<3x2x4xi32>
//      CHECK:  %[[MUL_0:.*]] = arith.muli %[[EXT_INPUT_0]], %[[EXT_FILTER_0]] : vector<3x2x4xi32>
//      CHECK:  %[[ADD_0:.*]] = arith.addi %[[MUL_0]], %[[V_OUTPUT_R]] : vector<3x2x4xi32>

/// w == 0, kw = 1
//      CHECK:  %[[EXT_INPUT_1:.*]] = arith.extsi %[[V_INPUT_1]] : vector<3x2x4xi8> to vector<3x2x4xi32>
//      CHECK:  %[[B_FILTER_1:.*]] = vector.broadcast %[[V_FILTER_1]] : vector<4xi8> to vector<3x2x4xi8>
//      CHECK:  %[[EXT_FILTER_1:.*]] = arith.extsi %[[B_FILTER_1]] : vector<3x2x4xi8> to vector<3x2x4xi32>
//      CHECK:  %[[MUL_1:.*]] = arith.muli %[[EXT_INPUT_1]], %[[EXT_FILTER_1]] : vector<3x2x4xi32>
//      CHECK:  %[[ADD_1:.*]] = arith.addi %[[MUL_1]], %[[ADD_0]] : vector<3x2x4xi32>

// Write the result back in one shot.
//      CHECK:   vector.transfer_write %[[ADD_1]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

// -----

func.func @conv_1d_nwc_wcf_mixed_type_memref(%input: memref<1x2x3xf16>, %filter: memref<1x3x2xf16>, %output: memref<1x2x2xf32>) {
  linalg.conv_1d_nwc_wcf
  {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>}
   ins(%input, %filter : memref<1x2x3xf16>, memref<1x3x2xf16>)
   outs(%output : memref<1x2x2xf32>)
  return
}

//       CHECK: func @conv_1d_nwc_wcf_mixed_type_memref
//  CHECK-SAME:   (%[[INPUT:[0-9a-z]+]]: memref<1x2x3xf16>, %[[FILTER:[0-9a-z]+]]: memref<1x3x2xf16>, %[[OUTPUT:[0-9a-z]+]]: memref<1x2x2xf32>)

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

/// Read the whole data in one shot.
//      CHECK:   %[[V_INPUT_R:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]]
//      CHECK:   %[[V_FILTER_R:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]], %[[C0]]]
//      CHECK:   %[[V_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]
//      CHECK:   %[[V_FILTER_1:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<3x2xf16> from vector<1x3x2xf16>
//      CHECK:   %[[CONT:.*]] = vector.contract
//        {{.*}} %[[V_INPUT_R]], %[[V_FILTER_1]], %[[V_OUTPUT_R]] : vector<1x2x3xf16>, vector<3x2xf16> into vector<1x2x2xf32>
//      CHECK:   vector.transfer_write %[[CONT]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

// -----

func.func @conv_1d_nwc_wcf_mixed_int_fp_memref(%input: memref<1x2x3xi8>, %filter: memref<1x3x2xi8>, %output: memref<1x2x2xf32>) {
  linalg.conv_1d_nwc_wcf
  {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>}
   ins(%input, %filter : memref<1x2x3xi8>, memref<1x3x2xi8>)
   outs(%output : memref<1x2x2xf32>)
  return
}


// CHECK-LABEL: func @conv_1d_nwc_wcf_mixed_int_fp_memref
// CHECK-SAME:   (%[[INPUT:[0-9a-z]+]]: memref<1x2x3xi8>, %[[FILTER:[0-9a-z]+]]: memref<1x3x2xi8>, %[[OUTPUT:[0-9a-z]+]]: memref<1x2x2xf32>)
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[I0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i8
// CHECK: %[[READ0:.+]] = vector.transfer_read %arg0[%[[I0]], %[[I0]], %[[I0]]], %[[C0]]
// CHECK: %[[READ1:.+]] = vector.transfer_read %arg1[%[[I0]], %[[I0]], %[[I0]]], %[[C0]]
// CHECK: %[[READ2:.+]] = vector.transfer_read %arg2[%[[I0]], %[[I0]], %[[I0]]], %[[CST]]
// CHECK: %[[EXT:.+]] = vector.extract %[[READ1]][0] : vector<3x2xi8> from vector<1x3x2xi8>
// CHECK: %[[CAST0:.+]] = arith.sitofp %[[READ0]] : vector<1x2x3xi8> to vector<1x2x3xf32>
// CHECK: %[[CAST1:.+]] = arith.sitofp %[[EXT]] : vector<3x2xi8> to vector<3x2xf32>
// CHECK: %[[CONTRACT:.+]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[CAST0]], %[[CAST1]], %[[READ2]]
// CHECK: vector.transfer_write %[[CONTRACT]], %arg2[%[[I0]], %[[I0]], %[[I0]]]

// -----

func.func @pooling_nwc_sum_memref_1_2_1_3(%input: memref<4x4x3xf32>, %filter: memref<1xf32>, %output: memref<4x2x3xf32>) {
  linalg.pooling_nwc_sum
    {dilations = dense<1> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x4x3xf32>, memref<1xf32>)
    outs(%output : memref<4x2x3xf32>)
  return
}

// CHECK-LABEL: func.func @pooling_nwc_sum_memref_1_2_1_3
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x4x3xf32>, %[[FILTER:.+]]: memref<1xf32>, %[[OUTPUT:.+]]: memref<4x2x3xf32>)
// CHECK-DAG: %[[Vc0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[Vcst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[V0:.+]] = vector.transfer_read %[[INPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<4x4x3xf32>, vector<4x4x3xf32>
// CHECK: %[[V1:.+]] = vector.transfer_read %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<4x2x3xf32>, vector<4x2x3xf32>
// CHECK: %[[V2:.+]] = vector.extract_strided_slice %[[V0]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V3:.+]] = vector.extract_strided_slice %[[V0]] {offsets = [0, 3, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V4:.+]] = vector.extract_strided_slice %[[V1]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V5:.+]] = vector.extract_strided_slice %[[V1]] {offsets = [0, 1, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V6:.+]] = arith.addf %[[V2]], %[[V4]] : vector<4x1x3xf32>
// CHECK: %[[V7:.+]] = arith.addf %[[V3]], %[[V5]] : vector<4x1x3xf32>
// CHECK: %[[V8:.+]] = vector.insert_strided_slice %[[V6]], %[[V1]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<4x1x3xf32> into vector<4x2x3xf32>
// CHECK: %[[V9:.+]] = vector.insert_strided_slice %[[V7]], %[[V8]] {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<4x1x3xf32> into vector<4x2x3xf32>
// CHECK: vector.transfer_write %[[V9]], %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]] {in_bounds = [true, true, true]} : vector<4x2x3xf32>, memref<4x2x3xf32>

// -----

func.func @pooling_nwc_max_memref_1_2_1_3(%input: memref<4x4x3xf32>, %filter: memref<1xf32>, %output: memref<4x2x3xf32>) {
  linalg.pooling_nwc_max
    {dilations = dense<1> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x4x3xf32>, memref<1xf32>)
    outs(%output : memref<4x2x3xf32>)
  return
}

// CHECK-LABEL: func.func @pooling_nwc_max_memref_1_2_1_3
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x4x3xf32>, %[[FILTER:.+]]: memref<1xf32>, %[[OUTPUT:.+]]: memref<4x2x3xf32>)
// CHECK-DAG: %[[Vc0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[Vcst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[V0:.+]] = vector.transfer_read %[[INPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<4x4x3xf32>, vector<4x4x3xf32>
// CHECK: %[[V1:.+]] = vector.transfer_read %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<4x2x3xf32>, vector<4x2x3xf32>
// CHECK: %[[V2:.+]] = vector.extract_strided_slice %[[V0]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V3:.+]] = vector.extract_strided_slice %[[V0]] {offsets = [0, 3, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V4:.+]] = vector.extract_strided_slice %[[V1]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V5:.+]] = vector.extract_strided_slice %[[V1]] {offsets = [0, 1, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V6:.+]] = arith.maximumf %[[V2]], %[[V4]] : vector<4x1x3xf32>
// CHECK: %[[V7:.+]] = arith.maximumf %[[V3]], %[[V5]] : vector<4x1x3xf32>
// CHECK: %[[V8:.+]] = vector.insert_strided_slice %[[V6]], %[[V1]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<4x1x3xf32> into vector<4x2x3xf32>
// CHECK: %[[V9:.+]] = vector.insert_strided_slice %[[V7]], %[[V8]] {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<4x1x3xf32> into vector<4x2x3xf32>
// CHECK: vector.transfer_write %[[V9]], %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]] {in_bounds = [true, true, true]} : vector<4x2x3xf32>, memref<4x2x3xf32>

// -----

// The i8i8i32 case is similar to f32 case, so checking one case is enough for
// test coverage.
func.func @pooling_nwc_sum_i8i8i32_memref_1_2_1_3(%input: memref<4x4x3xi8>, %filter: memref<1xi8>, %output: memref<4x2x3xi32>) {
  linalg.pooling_nwc_sum
    {dilations = dense<1> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x4x3xi8>, memref<1xi8>)
    outs(%output : memref<4x2x3xi32>)
  return
}

// CHECK-LABEL: func.func @pooling_nwc_sum_i8i8i32_memref_1_2_1_3
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x4x3xi8>, %[[FILTER:.+]]: memref<1xi8>, %[[OUTPUT:.+]]: memref<4x2x3xi32>)
// CHECK-DAG: %[[Vc0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[Vc0_i8:.+]] = arith.constant 0 : i8
// CHECK-DAG: %[[Vc0_i32:.+]] = arith.constant 0 : i32
// CHECK: %[[V0:.+]] = vector.transfer_read %[[INPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vc0_i8]] {in_bounds = [true, true, true]} : memref<4x4x3xi8>, vector<4x4x3xi8>
// CHECK: %[[V1:.+]] = vector.transfer_read %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vc0_i32]] {in_bounds = [true, true, true]} : memref<4x2x3xi32>, vector<4x2x3xi32>
// CHECK: %[[V2:.+]] = vector.extract_strided_slice %[[V0]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xi8> to vector<4x1x3xi8>
// CHECK: %[[V3:.+]] = vector.extract_strided_slice %[[V0]] {offsets = [0, 3, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xi8> to vector<4x1x3xi8>
// CHECK: %[[V4:.+]] = vector.extract_strided_slice %[[V1]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xi32> to vector<4x1x3xi32>
// CHECK: %[[V5:.+]] = vector.extract_strided_slice %[[V1]] {offsets = [0, 1, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xi32> to vector<4x1x3xi32>
// CHECK: %[[V6:.+]] = arith.extsi %[[V2]] : vector<4x1x3xi8> to vector<4x1x3xi32>
// CHECK: %[[V7:.+]] = arith.addi %[[V6]], %[[V4]] : vector<4x1x3xi32>
// CHECK: %[[V8:.+]] = arith.extsi %[[V3]] : vector<4x1x3xi8> to vector<4x1x3xi32>
// CHECK: %[[V9:.+]] = arith.addi %[[V8]], %[[V5]] : vector<4x1x3xi32>
// CHECK: %[[V10:.+]] = vector.insert_strided_slice %[[V7]], %[[V1]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<4x1x3xi32> into vector<4x2x3xi32>
// CHECK: %[[V11:.+]] = vector.insert_strided_slice %[[V9]], %[[V10]] {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<4x1x3xi32> into vector<4x2x3xi32>
// CHECK: vector.transfer_write %[[V11]], %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]] {in_bounds = [true, true, true]} : vector<4x2x3xi32>, memref<4x2x3xi32>
// CHECK: return

// -----

// The i8i8i32 case is similar to f32 case, so checking one case is enough for
// test coverage.
func.func @pooling_nwc_max_i8i8i32_memref_1_2_1_3(%input: memref<4x4x3xi8>, %filter: memref<1xi8>, %output: memref<4x2x3xi32>) {
  linalg.pooling_nwc_max
    {dilations = dense<1> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x4x3xi8>, memref<1xi8>)
    outs(%output : memref<4x2x3xi32>)
  return
}

// CHECK-LABEL: func.func @pooling_nwc_max_i8i8i32_memref_1_2_1_3
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x4x3xi8>, %[[FILTER:.+]]: memref<1xi8>, %[[OUTPUT:.+]]: memref<4x2x3xi32>)
// CHECK-DAG: %[[Vc0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[Vc0_i8:.+]] = arith.constant 0 : i8
// CHECK-DAG: %[[Vc0_i32:.+]] = arith.constant 0 : i32
// CHECK: %[[V0:.+]] = vector.transfer_read %[[INPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vc0_i8]] {in_bounds = [true, true, true]} : memref<4x4x3xi8>, vector<4x4x3xi8>
// CHECK: %[[V1:.+]] = vector.transfer_read %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vc0_i32]] {in_bounds = [true, true, true]} : memref<4x2x3xi32>, vector<4x2x3xi32>
// CHECK: %[[V2:.+]] = vector.extract_strided_slice %[[V0]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xi8> to vector<4x1x3xi8>
// CHECK: %[[V3:.+]] = vector.extract_strided_slice %[[V0]] {offsets = [0, 3, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xi8> to vector<4x1x3xi8>
// CHECK: %[[V4:.+]] = vector.extract_strided_slice %[[V1]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xi32> to vector<4x1x3xi32>
// CHECK: %[[V5:.+]] = vector.extract_strided_slice %[[V1]] {offsets = [0, 1, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xi32> to vector<4x1x3xi32>
// CHECK: %[[V6:.+]] = arith.extsi %[[V2]] : vector<4x1x3xi8> to vector<4x1x3xi32>
// CHECK: %[[V7:.+]] = arith.maxsi %[[V6]], %[[V4]] : vector<4x1x3xi32>
// CHECK: %[[V8:.+]] = arith.extsi %[[V3]] : vector<4x1x3xi8> to vector<4x1x3xi32>
// CHECK: %[[V9:.+]] = arith.maxsi %[[V8]], %[[V5]] : vector<4x1x3xi32>
// CHECK: %[[V10:.+]] = vector.insert_strided_slice %[[V7]], %[[V1]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<4x1x3xi32> into vector<4x2x3xi32>
// CHECK: %[[V11:.+]] = vector.insert_strided_slice %[[V9]], %[[V10]] {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<4x1x3xi32> into vector<4x2x3xi32>
// CHECK: vector.transfer_write %[[V11]], %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]] {in_bounds = [true, true, true]} : vector<4x2x3xi32>, memref<4x2x3xi32>
// CHECK: return

// -----

func.func @pooling_nwc_sum_memref_2_2_2_3(%input: memref<4x6x3xf32>, %filter: memref<2xf32>, %output: memref<4x2x3xf32>) {
  linalg.pooling_nwc_sum
    {dilations = dense<2> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x6x3xf32>, memref<2xf32>)
    outs(%output : memref<4x2x3xf32>)
  return
}

// CHECK-LABEL: func.func @pooling_nwc_sum_memref_2_2_2_3
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x6x3xf32>, %[[FILTER:.+]]: memref<2xf32>, %[[OUTPUT:.+]]: memref<4x2x3xf32>)
// CHECK-DAG: %[[Vc0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[Vcst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[V0:.+]] = vector.transfer_read %[[INPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<4x6x3xf32>, vector<4x6x3xf32>
// CHECK: %[[V1:.+]] = vector.transfer_read %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<4x2x3xf32>, vector<4x2x3xf32>
// CHECK: %[[V2:.+]] = vector.extract_strided_slice %[[V0]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V3:.+]] = vector.extract_strided_slice %[[V0]] {offsets = [0, 3, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V4:.+]] = vector.extract_strided_slice %[[V0]] {offsets = [0, 2, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V5:.+]] = vector.extract_strided_slice %[[V0]] {offsets = [0, 5, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V6:.+]] = vector.extract_strided_slice %[[V1]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V7:.+]] = vector.extract_strided_slice %[[V1]] {offsets = [0, 1, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V8:.+]] = arith.addf %[[V2]], %[[V6]] : vector<4x1x3xf32>
// CHECK: %[[V9:.+]] = arith.addf %[[V3]], %[[V7]] : vector<4x1x3xf32>
// CHECK: %[[V10:.+]] = arith.addf %[[V4]], %[[V8]] : vector<4x1x3xf32>
// CHECK: %[[V11:.+]] = arith.addf %[[V5]], %[[V9]] : vector<4x1x3xf32>
// CHECK: %[[V12:.+]] = vector.insert_strided_slice %[[V10]], %[[V1]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<4x1x3xf32> into vector<4x2x3xf32>
// CHECK: %[[V13:.+]] = vector.insert_strided_slice %[[V11]], %[[V12]] {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<4x1x3xf32> into vector<4x2x3xf32>
// CHECK: vector.transfer_write %[[V13:.+]], %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]] {in_bounds = [true, true, true]} : vector<4x2x3xf32>, memref<4x2x3xf32>


// -----

func.func @pooling_ncw_sum_memref_1_2_1_3(%input: memref<4x3x4xf32>, %filter: memref<1xf32>, %output: memref<4x3x2xf32>) {
  linalg.pooling_ncw_sum
    {dilations = dense<1> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x3x4xf32>, memref<1xf32>)
    outs(%output : memref<4x3x2xf32>)
  return
}

// CHECK-LABEL: func.func @pooling_ncw_sum_memref_1_2_1_3
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x3x4xf32>, %[[FILTER:.+]]: memref<1xf32>, %[[OUTPUT:.+]]: memref<4x3x2xf32>)
// CHECK-DAG: %[[Vc0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[Vcst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[V0:.+]] = vector.transfer_read %[[INPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<4x3x4xf32>, vector<4x3x4xf32>
// CHECK: %[[V1:.+]] = vector.transfer_read %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<4x3x2xf32>, vector<4x3x2xf32>
// CHECK: %[[V2:.+]] = vector.transpose %[[V0]], [0, 2, 1] : vector<4x3x4xf32> to vector<4x4x3xf32>
// CHECK: %[[V3:.+]] = vector.transpose %[[V1]], [0, 2, 1] : vector<4x3x2xf32> to vector<4x2x3xf32>
// CHECK: %[[V4:.+]] = vector.extract_strided_slice %[[V2]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V5:.+]] = vector.extract_strided_slice %[[V2]] {offsets = [0, 3, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V6:.+]] = vector.extract_strided_slice %[[V3]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V7:.+]] = vector.extract_strided_slice %[[V3]] {offsets = [0, 1, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V8:.+]] = arith.addf %[[V4]], %[[V6]] : vector<4x1x3xf32>
// CHECK: %[[V9:.+]] = arith.addf %[[V5]], %[[V7]] : vector<4x1x3xf32>
// CHECK: %[[V10:.+]] = vector.insert_strided_slice %[[V8]], %[[V3]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<4x1x3xf32> into vector<4x2x3xf32>
// CHECK: %[[V11:.+]] = vector.insert_strided_slice %[[V9]], %[[V10]] {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<4x1x3xf32> into vector<4x2x3xf32>
// CHECK: %[[V12:.+]] = vector.transpose %[[V11]], [0, 2, 1] : vector<4x2x3xf32> to vector<4x3x2xf32>
// CHECK: vector.transfer_write %[[V12:.+]], %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]] {in_bounds = [true, true, true]} : vector<4x3x2xf32>, memref<4x3x2xf32>


// -----

func.func @pooling_nwc_sum_mixed_type_memref_1_2_1_1(%input: memref<1x2x3xf16>, %filter: memref<1xf16>, %output: memref<1x2x3xf32>) {
  linalg.pooling_nwc_sum
  {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>}
   ins(%input, %filter : memref<1x2x3xf16>, memref<1xf16>)
   outs(%output : memref<1x2x3xf32>)
  return
}

// CHECK-LABEL: func.func @pooling_nwc_sum_mixed_type_memref_1_2_1_1
// CHECK-SAME: (%[[INPUT:.+]]: memref<1x2x3xf16>, %[[FILTER:.+]]: memref<1xf16>, %[[OUTPUT:.+]]: memref<1x2x3xf32>)
// CHECK-DAG: %[[Vc0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[Vcst:.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG: %[[Vcst_0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[V0:.+]] = vector.transfer_read %[[INPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<1x2x3xf16>, vector<1x2x3xf16>
// CHECK: %[[V1:.+]] = vector.transfer_read %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst_0]] {in_bounds = [true, true, true]} : memref<1x2x3xf32>, vector<1x2x3xf32>
// CHECK: %[[V2:.+]] = arith.extf %[[V0]] : vector<1x2x3xf16> to vector<1x2x3xf32>
// CHECK: %[[V3:.+]] = arith.addf %[[V2]], %[[V1]] : vector<1x2x3xf32>
// CHECK: vector.transfer_write %[[V3:.+]], %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]] {in_bounds = [true, true, true]} : vector<1x2x3xf32>, memref<1x2x3xf32>

// -----

func.func @pooling_nwc_sum_memref_2_2_2_1(%input: memref<4x4x3xf32>, %filter: memref<2xf32>, %output: memref<4x2x3xf32>) {
  linalg.pooling_nwc_sum
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<4x4x3xf32>, memref<2xf32>)
    outs(%output : memref<4x2x3xf32>)
  return
}

// CHECK-LABEL: func.func @pooling_nwc_sum_memref_2_2_2_1
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x4x3xf32>, %[[FILTER:.+]]: memref<2xf32>, %[[OUTPUT:.+]]: memref<4x2x3xf32>)
// CHECK-DAG: %[[Vc0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[Vcst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[V0:.+]] = vector.transfer_read %[[INPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<4x4x3xf32>, vector<4x4x3xf32>
// CHECK: %[[V1:.+]] = vector.transfer_read %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<4x2x3xf32>, vector<4x2x3xf32>
// CHECK: %[[V2:.+]] = vector.extract_strided_slice %[[V0]] {offsets = [0, 0, 0], sizes = [4, 2, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x2x3xf32>
// CHECK: %[[V3:.+]] = vector.extract_strided_slice %[[V0]] {offsets = [0, 2, 0], sizes = [4, 2, 3], strides = [1, 1, 1]} : vector<4x4x3xf32> to vector<4x2x3xf32>
// CHECK: %[[V4:.+]] = arith.addf %[[V2]], %[[V1]] : vector<4x2x3xf32>
// CHECK: %[[V5:.+]] = arith.addf %[[V3]], %[[V4]] : vector<4x2x3xf32>
// CHECK: vector.transfer_write %[[V5:.+]], %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]] {in_bounds = [true, true, true]} : vector<4x2x3xf32>, memref<4x2x3xf32>


// -----

func.func @pooling_ncw_sum_memref_2_2_2_3(%input: memref<4x3x6xf32>, %filter: memref<2xf32>, %output: memref<4x3x2xf32>) {
  linalg.pooling_ncw_sum
    {dilations = dense<2> : tensor<1xi64>, strides = dense<3> : tensor<1xi64>}
    ins(%input, %filter : memref<4x3x6xf32>, memref<2xf32>)
    outs(%output : memref<4x3x2xf32>)
  return
}

// CHECK-LABEL: func.func @pooling_ncw_sum_memref_2_2_2_3
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x3x6xf32>, %[[FILTER:.+]]: memref<2xf32>, %[[OUTPUT:.+]]: memref<4x3x2xf32>)
// CHECK-DAG: %[[Vc0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[Vcst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[V0:.+]] = vector.transfer_read %[[INPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<4x3x6xf32>, vector<4x3x6xf32>
// CHECK: %[[V1:.+]] = vector.transfer_read %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<4x3x2xf32>, vector<4x3x2xf32>
// CHECK: %[[V2:.+]] = vector.transpose %[[V0]], [0, 2, 1] : vector<4x3x6xf32> to vector<4x6x3xf32>
// CHECK: %[[V3:.+]] = vector.transpose %[[V1]], [0, 2, 1] : vector<4x3x2xf32> to vector<4x2x3xf32>
// CHECK: %[[V4:.+]] = vector.extract_strided_slice %[[V2]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V5:.+]] = vector.extract_strided_slice %[[V2]] {offsets = [0, 3, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V6:.+]] = vector.extract_strided_slice %[[V2]] {offsets = [0, 2, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V7:.+]] = vector.extract_strided_slice %[[V2]] {offsets = [0, 5, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x6x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V8:.+]] = vector.extract_strided_slice %[[V3]] {offsets = [0, 0, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V9:.+]] = vector.extract_strided_slice %[[V3]] {offsets = [0, 1, 0], sizes = [4, 1, 3], strides = [1, 1, 1]} : vector<4x2x3xf32> to vector<4x1x3xf32>
// CHECK: %[[V10:.+]] = arith.addf %[[V4]], %[[V8]] : vector<4x1x3xf32>
// CHECK: %[[V11:.+]] = arith.addf %[[V5]], %[[V9]] : vector<4x1x3xf32>
// CHECK: %[[V12:.+]] = arith.addf %[[V6]], %[[V10]] : vector<4x1x3xf32>
// CHECK: %[[V13:.+]] = arith.addf %[[V7]], %[[V11]] : vector<4x1x3xf32>
// CHECK: %[[V14:.+]] = vector.insert_strided_slice %[[V12]], %[[V3]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<4x1x3xf32> into vector<4x2x3xf32>
// CHECK: %[[V15:.+]] = vector.insert_strided_slice %[[V13]], %[[V14]] {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<4x1x3xf32> into vector<4x2x3xf32>
// CHECK: %[[V16:.+]] = vector.transpose %[[V15]], [0, 2, 1] : vector<4x2x3xf32> to vector<4x3x2xf32>
// CHECK: vector.transfer_write %[[V16:.+]], %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]] {in_bounds = [true, true, true]} : vector<4x3x2xf32>, memref<4x3x2xf32>

// -----

func.func @pooling_ncw_sum_memref_2_3_2_1(%input: memref<4x2x5xf32>, %filter: memref<2xf32>, %output: memref<4x2x3xf32>) {
  linalg.pooling_ncw_sum
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<4x2x5xf32>, memref<2xf32>)
    outs(%output : memref<4x2x3xf32>)
  return
}

// CHECK-LABEL: func.func @pooling_ncw_sum_memref_2_3_2_1
// CHECK-SAME: (%[[INPUT:.+]]: memref<4x2x5xf32>, %[[FILTER:.+]]: memref<2xf32>, %[[OUTPUT:.+]]: memref<4x2x3xf32>)
// CHECK-DAG: %[[Vc0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[Vcst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[V0:.+]] = vector.transfer_read %[[INPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<4x2x5xf32>, vector<4x2x5xf32>
// CHECK: %[[V1:.+]] = vector.transfer_read %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]], %[[Vcst]] {in_bounds = [true, true, true]} : memref<4x2x3xf32>, vector<4x2x3xf32>
// CHECK: %[[V2:.+]] = vector.transpose %[[V0]], [0, 2, 1] : vector<4x2x5xf32> to vector<4x5x2xf32>
// CHECK: %[[V3:.+]] = vector.transpose %[[V1]], [0, 2, 1] : vector<4x2x3xf32> to vector<4x3x2xf32>
// CHECK: %[[V4:.+]] = vector.extract_strided_slice %[[V2]] {offsets = [0, 0, 0], sizes = [4, 3, 2], strides = [1, 1, 1]} : vector<4x5x2xf32> to vector<4x3x2xf32>
// CHECK: %[[V5:.+]] = vector.extract_strided_slice %[[V2]] {offsets = [0, 2, 0], sizes = [4, 3, 2], strides = [1, 1, 1]} : vector<4x5x2xf32> to vector<4x3x2xf32>
// CHECK: %[[V6:.+]] = arith.addf %[[V4]], %[[V3]] : vector<4x3x2xf32>
// CHECK: %[[V7:.+]] = arith.addf %[[V5]], %[[V6]] : vector<4x3x2xf32>
// CHECK: %[[V8:.+]] = vector.transpose %[[V7]], [0, 2, 1] : vector<4x3x2xf32> to vector<4x2x3xf32>
// CHECK: vector.transfer_write %[[V8:.+]], %[[OUTPUT]][%[[Vc0]], %[[Vc0]], %[[Vc0]]] {in_bounds = [true, true, true]} : vector<4x2x3xf32>, memref<4x2x3xf32>
