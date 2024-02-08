// RUN: mlir-opt -split-input-file -transform-interpreter %s | FileCheck %s

func.func @depthwise_conv1d_nwc_wc_1x8x3xi8_tensor(%input: tensor<1x8x3xi8>,
                                                   %filter: tensor<1x3xi8>,
                                                   %output: tensor<1x8x3xi8>) -> (tensor<1x8x3xi8>) {
  %res = linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<1> : vector<1xi64>,
    strides = dense<1> : vector<1xi64>}
    ins(%input, %filter : tensor<1x8x3xi8>, tensor<1x3xi8>)
    outs(%output : tensor<1x8x3xi8>) -> tensor<1x8x3xi8>
  return %res : tensor<1x8x3xi8>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 {flatten_1d_depthwise_conv} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
// CHECK-LABEL:   func.func @depthwise_conv1d_nwc_wc_1x8x3xi8_tensor
// CHECK-SAME:      %[[INPUT:.*]]: tensor<1x8x3xi8>,
// CHECK-SAME:      %[[FILTER:.*]]: tensor<1x3xi8>,
// CHECK-SAME:      %[[OUTPUT:.*]]: tensor<1x8x3xi8>) -> tensor<1x8x3xi8> {

// CHECK-DAG:       %[[C0_IDX:.*]] = arith.constant 0 : index

/// Read the whole data in one shot.
// CHECK:           %[[V_INPUT_R:.*]] = vector.transfer_read %[[INPUT]][%[[C0_IDX]], %[[C0_IDX]], %[[C0_IDX]]]
// CHECK:           %[[V_FILTER_R:.*]] = vector.transfer_read %[[FILTER]][%[[C0_IDX]], %[[C0_IDX]]]
// CHECK:           %[[V_OUTPUT_R:.*]] = vector.transfer_read %[[OUTPUT]][%[[C0_IDX]], %[[C0_IDX]], %[[C0_IDX]]]

// CHECK:           %[[V_FILTER_0:.*]] = vector.extract %[[V_FILTER_R]][0] : vector<3xi8> from vector<1x3xi8>

/// w == 0, kw = 0
// CHECK:           %[[SC_INPUT:.*]] = vector.shape_cast %[[V_INPUT_R]] : vector<1x8x3xi8> to vector<1x24xi8>
// CHECK:           %[[SC_OUTPUT:.*]] = vector.shape_cast %[[V_OUTPUT_R]] : vector<1x8x3xi8> to vector<1x24xi8>
// CHECK:           %[[SH_FILTER_0:.*]] = vector.shuffle %[[V_FILTER_0]], %[[V_FILTER_0]]
// CHECK-SAME:        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] : vector<3xi8>, vector<3xi8>
// CHECK:           %[[B_FILTER:.*]] = vector.broadcast %[[SH_FILTER_0]] : vector<24xi8> to vector<1x24xi8>
// CHECK:           %[[MULI:.*]] = arith.muli %[[SC_INPUT]], %[[B_FILTER]] : vector<1x24xi8>
// CHECK:           %[[ADDI:.*]] = arith.addi %[[MULI]], %[[SC_OUTPUT]] : vector<1x24xi8>

// Write the result back in one shot.
// CHECK:           %[[SC_ADDI:.*]] = vector.shape_cast %[[ADDI]] : vector<1x24xi8> to vector<1x8x3xi8>
// CHECK:           vector.transfer_write %[[SC_ADDI]], %[[OUTPUT]][%[[C0_IDX]], %[[C0_IDX]], %[[C0_IDX]]]

//------

func.func @depthwise_conv1d_nwc_wc_3x5x4xf32_memref_dillation_2(%input: memref<3x5x4xf32>,
                                                                %filter: memref<2x4xf32>,
                                                                %output: memref<3x2x4xf32>) {
  linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<3x5x4xf32>, memref<2x4xf32>)
    outs(%output : memref<3x2x4xf32>)
  return
}

//       CHECK: func @depthwise_conv1d_nwc_wc_3x5x4xf32_memref_dillation_2
//  CHECK-SAME:   (%[[INPUT:[0-9a-z]+]]: memref<3x5x4xf32>, %[[FILTER:[0-9a-z]+]]: memref<2x4xf32>, %[[OUTPUT:[0-9a-z]+]]: memref<3x2x4xf32>)

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32

/// Read the whole data in one shot.
//      CHECK-DAG:   %[[V_INPUT_R:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]]
//      CHECK-DAG:  %[[V_FILTER_R:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]]]
//      CHECK-DAG:  %[[V_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x4xf32> to vector<3x2x4xf32>
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 2, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x4xf32> to vector<3x2x4xf32>

//      CHECK:  %[[V_FILTER_0:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<4xf32> from vector<2x4xf32>
//      CHECK:  %[[V_FILTER_1:.+]] = vector.extract %[[V_FILTER_R]][1] : vector<4xf32> from vector<2x4xf32>


/// w == 0, kw = 0
// CHECK:           %[[SC_V_INPUT_0:.*]] = vector.shape_cast %[[V_INPUT_0]] : vector<3x2x4xf32> to vector<3x8xf32>
// CHECK:           %[[SC_V_OUTPUT_R:.*]] = vector.shape_cast %[[V_OUTPUT_R]] : vector<3x2x4xf32> to vector<3x8xf32>
// CHECK:           %[[SH_FILTER_0:.*]] = vector.shuffle %[[V_FILTER_0]], %[[V_FILTER_0]] 
// CHECK-SAME:        [0, 1, 2, 3, 0, 1, 2, 3] : vector<4xf32>, vector<4xf32>
// CHECK:           %[[B_FILTER_0:.*]] = vector.broadcast %[[SH_FILTER_0]] : vector<8xf32> to vector<3x8xf32>
// CHECK:           %[[FMA_0:.*]] = vector.fma %[[SC_V_INPUT_0]], %[[B_FILTER_0]], %[[SC_V_OUTPUT_R]] : vector<3x8xf32>

/// w == 0, kw = 1
// CHECK:           %[[SC_V_INPUT_1:.*]] = vector.shape_cast %[[V_INPUT_1]] : vector<3x2x4xf32> to vector<3x8xf32>
// CHECK:           %[[SH_FILTER_1:.*]] = vector.shuffle %[[V_FILTER_1]], %[[V_FILTER_1]] 
// CHECK-SAME:        [0, 1, 2, 3, 0, 1, 2, 3] : vector<4xf32>, vector<4xf32>
// CHECK:           %[[B_FILTER_1:.*]] = vector.broadcast %[[SH_FILTER_1]] : vector<8xf32> to vector<3x8xf32>
// CHECK:           %[[FMA_1:.*]] = vector.fma %[[SC_V_INPUT_1]], %[[B_FILTER_1]], %[[FMA_0]] : vector<3x8xf32>

// Write the result back in one shot.
//      CHECK:   %[[SC_FMA_1:.*]] = vector.shape_cast %[[FMA_1]] : vector<3x8xf32> to vector<3x2x4xf32>
//      CHECK:   vector.transfer_write %[[SC_FMA_1]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 {flatten_1d_depthwise_conv} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @depthwise_conv1d_nwc_wc_3x5x4xi8_memref_dilation_2(%input: memref<3x5x4xi8>,
                                                              %filter: memref<2x4xi8>,
                                                              %output: memref<3x2x4xi32>) {
  linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<3x5x4xi8>, memref<2x4xi8>)
    outs(%output : memref<3x2x4xi32>)
  return
}

//       CHECK: func @depthwise_conv1d_nwc_wc_3x5x4xi8_memref_dilation_2
//  CHECK-SAME:   (%[[INPUT:[0-9a-z]+]]: memref<3x5x4xi8>, %[[FILTER:[0-9a-z]+]]: memref<2x4xi8>, %[[OUTPUT:[0-9a-z]+]]: memref<3x2x4xi32>)

//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index

/// Read the whole data in one shot.
//      CHECK-DAG:   %[[V_INPUT_R:.+]] = vector.transfer_read %[[INPUT]][%[[C0]], %[[C0]], %[[C0]]]
//      CHECK-DAG:  %[[V_FILTER_R:.+]] = vector.transfer_read %[[FILTER]][%[[C0]], %[[C0]]]
//      CHECK-DAG:  %[[V_OUTPUT_R:.+]] = vector.transfer_read %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

//      CHECK:   %[[V_INPUT_0:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 0, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x4xi8> to vector<3x2x4xi8>
//      CHECK:   %[[V_INPUT_1:.+]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:     {offsets = [0, 2, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x4xi8> to vector<3x2x4xi8>

//      CHECK:  %[[V_FILTER_0:.+]] = vector.extract %[[V_FILTER_R]][0] : vector<4xi8> from vector<2x4xi8>
//      CHECK:  %[[V_FILTER_1:.+]] = vector.extract %[[V_FILTER_R]][1] : vector<4xi8> from vector<2x4xi8>

/// w == 0, kw = 0
//      CHECK:  %[[SC_V_INPUT_0:.*]] = vector.shape_cast %[[V_INPUT_0]] : vector<3x2x4xi8> to vector<3x8xi8>
//      CHECK:  %[[SC_V_OUTPUT_R:.*]] = vector.shape_cast %[[V_OUTPUT_R]] : vector<3x2x4xi32> to vector<3x8xi32>
//      CHECK:  %[[EXT_INPUT_0:.*]] = arith.extsi %[[SC_V_INPUT_0]] : vector<3x8xi8> to vector<3x8xi32>
//      CHECK:  %[[SH_FILTER_0:.*]] = vector.shuffle %[[V_FILTER_0]], %[[V_FILTER_0]]
//      CHECK-SAME:  [0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi8>, vector<4xi8>
//      CHECK:  %[[EXT_FILTER_0:.*]] = arith.extsi %[[SH_FILTER_0]] : vector<8xi8> to vector<8xi32>
//      CHECK:  %[[B_FILTER_0:.*]] = vector.broadcast %[[EXT_FILTER_0]] : vector<8xi32> to vector<3x8xi32>
//      CHECK:  %[[MUL_0:.*]] = arith.muli %[[EXT_INPUT_0]], %[[B_FILTER_0]] : vector<3x8xi32>
//      CHECK:  %[[ADD_0:.*]] = arith.addi %[[MUL_0]], %[[SC_V_OUTPUT_R]] : vector<3x8xi32>

/// w == 0, kw = 1
//      CHECK:  %[[SC_V_INPUT_1:.*]] = vector.shape_cast %[[V_INPUT_1]] : vector<3x2x4xi8> to vector<3x8xi8>
//      CHECK:  %[[EXT_INPUT_1:.*]] = arith.extsi %[[SC_V_INPUT_1]] : vector<3x8xi8> to vector<3x8xi32>
//      CHECK:  %[[SH_FILTER_1:.*]] = vector.shuffle %[[V_FILTER_1]], %[[V_FILTER_1]]
//      CHECK-SAME:  [0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi8>, vector<4xi8>
//      CHECK:  %[[EXT_FILTER_1:.*]] = arith.extsi %[[SH_FILTER_1]] : vector<8xi8> to vector<8xi32>
//      CHECK:  %[[B_FILTER_1:.*]] = vector.broadcast %[[EXT_FILTER_1]] : vector<8xi32> to vector<3x8xi32>
//      CHECK:  %[[MUL_1:.*]] = arith.muli %[[EXT_INPUT_1]], %[[B_FILTER_1]] : vector<3x8xi32>
//      CHECK:  %[[ADD_1:.*]] = arith.addi %[[MUL_1]], %[[ADD_0]] : vector<3x8xi32>

// Write the result back in one shot.
//      CHECK:   %[[SC_ADD_1:.*]] = vector.shape_cast %[[ADD_1]] : vector<3x8xi32> to vector<3x2x4xi32>
//      CHECK:   vector.transfer_write %[[SC_ADD_1]], %[[OUTPUT]][%[[C0]], %[[C0]], %[[C0]]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 {flatten_1d_depthwise_conv} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @depthwise_conv1d_nwc_wc_3x9x4xi8_tensor_stride_2(%input: tensor<3x9x4xi8>,
                                                            %filter: tensor<3x4xi8>,
                                                            %output: tensor<3x3x4xi8>) -> tensor<3x3x4xi8> {
  %res = linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<1> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>}
    ins(%input, %filter : tensor<3x9x4xi8>, tensor<3x4xi8>)
    outs(%output : tensor<3x3x4xi8>) -> tensor<3x3x4xi8>
  return %res : tensor<3x3x4xi8>
}
// CHECK-LABEL:   func.func @depthwise_conv1d_nwc_wc_3x9x4xi8_tensor_stride_2
// CHECK-SAME:      %[[INPUT:.*]]: tensor<3x9x4xi8>,
// CHECK-SAME:      %[[FILTER:.*]]: tensor<3x4xi8>,
// CHECK-SAME:      %[[OUTPUT:.*]]: tensor<3x3x4xi8>) -> tensor<3x3x4xi8> {

// CHECK-DAG:           %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[C0_I8:.*]] = arith.constant 0 : i8

/// Read the whole data in one shot.
// CHECK:           %[[V_INPUT_R:.*]] = vector.transfer_read %[[INPUT]][%[[C0_IDX]], %[[C0_IDX]], %[[C0_IDX]]], %[[C0_I8]]
// CHECK:           %[[V_FILTER_R:.*]] = vector.transfer_read %[[FILTER]][%[[C0_IDX]], %[[C0_IDX]]], %[[C0_I8]]
// CHECK:           %[[V_OUTPUT_R:.*]] = vector.transfer_read %[[OUTPUT]][%[[C0_IDX]], %[[C0_IDX]], %[[C0_IDX]]], %[[C0_I8]]

// CHECK:           %[[V_INPUT_0:.*]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:        {offsets = [0, 0, 0], sizes = [3, 1, 4], strides = [1, 1, 1]} : vector<3x7x4xi8> to vector<3x1x4xi8>
// CHECK:           %[[V_INPUT_1:.*]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:        {offsets = [0, 2, 0], sizes = [3, 1, 4], strides = [1, 1, 1]} : vector<3x7x4xi8> to vector<3x1x4xi8>
// CHECK:           %[[V_INPUT_2:.*]] = vector.extract_strided_slice %[[V_INPUT_R]] 
// CHECK-SAME:        {offsets = [0, 4, 0], sizes = [3, 1, 4], strides = [1, 1, 1]} : vector<3x7x4xi8> to vector<3x1x4xi8>
// CHECK:           %[[V_INPUT_3:.*]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:        {offsets = [0, 1, 0], sizes = [3, 1, 4], strides = [1, 1, 1]} : vector<3x7x4xi8> to vector<3x1x4xi8>
// CHECK:           %[[V_INPUT_4:.*]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:        {offsets = [0, 3, 0], sizes = [3, 1, 4], strides = [1, 1, 1]} : vector<3x7x4xi8> to vector<3x1x4xi8>
// CHECK:           %[[V_INPUT_5:.*]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:        {offsets = [0, 5, 0], sizes = [3, 1, 4], strides = [1, 1, 1]} : vector<3x7x4xi8> to vector<3x1x4xi8>
// CHECK:           %[[V_INPUT_6:.*]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:        {offsets = [0, 2, 0], sizes = [3, 1, 4], strides = [1, 1, 1]} : vector<3x7x4xi8> to vector<3x1x4xi8>
// CHECK:           %[[V_INPUT_7:.*]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:        {offsets = [0, 4, 0], sizes = [3, 1, 4], strides = [1, 1, 1]} : vector<3x7x4xi8> to vector<3x1x4xi8>
// CHECK:           %[[V_INPUT_8:.*]] = vector.extract_strided_slice %[[V_INPUT_R]]
// CHECK-SAME:        {offsets = [0, 6, 0], sizes = [3, 1, 4], strides = [1, 1, 1]} : vector<3x7x4xi8> to vector<3x1x4xi8>

// CHECK:           %[[V_FILTER_0:.*]] = vector.extract %[[V_FILTER_R]][0] : vector<4xi8> from vector<3x4xi8>
// CHECK:           %[[V_FILTER_1:.*]] = vector.extract %[[V_FILTER_R]][1] : vector<4xi8> from vector<3x4xi8>
// CHECK:           %[[V_FILTER_2:.*]] = vector.extract %[[V_FILTER_R]][2] : vector<4xi8> from vector<3x4xi8>

// CHECK:           %[[V_OUTPUT_0:.*]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:        {offsets = [0, 0, 0], sizes = [3, 1, 4], strides = [1, 1, 1]} : vector<3x3x4xi8> to vector<3x1x4xi8>
// CHECK:           %[[V_OUTPUT_1:.*]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:       {offsets = [0, 1, 0], sizes = [3, 1, 4], strides = [1, 1, 1]} : vector<3x3x4xi8> to vector<3x1x4xi8>
// CHECK:           %[[V_OUTPUT_2:.*]] = vector.extract_strided_slice %[[V_OUTPUT_R]]
// CHECK-SAME:        {offsets = [0, 2, 0], sizes = [3, 1, 4], strides = [1, 1, 1]} : vector<3x3x4xi8> to vector<3x1x4xi8>

/// w == 0, kw == 0
// CHECK:           %[[VAL_23:.*]] = vector.shape_cast %[[V_INPUT_0]] : vector<3x1x4xi8> to vector<3x4xi8>
// CHECK:           %[[VAL_24:.*]] = vector.shape_cast %[[V_OUTPUT_0]] : vector<3x1x4xi8> to vector<3x4xi8>
// CHECK:           %[[B_FILTER_0:.*]] = vector.broadcast %[[V_FILTER_0]] : vector<4xi8> to vector<3x4xi8>
// CHECK:           %[[VAL_27:.*]] = arith.muli %[[VAL_23]], %[[B_FILTER_0]] : vector<3x4xi8>
// CHECK:           %[[VAL_28:.*]] = arith.addi %[[VAL_27]], %[[VAL_24]] : vector<3x4xi8>

/// w == 1, kw == 0
// CHECK:           %[[VAL_29:.*]] = vector.shape_cast %[[V_INPUT_1]] : vector<3x1x4xi8> to vector<3x4xi8>
// CHECK:           %[[VAL_30:.*]] = vector.shape_cast %[[V_OUTPUT_1]] : vector<3x1x4xi8> to vector<3x4xi8>
// CHECK:           %[[B_FILTER_0_1:.*]] = vector.broadcast %[[V_FILTER_0]] : vector<4xi8> to vector<3x4xi8>
// CHECK:           %[[VAL_33:.*]] = arith.muli %[[VAL_29]], %[[B_FILTER_0_1]] : vector<3x4xi8>
// CHECK:           %[[VAL_34:.*]] = arith.addi %[[VAL_33]], %[[VAL_30]] : vector<3x4xi8>

/// w == 2, kw == 0
// CHECK:           %[[VAL_35:.*]] = vector.shape_cast %[[V_INPUT_2]] : vector<3x1x4xi8> to vector<3x4xi8>
// CHECK:           %[[VAL_36:.*]] = vector.shape_cast %[[V_OUTPUT_2]] : vector<3x1x4xi8> to vector<3x4xi8>
// CHECK:           %[[B_FILTER_0_2:.*]] = vector.broadcast %[[V_FILTER_0]] : vector<4xi8> to vector<3x4xi8>
// CHECK:           %[[VAL_39:.*]] = arith.muli %[[VAL_35]], %[[B_FILTER_0_2]] : vector<3x4xi8>
// CHECK:           %[[VAL_40:.*]] = arith.addi %[[VAL_39]], %[[VAL_36]] : vector<3x4xi8>

/// w == 3, kw == 1
// CHECK:           %[[VAL_41:.*]] = vector.shape_cast %[[V_INPUT_3]] : vector<3x1x4xi8> to vector<3x4xi8>
// CHECK:           %[[B_FILTER_1:.*]] = vector.broadcast %[[V_FILTER_1]] : vector<4xi8> to vector<3x4xi8>
// CHECK:           %[[VAL_44:.*]] = arith.muli %[[VAL_41]], %[[B_FILTER_1]] : vector<3x4xi8>
// CHECK:           %[[VAL_45:.*]] = arith.addi %[[VAL_44]], %[[VAL_28]] : vector<3x4xi8>

/// w == 4, kw == 1
// CHECK:           %[[VAL_46:.*]] = vector.shape_cast %[[V_INPUT_4]] : vector<3x1x4xi8> to vector<3x4xi8>
// CHECK:           %[[B_FILTER_1_1:.*]] = vector.broadcast %[[V_FILTER_1]] : vector<4xi8> to vector<3x4xi8>
// CHECK:           %[[VAL_49:.*]] = arith.muli %[[VAL_46]], %[[B_FILTER_1_1]] : vector<3x4xi8>
// CHECK:           %[[VAL_50:.*]] = arith.addi %[[VAL_49]], %[[VAL_34]] : vector<3x4xi8>

/// w == 5, kw == 1
// CHECK:           %[[VAL_51:.*]] = vector.shape_cast %[[V_INPUT_5]] : vector<3x1x4xi8> to vector<3x4xi8>
// CHECK:           %[[B_FILTER_1_2:.*]] = vector.broadcast %[[V_FILTER_1]] : vector<4xi8> to vector<3x4xi8>
// CHECK:           %[[VAL_54:.*]] = arith.muli %[[VAL_51]], %[[B_FILTER_1_2]] : vector<3x4xi8>
// CHECK:           %[[VAL_55:.*]] = arith.addi %[[VAL_54]], %[[VAL_40]] : vector<3x4xi8>

/// w == 6, kw == 2
// CHECK:           %[[VAL_56:.*]] = vector.shape_cast %[[V_INPUT_6]] : vector<3x1x4xi8> to vector<3x4xi8>
// CHECK:           %[[B_FILTER_2:.*]] = vector.broadcast %[[V_FILTER_2]] : vector<4xi8> to vector<3x4xi8>
// CHECK:           %[[VAL_59:.*]] = arith.muli %[[VAL_56]], %[[B_FILTER_2]] : vector<3x4xi8>
// CHECK:           %[[VAL_60:.*]] = arith.addi %[[VAL_59]], %[[VAL_45]] : vector<3x4xi8>

/// w == 7, kw == 2
// CHECK:           %[[VAL_61:.*]] = vector.shape_cast %[[VAL_60]] : vector<3x4xi8> to vector<3x1x4xi8>
// CHECK:           %[[VAL_62:.*]] = vector.shape_cast %[[V_INPUT_7]] : vector<3x1x4xi8> to vector<3x4xi8>
// CHECK:           %[[B_FILTER_2_1:.*]] = vector.broadcast %[[V_FILTER_2]] : vector<4xi8> to vector<3x4xi8>
// CHECK:           %[[VAL_65:.*]] = arith.muli %[[VAL_62]], %[[B_FILTER_2_1]] : vector<3x4xi8>
// CHECK:           %[[VAL_66:.*]] = arith.addi %[[VAL_65]], %[[VAL_50]] : vector<3x4xi8>

/// w == 8, kw == 2
// CHECK:           %[[VAL_67:.*]] = vector.shape_cast %[[VAL_66]] : vector<3x4xi8> to vector<3x1x4xi8>
// CHECK:           %[[VAL_68:.*]] = vector.shape_cast %[[V_INPUT_8]] : vector<3x1x4xi8> to vector<3x4xi8>
// CHECK:           %[[B_FILTER_2_2:.*]] = vector.broadcast %[[V_FILTER_2]] : vector<4xi8> to vector<3x4xi8>
// CHECK:           %[[VAL_71:.*]] = arith.muli %[[VAL_68]], %[[B_FILTER_2_2]] : vector<3x4xi8>
// CHECK:           %[[VAL_72:.*]] = arith.addi %[[VAL_71]], %[[VAL_55]] : vector<3x4xi8>

// Write the result back.
// CHECK:           %[[VAL_73:.*]] = vector.shape_cast %[[VAL_72]] : vector<3x4xi8> to vector<3x1x4xi8>
// CHECK:           %[[VAL_74:.*]] = vector.insert_strided_slice %[[VAL_61]], %[[V_OUTPUT_R]]
// CHECK-SAME:        {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<3x1x4xi8> into vector<3x3x4xi8>
// CHECK:           %[[VAL_75:.*]] = vector.insert_strided_slice %[[VAL_67]], %[[VAL_74]]
// CHECK-SAME:        {offsets = [0, 1, 0], strides = [1, 1, 1]} : vector<3x1x4xi8> into vector<3x3x4xi8>
// CHECK:           %[[VAL_76:.*]] = vector.insert_strided_slice %[[VAL_73]], %[[VAL_75]]
// CHECK-SAME:        {offsets = [0, 2, 0], strides = [1, 1, 1]} : vector<3x1x4xi8> into vector<3x3x4xi8>
// CHECK:           %[[VAL_77:.*]] = vector.transfer_write %[[VAL_76]], %[[OUTPUT]][%[[C0_IDX]], %[[C0_IDX]], %[[C0_IDX]]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 {flatten_1d_depthwise_conv} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

