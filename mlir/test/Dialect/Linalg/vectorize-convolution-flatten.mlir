// RUN: mlir-opt -split-input-file -transform-interpreter %s | FileCheck %s

func.func @flatten_tensor(%input: tensor<1x8x3xi8>, %filter: tensor<1x3xi8>, %output: tensor<1x8x3xi8>) -> (tensor<1x8x3xi8>) {
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
// CHECK-LABEL:   func.func @flatten_tensor(
// CHECK-SAME:                              %[[VAL_0:.*]]: tensor<1x8x3xi8>,
// CHECK-SAME:                              %[[VAL_1:.*]]: tensor<1x3xi8>,
// CHECK-SAME:                              %[[VAL_2:.*]]: tensor<1x8x3xi8>) -> tensor<1x8x3xi8> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_5:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true, true]} : tensor<1x8x3xi8>, vector<1x8x3xi8>
// CHECK:           %[[VAL_6:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : tensor<1x3xi8>, vector<1x3xi8>
// CHECK:           %[[VAL_7:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true, true]} : tensor<1x8x3xi8>, vector<1x8x3xi8>
// CHECK:           %[[VAL_8:.*]] = vector.extract %[[VAL_6]][0] : vector<3xi8> from vector<1x3xi8>
// CHECK:           %[[VAL_9:.*]] = vector.shape_cast %[[VAL_5]] : vector<1x8x3xi8> to vector<1x24xi8>
// CHECK:           %[[VAL_10:.*]] = vector.shape_cast %[[VAL_7]] : vector<1x8x3xi8> to vector<1x24xi8>
// CHECK:           %[[VAL_11:.*]] = vector.broadcast %[[VAL_8]] : vector<3xi8> to vector<1x8x3xi8>
// CHECK:           %[[VAL_12:.*]] = vector.shape_cast %[[VAL_11]] : vector<1x8x3xi8> to vector<1x24xi8>
// CHECK:           %[[VAL_13:.*]] = arith.muli %[[VAL_9]], %[[VAL_12]] : vector<1x24xi8>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_10]] : vector<1x24xi8>
// CHECK:           %[[VAL_15:.*]] = vector.shape_cast %[[VAL_14]] : vector<1x24xi8> to vector<1x8x3xi8>
// CHECK:           %[[VAL_16:.*]] = vector.transfer_write %[[VAL_15]], %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true, true]} : vector<1x8x3xi8>, tensor<1x8x3xi8>
// CHECK:           return %[[VAL_16]] : tensor<1x8x3xi8>
// CHECK:         }

//------

func.func @flatten_memref(%input: memref<1x8x3xi8>, %filter: memref<1x3xi8>, %output: memref<1x8x3xi8>) {
  linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<1> : vector<1xi64>,
    strides = dense<1> : vector<1xi64>}
    ins(%input, %filter : memref<1x8x3xi8>, memref<1x3xi8>)
    outs(%output : memref<1x8x3xi8>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 {flatten_1d_depthwise_conv} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL:   func.func @flatten_memref(
// CHECK-SAME:                              %[[VAL_0:.*]]: memref<1x8x3xi8>,
// CHECK-SAME:                              %[[VAL_1:.*]]: memref<1x3xi8>,
// CHECK-SAME:                              %[[VAL_2:.*]]: memref<1x8x3xi8>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_5:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true, true]} : memref<1x8x3xi8>, vector<1x8x3xi8>
// CHECK:           %[[VAL_6:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<1x3xi8>, vector<1x3xi8>
// CHECK:           %[[VAL_7:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true, true]} : memref<1x8x3xi8>, vector<1x8x3xi8>
// CHECK:           %[[VAL_8:.*]] = vector.extract %[[VAL_6]][0] : vector<3xi8> from vector<1x3xi8>
// CHECK:           %[[VAL_9:.*]] = vector.shape_cast %[[VAL_5]] : vector<1x8x3xi8> to vector<1x24xi8>
// CHECK:           %[[VAL_10:.*]] = vector.shape_cast %[[VAL_7]] : vector<1x8x3xi8> to vector<1x24xi8>
// CHECK:           %[[VAL_11:.*]] = vector.broadcast %[[VAL_8]] : vector<3xi8> to vector<1x8x3xi8>
// CHECK:           %[[VAL_12:.*]] = vector.shape_cast %[[VAL_11]] : vector<1x8x3xi8> to vector<1x24xi8>
// CHECK:           %[[VAL_13:.*]] = arith.muli %[[VAL_9]], %[[VAL_12]] : vector<1x24xi8>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_10]] : vector<1x24xi8>
// CHECK:           %[[VAL_15:.*]] = vector.shape_cast %[[VAL_14]] : vector<1x24xi8> to vector<1x8x3xi8>
// CHECK:           vector.transfer_write %[[VAL_15]], %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true, true]} : vector<1x8x3xi8>, memref<1x8x3xi8>
// CHECK:           return
// CHECK:         }

// -----

func.func @flatten_memref_wider_filter(%input: memref<1x8x3xi8>, %filter: memref<2x3xi8>, %output: memref<1x7x3xi8>) {
  linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<1> : vector<1xi64>,
    strides = dense<1> : vector<1xi64>}
    ins(%input, %filter : memref<1x8x3xi8>, memref<2x3xi8>)
    outs(%output : memref<1x7x3xi8>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 {flatten_1d_depthwise_conv} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL:   func.func @flatten_memref_wider_filter(
// CHECK-SAME:                                           %[[VAL_0:.*]]: memref<1x8x3xi8>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: memref<2x3xi8>,
// CHECK-SAME:                                           %[[VAL_2:.*]]: memref<1x7x3xi8>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_5:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true, true]} : memref<1x8x3xi8>, vector<1x8x3xi8>
// CHECK:           %[[VAL_6:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<2x3xi8>, vector<2x3xi8>
// CHECK:           %[[VAL_7:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true, true]} : memref<1x7x3xi8>, vector<1x7x3xi8>
// CHECK:           %[[VAL_8:.*]] = vector.extract_strided_slice %[[VAL_5]] {offsets = [0, 0, 0], sizes = [1, 7, 3], strides = [1, 1, 1]} : vector<1x8x3xi8> to vector<1x7x3xi8>
// CHECK:           %[[VAL_9:.*]] = vector.extract_strided_slice %[[VAL_5]] {offsets = [0, 1, 0], sizes = [1, 7, 3], strides = [1, 1, 1]} : vector<1x8x3xi8> to vector<1x7x3xi8>
// CHECK:           %[[VAL_10:.*]] = vector.extract %[[VAL_6]][0] : vector<3xi8> from vector<2x3xi8>
// CHECK:           %[[VAL_11:.*]] = vector.extract %[[VAL_6]][1] : vector<3xi8> from vector<2x3xi8>
// CHECK:           %[[VAL_12:.*]] = vector.shape_cast %[[VAL_8]] : vector<1x7x3xi8> to vector<1x21xi8>
// CHECK:           %[[VAL_13:.*]] = vector.shape_cast %[[VAL_7]] : vector<1x7x3xi8> to vector<1x21xi8>
// CHECK:           %[[VAL_14:.*]] = vector.broadcast %[[VAL_10]] : vector<3xi8> to vector<1x7x3xi8>
// CHECK:           %[[VAL_15:.*]] = vector.shape_cast %[[VAL_14]] : vector<1x7x3xi8> to vector<1x21xi8>
// CHECK:           %[[VAL_16:.*]] = arith.muli %[[VAL_12]], %[[VAL_15]] : vector<1x21xi8>
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_13]] : vector<1x21xi8>
// CHECK:           %[[VAL_18:.*]] = vector.shape_cast %[[VAL_9]] : vector<1x7x3xi8> to vector<1x21xi8>
// CHECK:           %[[VAL_19:.*]] = vector.broadcast %[[VAL_11]] : vector<3xi8> to vector<1x7x3xi8>
// CHECK:           %[[VAL_20:.*]] = vector.shape_cast %[[VAL_19]] : vector<1x7x3xi8> to vector<1x21xi8>
// CHECK:           %[[VAL_21:.*]] = arith.muli %[[VAL_18]], %[[VAL_20]] : vector<1x21xi8>
// CHECK:           %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_17]] : vector<1x21xi8>
// CHECK:           %[[VAL_23:.*]] = vector.shape_cast %[[VAL_22]] : vector<1x21xi8> to vector<1x7x3xi8>
// CHECK:           vector.transfer_write %[[VAL_23]], %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true, true]} : vector<1x7x3xi8>, memref<1x7x3xi8>
// CHECK:           return
// CHECK:         }


// -----

func.func @depthwise_conv1d_nwc_wc_3x5x4xf32_memref(%input: memref<3x5x4xf32>, %filter: memref<2x4xf32>, %output: memref<3x2x4xf32>) {
  linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<3x5x4xf32>, memref<2x4xf32>)
    outs(%output : memref<3x2x4xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 {flatten_1d_depthwise_conv} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL:   func.func @depthwise_conv1d_nwc_wc_3x5x4xf32_memref(
// CHECK-SAME:                                                        %[[VAL_0:.*]]: memref<3x5x4xf32>,
// CHECK-SAME:                                                        %[[VAL_1:.*]]: memref<2x4xf32>,
// CHECK-SAME:                                                        %[[VAL_2:.*]]: memref<3x2x4xf32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_5:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true, true]} : memref<3x5x4xf32>, vector<3x4x4xf32>
// CHECK:           %[[VAL_6:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<2x4xf32>, vector<2x4xf32>
// CHECK:           %[[VAL_7:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true, true]} : memref<3x2x4xf32>, vector<3x2x4xf32>
// CHECK:           %[[VAL_8:.*]] = vector.extract_strided_slice %[[VAL_5]] {offsets = [0, 0, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x4xf32> to vector<3x2x4xf32>
// CHECK:           %[[VAL_9:.*]] = vector.extract_strided_slice %[[VAL_5]] {offsets = [0, 2, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x4xf32> to vector<3x2x4xf32>
// CHECK:           %[[VAL_10:.*]] = vector.extract %[[VAL_6]][0] : vector<4xf32> from vector<2x4xf32>
// CHECK:           %[[VAL_11:.*]] = vector.extract %[[VAL_6]][1] : vector<4xf32> from vector<2x4xf32>
// CHECK:           %[[VAL_12:.*]] = vector.shape_cast %[[VAL_8]] : vector<3x2x4xf32> to vector<3x8xf32>
// CHECK:           %[[VAL_13:.*]] = vector.shape_cast %[[VAL_7]] : vector<3x2x4xf32> to vector<3x8xf32>
// CHECK:           %[[VAL_14:.*]] = vector.broadcast %[[VAL_10]] : vector<4xf32> to vector<3x2x4xf32>
// CHECK:           %[[VAL_15:.*]] = vector.shape_cast %[[VAL_14]] : vector<3x2x4xf32> to vector<3x8xf32>
// CHECK:           %[[VAL_16:.*]] = vector.fma %[[VAL_12]], %[[VAL_15]], %[[VAL_13]] : vector<3x8xf32>
// CHECK:           %[[VAL_17:.*]] = vector.shape_cast %[[VAL_9]] : vector<3x2x4xf32> to vector<3x8xf32>
// CHECK:           %[[VAL_18:.*]] = vector.broadcast %[[VAL_11]] : vector<4xf32> to vector<3x2x4xf32>
// CHECK:           %[[VAL_19:.*]] = vector.shape_cast %[[VAL_18]] : vector<3x2x4xf32> to vector<3x8xf32>
// CHECK:           %[[VAL_20:.*]] = vector.fma %[[VAL_17]], %[[VAL_19]], %[[VAL_16]] : vector<3x8xf32>
// CHECK:           %[[VAL_21:.*]] = vector.shape_cast %[[VAL_20]] : vector<3x8xf32> to vector<3x2x4xf32>
// CHECK:           vector.transfer_write %[[VAL_21]], %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true, true]} : vector<3x2x4xf32>, memref<3x2x4xf32>
// CHECK:           return
// CHECK:         }

// -----

func.func @depthwise_conv1d_nwc_wc_3x5x4xi8_memref(%input: memref<3x5x4xi8>, %filter: memref<2x4xi8>, %output: memref<3x2x4xi32>) {
  linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<3x5x4xi8>, memref<2x4xi8>)
    outs(%output : memref<3x2x4xi32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL:   func.func @depthwise_conv1d_nwc_wc_3x5x4xi8_memref(
// CHECK-SAME:                                                       %[[VAL_0:.*]]: memref<3x5x4xi8>,
// CHECK-SAME:                                                       %[[VAL_1:.*]]: memref<2x4xi8>,
// CHECK-SAME:                                                       %[[VAL_2:.*]]: memref<3x2x4xi32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true, true]} : memref<3x5x4xi8>, vector<3x4x4xi8>
// CHECK:           %[[VAL_7:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<2x4xi8>, vector<2x4xi8>
// CHECK:           %[[VAL_8:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]], %[[VAL_5]] {in_bounds = [true, true, true]} : memref<3x2x4xi32>, vector<3x2x4xi32>
// CHECK:           %[[VAL_9:.*]] = vector.extract_strided_slice %[[VAL_6]] {offsets = [0, 0, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x4xi8> to vector<3x2x4xi8>
// CHECK:           %[[VAL_10:.*]] = vector.extract_strided_slice %[[VAL_6]] {offsets = [0, 2, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x4xi8> to vector<3x2x4xi8>
// CHECK:           %[[VAL_11:.*]] = vector.extract %[[VAL_7]][0] : vector<4xi8> from vector<2x4xi8>
// CHECK:           %[[VAL_12:.*]] = vector.extract %[[VAL_7]][1] : vector<4xi8> from vector<2x4xi8>
// CHECK:           %[[VAL_13:.*]] = arith.extsi %[[VAL_9]] : vector<3x2x4xi8> to vector<3x2x4xi32>
// CHECK:           %[[VAL_14:.*]] = arith.extsi %[[VAL_11]] : vector<4xi8> to vector<4xi32>
// CHECK:           %[[VAL_15:.*]] = vector.broadcast %[[VAL_14]] : vector<4xi32> to vector<3x2x4xi32>
// CHECK:           %[[VAL_16:.*]] = arith.muli %[[VAL_13]], %[[VAL_15]] : vector<3x2x4xi32>
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_8]] : vector<3x2x4xi32>
// CHECK:           %[[VAL_18:.*]] = arith.extsi %[[VAL_10]] : vector<3x2x4xi8> to vector<3x2x4xi32>
// CHECK:           %[[VAL_19:.*]] = arith.extsi %[[VAL_12]] : vector<4xi8> to vector<4xi32>
// CHECK:           %[[VAL_20:.*]] = vector.broadcast %[[VAL_19]] : vector<4xi32> to vector<3x2x4xi32>
// CHECK:           %[[VAL_21:.*]] = arith.muli %[[VAL_18]], %[[VAL_20]] : vector<3x2x4xi32>
// CHECK:           %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_17]] : vector<3x2x4xi32>
// CHECK:           vector.transfer_write %[[VAL_22]], %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true, true]} : vector<3x2x4xi32>, memref<3x2x4xi32>
// CHECK:           return
// CHECK:         }

