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
// CHECK:           %[[VAL_5:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : tensor<1x3xi8>, vector<1x3xi8>
// CHECK:           %[[VAL_6:.*]] = tensor.collapse_shape %[[VAL_0]] {{\[\[}}0], [1, 2]] : tensor<1x8x3xi8> into tensor<1x24xi8>
// CHECK:           %[[VAL_7:.*]] = tensor.collapse_shape %[[VAL_2]] {{\[\[}}0], [1, 2]] : tensor<1x8x3xi8> into tensor<1x24xi8>
// CHECK:           %[[VAL_8:.*]] = vector.transfer_read %[[VAL_6]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : tensor<1x24xi8>, vector<1x24xi8>
// CHECK:           %[[VAL_9:.*]] = vector.transfer_read %[[VAL_7]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : tensor<1x24xi8>, vector<1x24xi8>
// CHECK:           %[[VAL_10:.*]] = vector.extract %[[VAL_5]][0] : vector<3xi8> from vector<1x3xi8>
// CHECK:           %[[VAL_11:.*]] = vector.shuffle %[[VAL_10]], %[[VAL_10]] [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] : vector<3xi8>, vector<3xi8>
// CHECK:           %[[VAL_12:.*]] = vector.broadcast %[[VAL_11]] : vector<24xi8> to vector<1x24xi8>
// CHECK:           %[[VAL_13:.*]] = arith.muli %[[VAL_8]], %[[VAL_12]] : vector<1x24xi8>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_9]] : vector<1x24xi8>
// CHECK:           %[[VAL_15:.*]] = vector.transfer_write %[[VAL_14]], %[[VAL_7]]{{\[}}%[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true]} : vector<1x24xi8>, tensor<1x24xi8>
// CHECK:           %[[VAL_16:.*]] = tensor.expand_shape %[[VAL_15]] {{\[\[}}0], [1, 2]] : tensor<1x24xi8> into tensor<1x8x3xi8>
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
// CHECK:           %[[VAL_5:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<1x3xi8>, vector<1x3xi8>
// CHECK:           %[[VAL_6:.*]] = memref.collapse_shape %[[VAL_0]] {{\[\[}}0], [1, 2]] : memref<1x8x3xi8> into memref<1x24xi8>
// CHECK:           %[[VAL_7:.*]] = memref.collapse_shape %[[VAL_2]] {{\[\[}}0], [1, 2]] : memref<1x8x3xi8> into memref<1x24xi8>
// CHECK:           %[[VAL_8:.*]] = vector.transfer_read %[[VAL_6]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<1x24xi8>, vector<1x24xi8>
// CHECK:           %[[VAL_9:.*]] = vector.transfer_read %[[VAL_7]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<1x24xi8>, vector<1x24xi8>
// CHECK:           %[[VAL_10:.*]] = vector.extract %[[VAL_5]][0] : vector<3xi8> from vector<1x3xi8>
// CHECK:           %[[VAL_11:.*]] = vector.shuffle %[[VAL_10]], %[[VAL_10]] [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] : vector<3xi8>, vector<3xi8>
// CHECK:           %[[VAL_12:.*]] = vector.broadcast %[[VAL_11]] : vector<24xi8> to vector<1x24xi8>
// CHECK:           %[[VAL_13:.*]] = arith.muli %[[VAL_8]], %[[VAL_12]] : vector<1x24xi8>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_9]] : vector<1x24xi8>
// CHECK:           vector.transfer_write %[[VAL_14]], %[[VAL_7]]{{\[}}%[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true]} : vector<1x24xi8>, memref<1x24xi8>
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
// CHECK:           %[[VAL_5:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<2x3xi8>, vector<2x3xi8>
// CHECK:           %[[VAL_6:.*]] = memref.collapse_shape %[[VAL_0]] {{\[\[}}0], [1, 2]] : memref<1x8x3xi8> into memref<1x24xi8>
// CHECK:           %[[VAL_7:.*]] = memref.collapse_shape %[[VAL_2]] {{\[\[}}0], [1, 2]] : memref<1x7x3xi8> into memref<1x21xi8>
// CHECK:           %[[VAL_8:.*]] = vector.transfer_read %[[VAL_6]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<1x24xi8>, vector<1x24xi8>
// CHECK:           %[[VAL_9:.*]] = vector.transfer_read %[[VAL_7]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<1x21xi8>, vector<1x21xi8>
// CHECK:           %[[VAL_10:.*]] = vector.extract_strided_slice %[[VAL_8]] {offsets = [0, 0], sizes = [1, 21], strides = [1, 1]} : vector<1x24xi8> to vector<1x21xi8>
// CHECK:           %[[VAL_11:.*]] = vector.extract_strided_slice %[[VAL_8]] {offsets = [0, 3], sizes = [1, 21], strides = [1, 1]} : vector<1x24xi8> to vector<1x21xi8>
// CHECK:           %[[VAL_12:.*]] = vector.extract %[[VAL_5]][0] : vector<3xi8> from vector<2x3xi8>
// CHECK:           %[[VAL_13:.*]] = vector.extract %[[VAL_5]][1] : vector<3xi8> from vector<2x3xi8>
// CHECK:           %[[VAL_14:.*]] = vector.shuffle %[[VAL_12]], %[[VAL_12]] [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] : vector<3xi8>, vector<3xi8>
// CHECK:           %[[VAL_15:.*]] = vector.broadcast %[[VAL_14]] : vector<21xi8> to vector<1x21xi8>
// CHECK:           %[[VAL_16:.*]] = arith.muli %[[VAL_10]], %[[VAL_15]] : vector<1x21xi8>
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_9]] : vector<1x21xi8>
// CHECK:           %[[VAL_18:.*]] = vector.shuffle %[[VAL_13]], %[[VAL_13]] [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] : vector<3xi8>, vector<3xi8>
// CHECK:           %[[VAL_19:.*]] = vector.broadcast %[[VAL_18]] : vector<21xi8> to vector<1x21xi8>
// CHECK:           %[[VAL_20:.*]] = arith.muli %[[VAL_11]], %[[VAL_19]] : vector<1x21xi8>
// CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_17]] : vector<1x21xi8>
// CHECK:           vector.transfer_write %[[VAL_21]], %[[VAL_7]]{{\[}}%[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true]} : vector<1x21xi8>, memref<1x21xi8>
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
// CHECK:           %[[VAL_5:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<2x4xf32>, vector<2x4xf32>
// CHECK:           %[[VAL_6:.*]] = memref.collapse_shape %[[VAL_0]] {{\[\[}}0], [1, 2]] : memref<3x5x4xf32> into memref<3x20xf32>
// CHECK:           %[[VAL_7:.*]] = memref.collapse_shape %[[VAL_2]] {{\[\[}}0], [1, 2]] : memref<3x2x4xf32> into memref<3x8xf32>
// CHECK:           %[[VAL_8:.*]] = vector.transfer_read %[[VAL_6]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<3x20xf32>, vector<3x16xf32>
// CHECK:           %[[VAL_9:.*]] = vector.transfer_read %[[VAL_7]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<3x8xf32>, vector<3x8xf32>
// CHECK:           %[[VAL_10:.*]] = vector.extract_strided_slice %[[VAL_8]] {offsets = [0, 0], sizes = [3, 8], strides = [1, 1]} : vector<3x16xf32> to vector<3x8xf32>
// CHECK:           %[[VAL_11:.*]] = vector.extract_strided_slice %[[VAL_8]] {offsets = [0, 8], sizes = [3, 8], strides = [1, 1]} : vector<3x16xf32> to vector<3x8xf32>
// CHECK:           %[[VAL_12:.*]] = vector.extract %[[VAL_5]][0] : vector<4xf32> from vector<2x4xf32>
// CHECK:           %[[VAL_13:.*]] = vector.extract %[[VAL_5]][1] : vector<4xf32> from vector<2x4xf32>
// CHECK:           %[[VAL_14:.*]] = vector.shuffle %[[VAL_12]], %[[VAL_12]] [0, 1, 2, 3, 0, 1, 2, 3] : vector<4xf32>, vector<4xf32>
// CHECK:           %[[VAL_15:.*]] = vector.broadcast %[[VAL_14]] : vector<8xf32> to vector<3x8xf32>
// CHECK:           %[[VAL_16:.*]] = vector.fma %[[VAL_10]], %[[VAL_15]], %[[VAL_9]] : vector<3x8xf32>
// CHECK:           %[[VAL_17:.*]] = vector.shuffle %[[VAL_13]], %[[VAL_13]] [0, 1, 2, 3, 0, 1, 2, 3] : vector<4xf32>, vector<4xf32>
// CHECK:           %[[VAL_18:.*]] = vector.broadcast %[[VAL_17]] : vector<8xf32> to vector<3x8xf32>
// CHECK:           %[[VAL_19:.*]] = vector.fma %[[VAL_11]], %[[VAL_18]], %[[VAL_16]] : vector<3x8xf32>
// CHECK:           vector.transfer_write %[[VAL_19]], %[[VAL_7]]{{\[}}%[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true]} : vector<3x8xf32>, memref<3x8xf32>
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
// CHECK:           %[[VAL_6:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<2x4xi8>, vector<2x4xi8>
// CHECK:           %[[VAL_7:.*]] = memref.collapse_shape %[[VAL_0]] {{\[\[}}0], [1, 2]] : memref<3x5x4xi8> into memref<3x20xi8>
// CHECK:           %[[VAL_8:.*]] = memref.collapse_shape %[[VAL_2]] {{\[\[}}0], [1, 2]] : memref<3x2x4xi32> into memref<3x8xi32>
// CHECK:           %[[VAL_9:.*]] = vector.transfer_read %[[VAL_7]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : memref<3x20xi8>, vector<3x16xi8>
// CHECK:           %[[VAL_10:.*]] = vector.transfer_read %[[VAL_8]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_5]] {in_bounds = [true, true]} : memref<3x8xi32>, vector<3x8xi32>
// CHECK:           %[[VAL_11:.*]] = vector.extract_strided_slice %[[VAL_9]] {offsets = [0, 0], sizes = [3, 8], strides = [1, 1]} : vector<3x16xi8> to vector<3x8xi8>
// CHECK:           %[[VAL_12:.*]] = vector.extract_strided_slice %[[VAL_9]] {offsets = [0, 8], sizes = [3, 8], strides = [1, 1]} : vector<3x16xi8> to vector<3x8xi8>
// CHECK:           %[[VAL_13:.*]] = vector.extract %[[VAL_6]][0] : vector<4xi8> from vector<2x4xi8>
// CHECK:           %[[VAL_14:.*]] = vector.extract %[[VAL_6]][1] : vector<4xi8> from vector<2x4xi8>
// CHECK:           %[[VAL_15:.*]] = arith.extsi %[[VAL_11]] : vector<3x8xi8> to vector<3x8xi32>
// CHECK:           %[[VAL_16:.*]] = vector.shuffle %[[VAL_13]], %[[VAL_13]] [0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi8>, vector<4xi8>
// CHECK:           %[[VAL_17:.*]] = vector.broadcast %[[VAL_16]] : vector<8xi8> to vector<3x8xi8>
// CHECK:           %[[VAL_18:.*]] = arith.extsi %[[VAL_17]] : vector<3x8xi8> to vector<3x8xi32>
// CHECK:           %[[VAL_19:.*]] = arith.muli %[[VAL_15]], %[[VAL_18]] : vector<3x8xi32>
// CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_10]] : vector<3x8xi32>
// CHECK:           %[[VAL_21:.*]] = arith.extsi %[[VAL_12]] : vector<3x8xi8> to vector<3x8xi32>
// CHECK:           %[[VAL_22:.*]] = vector.shuffle %[[VAL_14]], %[[VAL_14]] [0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi8>, vector<4xi8>
// CHECK:           %[[VAL_23:.*]] = vector.broadcast %[[VAL_22]] : vector<8xi8> to vector<3x8xi8>
// CHECK:           %[[VAL_24:.*]] = arith.extsi %[[VAL_23]] : vector<3x8xi8> to vector<3x8xi32>
// CHECK:           %[[VAL_25:.*]] = arith.muli %[[VAL_21]], %[[VAL_24]] : vector<3x8xi32>
// CHECK:           %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_20]] : vector<3x8xi32>
// CHECK:           vector.transfer_write %[[VAL_26]], %[[VAL_8]]{{\[}}%[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true]} : vector<3x8xi32>, memref<3x8xi32>
// CHECK:           return
// CHECK:         }
