// RUN: mlir-opt -split-input-file -transform-interpreter -cse %s | FileCheck %s

func.func @depthwise_conv1d_nwc_wc_1x8x3xi8_tensor(%input: tensor<1x8x?xi8>,
                                                   %filter: tensor<1x?xi8>,
                                                   %output: tensor<1x8x?xi8>) -> (tensor<1x8x?xi8>) {
  %res = linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<1> : vector<1xi64>,
    strides = dense<1> : vector<1xi64>}
    ins(%input, %filter : tensor<1x8x?xi8>, tensor<1x?xi8>)
    outs(%output : tensor<1x8x?xi8>) -> tensor<1x8x?xi8>
  return %res : tensor<1x8x?xi8>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [1, 8, 4, 1] : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL:   func.func @depthwise_conv1d_nwc_wc_1x8x3xi8_tensor(
// CHECK-SAME:      %[[INPUT:.*]]: tensor<1x8x?xi8>,
// CHECK-SAME:      %[[FILTER:.*]]: tensor<1x?xi8>,
// CHECK-SAME:      %[[OUTPUT:.*]]: tensor<1x8x?xi8>) -> tensor<1x8x?xi8> {

// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[PAD:.*]] = arith.constant 0 : i8

/// Create a mask for the input tensor
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[CH_DIM_IN:.*]] = tensor.dim %[[INPUT]], %[[C2]] : tensor<1x8x?xi8>
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[MASK_IN:.*]] = vector.create_mask %[[C1]], %[[C8]], %[[CH_DIM_IN]] : vector<1x8x4xi1>
/// Read the input tensor
// CHECK:           %[[VEC_IN:.*]] = vector.mask %[[MASK_IN]] { vector.transfer_read %[[INPUT]]{{\[}}%[[C0]], %[[C0]], %[[C0]]], %[[PAD]] : tensor<1x8x?xi8>, vector<1x8x4xi8> } : vector<1x8x4xi1> -> vector<1x8x4xi8>

/// Create a mask for the filter tensor
// CHECK:           %[[CH_DIM_FLT:.*]] = tensor.dim %[[FILTER]], %[[C1]] : tensor<1x?xi8>
// CHECK:           %[[MASK_FLT:.*]] = vector.create_mask %[[C1]], %[[CH_DIM_FLT]] : vector<1x4xi1>
/// Read the filter tensor
// CHECK:           %[[VEC_FLT:.*]] = vector.mask %[[MASK_FLT]] { vector.transfer_read %[[FILTER]]{{\[}}%[[C0]], %[[C0]]], %[[PAD]] : tensor<1x?xi8>, vector<1x4xi8> } : vector<1x4xi1> -> vector<1x4xi8>

/// Create a mask for the output tensor
// CHECK:           %[[CH_DIM_OUT:.*]] = tensor.dim %[[OUTPUT]], %[[C2]] : tensor<1x8x?xi8>
// CHECK:           %[[MASK_OUT:.*]] = vector.create_mask %[[C1]], %[[C8]], %[[CH_DIM_OUT]] : vector<1x8x4xi1>
// CHECK:           %[[VEC_OUT:.*]] = vector.mask %[[MASK_OUT]] { vector.transfer_read %[[OUTPUT]]{{\[}}%[[C0]], %[[C0]], %[[C0]]], %[[PAD]] : tensor<1x8x?xi8>, vector<1x8x4xi8> } : vector<1x8x4xi1> -> vector<1x8x4xi8>

/// Convolution
// CHECK:           %[[IN_1:.*]] = vector.extract_strided_slice %[[VEC_IN]] {offsets = [0, 0, 0], sizes = [1, 8, 4], strides = [1, 1, 1]} : vector<1x8x4xi8> to vector<1x8x4xi8>
// CHECK:           %[[FLT_1:.*]] = vector.extract %[[VEC_FLT]][0] : vector<4xi8> from vector<1x4xi8>
// CHECK:           %[[OUT_1:.*]] = vector.extract_strided_slice %[[VEC_OUT]] {offsets = [0, 0, 0], sizes = [1, 8, 4], strides = [1, 1, 1]} : vector<1x8x4xi8> to vector<1x8x4xi8>
// CHECK:           %[[FLT_1_B:.*]] = vector.broadcast %[[FLT_1]] : vector<4xi8> to vector<1x8x4xi8>
// CHECK:           %[[MULI:.*]] = arith.muli %[[IN_1]], %[[FLT_1_B]] : vector<1x8x4xi8>
// CHECK:           %[[ADDI:.*]] = arith.addi %[[MULI]], %[[OUT_1]] : vector<1x8x4xi8>
// CHECK:           %[[OUT_INS:.*]] = vector.insert_strided_slice %[[ADDI]], %[[VEC_OUT]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<1x8x4xi8> into vector<1x8x4xi8>
// CHECK:           %[[OUT:.*]] = vector.mask %[[MASK_OUT]] { vector.transfer_write %[[OUT_INS]], %[[OUTPUT]]{{\[}}%[[C0]], %[[C0]], %[[C0]]] : vector<1x8x4xi8>, tensor<1x8x?xi8> } : vector<1x8x4xi1> -> tensor<1x8x?xi8>
// CHECK:           return %[[OUT]] : tensor<1x8x?xi8>

// -----

func.func @depthwise_conv1d_nwc_wc_1x8x3xi8_tensor_scalable(
      %input: tensor<1x8x?xi8>,
      %filter: tensor<1x?xi8>,
      %output: tensor<1x8x?xi8>) -> (tensor<1x8x?xi8>) {
  %res = linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<1> : vector<1xi64>,
    strides = dense<1> : vector<1xi64>}
    ins(%input, %filter : tensor<1x8x?xi8>, tensor<1x?xi8>)
    outs(%output : tensor<1x8x?xi8>) -> tensor<1x8x?xi8>
  return %res : tensor<1x8x?xi8>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [1, 8, [4], 1] : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL:   func.func @depthwise_conv1d_nwc_wc_1x8x3xi8_tensor_scalable(
// CHECK-SAME:      %[[INPUT:.*]]: tensor<1x8x?xi8>,
// CHECK-SAME:      %[[FILTER:.*]]: tensor<1x?xi8>,
// CHECK-SAME:      %[[OUTPUT:.*]]: tensor<1x8x?xi8>) -> tensor<1x8x?xi8> {

// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[PAD:.*]] = arith.constant 0 : i8

/// Create a mask for the input tensor
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[CH_DIM_IN:.*]] = tensor.dim %[[INPUT]], %[[C2]] : tensor<1x8x?xi8>
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[MASK_IN:.*]] = vector.create_mask %[[C1]], %[[C8]], %[[CH_DIM_IN]] : vector<1x8x[4]xi1>
/// Read the input tensor
// CHECK:           %[[VEC_IN:.*]] = vector.mask %[[MASK_IN]] { vector.transfer_read %[[INPUT]]{{\[}}%[[C0]], %[[C0]], %[[C0]]], %[[PAD]] : tensor<1x8x?xi8>, vector<1x8x[4]xi8> } : vector<1x8x[4]xi1> -> vector<1x8x[4]xi8>

/// Create a mask for the filter tensor
// CHECK:           %[[CH_DIM_FLT:.*]] = tensor.dim %[[FILTER]], %[[C1]] : tensor<1x?xi8>
// CHECK:           %[[MASK_FLT:.*]] = vector.create_mask %[[C1]], %[[CH_DIM_FLT]] : vector<1x[4]xi1>
/// Read the filter tensor
// CHECK:           %[[VEC_FLT:.*]] = vector.mask %[[MASK_FLT]] { vector.transfer_read %[[FILTER]]{{\[}}%[[C0]], %[[C0]]], %[[PAD]] : tensor<1x?xi8>, vector<1x[4]xi8> } : vector<1x[4]xi1> -> vector<1x[4]xi8>

/// Create a mask for the output tensor
// CHECK:           %[[CH_DIM_OUT:.*]] = tensor.dim %[[OUTPUT]], %[[C2]] : tensor<1x8x?xi8>
// CHECK:           %[[MASK_OUT:.*]] = vector.create_mask %[[C1]], %[[C8]], %[[CH_DIM_OUT]] : vector<1x8x[4]xi1>
/// Read the output tensor
// CHECK:           %[[VEC_OUT:.*]] = vector.mask %[[MASK_OUT]] { vector.transfer_read %[[OUTPUT]]{{\[}}%[[C0]], %[[C0]], %[[C0]]], %[[PAD]] : tensor<1x8x?xi8>, vector<1x8x[4]xi8> } : vector<1x8x[4]xi1> -> vector<1x8x[4]xi8>

/// Convolution
// CHECK:           %[[IN_1:.*]] = vector.extract_strided_slice %[[VEC_IN]] {offsets = [0, 0, 0], sizes = [1, 8, 4], strides = [1, 1, 1]} : vector<1x8x[4]xi8> to vector<1x8x[4]xi8>
// CHECK:           %[[FLT_1:.*]] = vector.extract %[[VEC_FLT]][0] : vector<[4]xi8> from vector<1x[4]xi8>
// CHECK:           %[[OUT_1:.*]] = vector.extract_strided_slice %[[VEC_OUT]] {offsets = [0, 0, 0], sizes = [1, 8, 4], strides = [1, 1, 1]} : vector<1x8x[4]xi8> to vector<1x8x[4]xi8>
// CHECK:           %[[FLT_1_B:.*]] = vector.broadcast %[[FLT_1]] : vector<[4]xi8> to vector<1x8x[4]xi8>
// CHECK:           %[[MULI:.*]] = arith.muli %[[IN_1]], %[[FLT_1_B]] : vector<1x8x[4]xi8>
// CHECK:           %[[ADDI:.*]] = arith.addi %[[MULI]], %[[OUT_1]] : vector<1x8x[4]xi8>
// CHECK:           %[[OUT_INS:.*]] = vector.insert_strided_slice %[[ADDI]], %[[VEC_OUT]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<1x8x[4]xi8> into vector<1x8x[4]xi8>
// CHECK:           %[[OUT:.*]] = vector.mask %[[MASK_OUT]] { vector.transfer_write %[[OUT_INS]], %[[OUTPUT]]{{\[}}%[[C0]], %[[C0]], %[[C0]]] : vector<1x8x[4]xi8>, tensor<1x8x?xi8> } : vector<1x8x[4]xi1> -> tensor<1x8x?xi8>
// CHECK:           return %[[OUT]] : tensor<1x8x?xi8>

// -----

func.func @depthwise_conv1d_nwc_wc_3x5x4xf32_memref_dilation_2(
      %input: memref<3x5x?xf32>,
      %filter: memref<2x?xf32>,
      %output: memref<3x2x?xf32>) {
  linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<3x5x?xf32>, memref<2x?xf32>)
    outs(%output : memref<3x2x?xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [3, 2, [4], 2] : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL:   func.func @depthwise_conv1d_nwc_wc_3x5x4xf32_memref_dilation_2(
// CHECK-SAME:      %[[INPUT:.*]]: memref<3x5x?xf32>,
// CHECK-SAME:      %[[FILTER:.*]]: memref<2x?xf32>,
// CHECK-SAME:      %[[OUTPUT:.*]]: memref<3x2x?xf32>) {

// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C2:.*]] = arith.constant 2 : index

/// Create a mask for the input tensor
// CHECK:           %[[CH_DIM_IN:.*]] = memref.dim %[[INPUT]], %[[C2]] : memref<3x5x?xf32>
// CHECK:           %[[C3:.*]] = arith.constant 3 : index
// CHECK:           %[[C5:.*]] = arith.constant 5 : index
// CHECK:           %[[MASK_IN:.*]] = vector.create_mask %[[C3]], %[[C5]], %[[CH_DIM_IN]] : vector<3x4x[4]xi1>
/// Read the input tensor
// CHECK:           %[[VEC_IN:.*]] = vector.mask %[[MASK_IN]] { vector.transfer_read %[[INPUT]]{{\[}}%[[C0]], %[[C0]], %[[C0]]], %[[PAD]] : memref<3x5x?xf32>, vector<3x4x[4]xf32> } : vector<3x4x[4]xi1> -> vector<3x4x[4]xf32>

/// Create a mask for the filter tensor
// CHECK:           %[[CH_DIM_FLT:.*]] = memref.dim %[[FILTER]], %[[C1]] : memref<2x?xf32>
// CHECK:           %[[MASK_FLT:.*]] = vector.create_mask %[[C2]], %[[CH_DIM_FLT]] : vector<2x[4]xi1>
/// Read the filter tensor
// CHECK:           %[[VEC_FLT:.*]] = vector.mask %[[MASK_FLT]] { vector.transfer_read %[[FILTER]]{{\[}}%[[C0]], %[[C0]]], %[[PAD]] : memref<2x?xf32>, vector<2x[4]xf32> } : vector<2x[4]xi1> -> vector<2x[4]xf32>

/// Create a mask for the output tensor
// CHECK:           %[[CH_DIM_OUT:.*]] = memref.dim %[[OUTPUT]], %[[C2]] : memref<3x2x?xf32>
// CHECK:           %[[MASK_OUT:.*]] = vector.create_mask %[[C3]], %[[C2]], %[[CH_DIM_OUT]] : vector<3x2x[4]xi1>
/// Read the output tensor
// CHECK:           %[[VEC_OUT:.*]] = vector.mask %[[MASK_OUT]] { vector.transfer_read %[[OUTPUT]]{{\[}}%[[C0]], %[[C0]], %[[C0]]], %[[PAD]] : memref<3x2x?xf32>, vector<3x2x[4]xf32> } : vector<3x2x[4]xi1> -> vector<3x2x[4]xf32>

/// Convolution
// CHECK:           %[[IN_1:.*]] = vector.extract_strided_slice %[[VEC_IN]] {offsets = [0, 0, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[IN_2:.*]] = vector.extract_strided_slice %[[VEC_IN]] {offsets = [0, 2, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[FLT_1:.*]] = vector.extract %[[VEC_FLT]][0] : vector<[4]xf32> from vector<2x[4]xf32>
// CHECK:           %[[FLT_2:.*]] = vector.extract %[[VEC_FLT]][1] : vector<[4]xf32> from vector<2x[4]xf32>
// CHECK:           %[[OUT_1:.*]] = vector.extract_strided_slice %[[VEC_OUT]] {offsets = [0, 0, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x2x[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[FLT_1_B:.*]] = vector.broadcast %[[FLT_1]] : vector<[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[FMA_1:.*]] = vector.fma %[[IN_1]], %[[FLT_1_B]], %[[OUT_1]] : vector<3x2x[4]xf32>
// CHECK:           %[[FLT_2_B:.*]] = vector.broadcast %[[FLT_2]] : vector<[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[FMA_2:.*]] = vector.fma %[[IN_2]], %[[FLT_2_B]], %[[FMA_1]] : vector<3x2x[4]xf32>
// CHECK:           %[[OUT_INS:.*]] = vector.insert_strided_slice %[[FMA_2]], %[[VEC_OUT]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<3x2x[4]xf32> into vector<3x2x[4]xf32>
// CHECK:           vector.mask %[[MASK_OUT]] { vector.transfer_write %[[OUT_INS]], %[[OUTPUT]]{{\[}}%[[C0]], %[[C0]], %[[C0]]] : vector<3x2x[4]xf32>, memref<3x2x?xf32> } : vector<3x2x[4]xi1>
