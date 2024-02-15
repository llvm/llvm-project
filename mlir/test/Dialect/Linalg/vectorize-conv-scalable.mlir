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

func.func @depthwise_conv1d_nwc_wc_3x5x4xf32_memref_dillation_2(%input: memref<3x5x?xf32>,
                                                                %filter: memref<2x?xf32>,
                                                                %output: memref<3x2x?xf32>) {
  linalg.depthwise_conv_1d_nwc_wc
    {dilations = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}
    ins(%input, %filter : memref<3x5x?xf32>, memref<2x?xf32>)
    outs(%output : memref<3x2x?xf32>)
  return
}

// TODO - nice variable names
// CHECK-LABEL:   func.func @depthwise_conv1d_nwc_wc_3x5x4xf32_memref_dillation_2(
// CHECK-SAME:     %[[VAL_0:.*]]: memref<3x5x?xf32>,
// CHECK-SAME:     %[[VAL_1:.*]]: memref<2x?xf32>,
// CHECK-SAME:     %[[VAL_2:.*]]: memref<3x2x?xf32>) {

// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index

/// Create a mask for the input tensor
// CHECK:           %[[VAL_7:.*]] = memref.dim %[[VAL_0]], %[[VAL_6]] : memref<3x5x?xf32>
// CHECK:           %[[VAL_8:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_10:.*]] = vector.create_mask %[[VAL_8]], %[[VAL_9]], %[[VAL_7]] : vector<3x4x[4]xi1>
/// Read the input tensor
// CHECK:           %[[VAL_11:.*]] = vector.mask %[[VAL_10]] { vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_4]], %[[VAL_4]], %[[VAL_4]]], %[[VAL_5]] : memref<3x5x?xf32>, vector<3x4x[4]xf32> } : vector<3x4x[4]xi1> -> vector<3x4x[4]xf32>

/// Create a mask for the filter tensor
// CHECK:           %[[VAL_12:.*]] = memref.dim %[[VAL_1]], %[[VAL_3]] : memref<2x?xf32>
// CHECK:           %[[VAL_13:.*]] = vector.create_mask %[[VAL_6]], %[[VAL_12]] : vector<2x[4]xi1>
/// Read the filter tensor
// CHECK:           %[[VAL_14:.*]] = vector.mask %[[VAL_13]] { vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_4]], %[[VAL_4]]], %[[VAL_5]] : memref<2x?xf32>, vector<2x[4]xf32> } : vector<2x[4]xi1> -> vector<2x[4]xf32>

/// Create a mask for the output tensor
// CHECK:           %[[VAL_15:.*]] = memref.dim %[[VAL_2]], %[[VAL_6]] : memref<3x2x?xf32>
// CHECK:           %[[VAL_16:.*]] = vector.create_mask %[[VAL_8]], %[[VAL_6]], %[[VAL_15]] : vector<3x2x[4]xi1>
/// Read the output tensor
// CHECK:           %[[VAL_17:.*]] = vector.mask %[[VAL_16]] { vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_4]], %[[VAL_4]], %[[VAL_4]]], %[[VAL_5]] : memref<3x2x?xf32>, vector<3x2x[4]xf32> } : vector<3x2x[4]xi1> -> vector<3x2x[4]xf32>

/// Convolution
// CHECK:           %[[VAL_18:.*]] = vector.extract_strided_slice %[[VAL_11]] {offsets = [0, 0, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[VAL_19:.*]] = vector.extract_strided_slice %[[VAL_11]] {offsets = [0, 2, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[VAL_20:.*]] = vector.extract %[[VAL_14]][0] : vector<[4]xf32> from vector<2x[4]xf32>
// CHECK:           %[[VAL_21:.*]] = vector.extract %[[VAL_14]][1] : vector<[4]xf32> from vector<2x[4]xf32>
// CHECK:           %[[VAL_22:.*]] = vector.extract_strided_slice %[[VAL_17]] {offsets = [0, 0, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x2x[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[VAL_23:.*]] = vector.broadcast %[[VAL_20]] : vector<[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[VAL_24:.*]] = vector.fma %[[VAL_18]], %[[VAL_23]], %[[VAL_22]] : vector<3x2x[4]xf32>
// CHECK:           %[[VAL_25:.*]] = vector.broadcast %[[VAL_21]] : vector<[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[VAL_26:.*]] = vector.fma %[[VAL_19]], %[[VAL_25]], %[[VAL_24]] : vector<3x2x[4]xf32>
// CHECK:           %[[VAL_27:.*]] = vector.insert_strided_slice %[[VAL_26]], %[[VAL_17]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<3x2x[4]xf32> into vector<3x2x[4]xf32>
// CHECK:           vector.mask %[[VAL_16]] { vector.transfer_write %[[VAL_27]], %[[VAL_2]]{{\[}}%[[VAL_4]], %[[VAL_4]], %[[VAL_4]]] : vector<3x2x[4]xf32>, memref<3x2x?xf32> } : vector<3x2x[4]xi1>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [3, 2, [4], 2] : !transform.any_op
    transform.yield
  }
}
