// RUN: mlir-opt -split-input-file -transform-interpreter %s | FileCheck %s

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
    transform.structured.vectorize %0 vector_sizes [1, 8, [4], 1] : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL:   func.func @depthwise_conv1d_nwc_wc_1x8x3xi8_tensor(
// CHECK-SAME:      %[[INPUT:.*]]: tensor<1x8x?xi8>,
// CHECK-SAME:      %[[FILTER:.*]]: tensor<1x?xi8>,
// CHECK-SAME:      %[[OUTPUT:.*]]: tensor<1x8x?xi8>) -> tensor<1x8x?xi8> {

// CHECK-DAG:       arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = tensor.dim %[[FILTER]], %[[VAL_5]] : tensor<1x?xi8>
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C0_I8:.*]] = arith.constant 0 : i8

/// Create a mask for the input tensor
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[CH_DIM_SIZE_INPUT:.*]] = tensor.dim %[[INPUT]], %[[C2]] : tensor<1x8x?xi8>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[MASK_IN:.*]] = vector.create_mask %[[C1]], %[[C8]], %[[CH_DIM_SIZE_INPUT]] : vector<1x8x[4]xi1>
/// Read the input tensor
// CHECK:           %[[VEC_IN:.*]] = vector.mask %[[MASK_IN]] { vector.transfer_read %[[INPUT]]{{\[}}%[[C0]], %[[C0]], %[[C0]]], %[[C0_I8]] : tensor<1x8x?xi8>, vector<1x8x[4]xi8> } : vector<1x8x[4]xi1> -> vector<1x8x[4]xi8>

/// Create a mask for the filter tensor
// CHECK:           %[[C0_I8_1:.*]] = arith.constant 0 : i8
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[CH_DIM_SIZE_FLT:.*]] = tensor.dim %[[FILTER]], %[[C1]] : tensor<1x?xi8>
// CHECK:           %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:           %[[MASK_FLT:.*]] = vector.create_mask %[[C1_1]], %[[CH_DIM_SIZE_FLT]] : vector<1x[4]xi1>
/// Read the filter tensor
// CHECK:           %[[VEC_FLT:.*]] = vector.mask %[[MASK_FLT]] { vector.transfer_read %[[FILTER]]{{\[}}%[[C0]], %[[C0]]], %[[C0_I8_1]] : tensor<1x?xi8>, vector<1x[4]xi8> } : vector<1x[4]xi1> -> vector<1x[4]xi8>

/// Create a mask for the output tensor
// CHECK:           %[[VAL_22:.*]] = arith.constant 0 : i8
// CHECK:           %[[VAL_23:.*]] = arith.constant 2 : index
// CHECK:           %[[CH_DIM_SIZE_OUT:.*]] = tensor.dim %[[OUTPUT]], %[[VAL_23]] : tensor<1x8x?xi8>
// CHECK:           %[[VAL_25:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_26:.*]] = arith.constant 8 : index
// CHECK:           %[[MASK_OUT:.*]] = vector.create_mask %[[VAL_25]], %[[VAL_26]], %[[CH_DIM_SIZE_OUT]] : vector<1x8x[4]xi1>
/// Read the output tensor
// CHECK:           %[[VEC_OUT:.*]] = vector.mask %[[MASK_OUT]] { vector.transfer_read %[[OUTPUT]]{{\[}}%[[C0]], %[[C0]], %[[C0]]], %[[VAL_22]] : tensor<1x8x?xi8>, vector<1x8x[4]xi8> } : vector<1x8x[4]xi1> -> vector<1x8x[4]xi8>

/// Convolution
// CHECK:           %[[VEC_IN_0:.*]] = vector.extract_strided_slice %[[VEC_IN]] {offsets = [0, 0, 0], sizes = [1, 8, 4], strides = [1, 1, 1]} : vector<1x8x[4]xi8> to vector<1x8x[4]xi8>
// CHECK:           %[[VEC_FLT_0:.*]] = vector.extract %[[VEC_FLT]][0] : vector<[4]xi8> from vector<1x[4]xi8>
// CHECK:           %[[VEC_OUT_0:.*]] = vector.extract_strided_slice %[[VEC_OUT]] {offsets = [0, 0, 0], sizes = [1, 8, 4], strides = [1, 1, 1]} : vector<1x8x[4]xi8> to vector<1x8x[4]xi8>
// CHECK:           %[[FLT_B:.*]] = vector.broadcast %[[VEC_FLT_0]] : vector<[4]xi8> to vector<1x8x[4]xi8>
// CHECK:           %[[MULI:.*]] = arith.muli %[[VEC_IN_0]], %[[FLT_B]] : vector<1x8x[4]xi8>
// CHECK:           %[[ADDI:.*]] = arith.addi %[[MULI]], %[[VEC_OUT_0]] : vector<1x8x[4]xi8>
// CHECK:           %[[VEC_OUT_1:.*]] = vector.insert_strided_slice %[[ADDI]], %[[VEC_OUT]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<1x8x[4]xi8> into vector<1x8x[4]xi8>

/// Create a mask for the output tensor
// CHECK:           %[[VAL_36:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_37:.*]] = tensor.dim %[[OUTPUT]], %[[VAL_36]] : tensor<1x8x?xi8>
// CHECK:           %[[VAL_38:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_39:.*]] = arith.constant 8 : index
// CHECK:           %[[MASK_OUT:.*]] = vector.create_mask %[[VAL_38]], %[[VAL_39]], %[[VAL_37]] : vector<1x8x[4]xi1>

/// Write the output tensor
// CHECK:           vector.mask %[[MASK_OUT]] { vector.transfer_write %[[VEC_OUT_1]], %[[OUTPUT]]{{\[}}%[[C0]], %[[C0]], %[[C0]]] : vector<1x8x[4]xi8>, tensor<1x8x?xi8> } : vector<1x8x[4]xi1> -> tensor<1x8x?xi8>


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

// CHECK-LABEL:   func.func @depthwise_conv1d_nwc_wc_3x5x4xf32_memref_dillation_2(
// CHECK-SAME:      %[[INPUT:.*]]: memref<3x5x?xf32>,
// CHECK-SAME:      %[[FILTER:.*]]: memref<2x?xf32>,
// CHECK-SAME:      %[[OUTPUT:.*]]: memref<3x2x?xf32>) {

// CHECK:           %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = memref.dim %[[FILTER]], %[[VAL_5]] : memref<2x?xf32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32

/// Create a mask for the input tensor
// CHECK:           %[[VAL_10:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_11:.*]] = memref.dim %[[INPUT]], %[[VAL_10]] : memref<3x5x?xf32>
// CHECK:           %[[VAL_12:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_13:.*]] = arith.constant 5 : index
// CHECK:           %[[MASK_IN:.*]] = vector.create_mask %[[VAL_12]], %[[VAL_13]], %[[VAL_11]] : vector<3x4x[4]xi1>
/// Read the input tensor
// CHECK:           %[[VEC_IN:.*]] = vector.mask %[[MASK_IN]] { vector.transfer_read %[[INPUT]]{{\[}}%[[VAL_8]], %[[VAL_8]], %[[VAL_8]]], %[[VAL_9]] : memref<3x5x?xf32>, vector<3x4x[4]xf32> } : vector<3x4x[4]xi1> -> vector<3x4x[4]xf32>

/// Create a mask for the filter tensor
// CHECK:           %[[VAL_16:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_18:.*]] = memref.dim %[[FILTER]], %[[VAL_17]] : memref<2x?xf32>
// CHECK:           %[[VAL_19:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_20:.*]] = vector.create_mask %[[VAL_19]], %[[VAL_18]] : vector<2x[4]xi1>
/// Read the filter tensor
// CHECK:           %[[VEC_FLT:.*]] = vector.mask %[[VAL_20]] { vector.transfer_read %[[FILTER]]{{\[}}%[[VAL_8]], %[[VAL_8]]], %[[VAL_16]] : memref<2x?xf32>, vector<2x[4]xf32> } : vector<2x[4]xi1> -> vector<2x[4]xf32>

/// Create a mask for the output tensor
// CHECK:           %[[VAL_22:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_23:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_24:.*]] = memref.dim %[[OUTPUT]], %[[VAL_23]] : memref<3x2x?xf32>
// CHECK:           %[[VAL_25:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_26:.*]] = arith.constant 2 : index
// CHECK:           %[[MASK_OUT:.*]] = vector.create_mask %[[VAL_25]], %[[VAL_26]], %[[VAL_24]] : vector<3x2x[4]xi1>
/// Read the output tensor
// CHECK:           %[[VEC_OUT:.*]] = vector.mask %[[MASK_OUT]] { vector.transfer_read %[[OUTPUT]]{{\[}}%[[VAL_8]], %[[VAL_8]], %[[VAL_8]]], %[[VAL_22]] : memref<3x2x?xf32>, vector<3x2x[4]xf32> } : vector<3x2x[4]xi1> -> vector<3x2x[4]xf32>

/// Convolution
// CHECK:           %[[VEC_IN_0:.*]] = vector.extract_strided_slice %[[VEC_IN]] {offsets = [0, 0, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[VEC_IN_1:.*]] = vector.extract_strided_slice %[[VEC_IN]] {offsets = [0, 2, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x4x[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[VEC_FLT_0:.*]] = vector.extract %[[VEC_FLT]][0] : vector<[4]xf32> from vector<2x[4]xf32>
// CHECK:           %[[VEC_FLT_1:.*]] = vector.extract %[[VEC_FLT]][1] : vector<[4]xf32> from vector<2x[4]xf32>
// CHECK:           %[[VEC_OUT_0:.*]] = vector.extract_strided_slice %[[VEC_OUT]] {offsets = [0, 0, 0], sizes = [3, 2, 4], strides = [1, 1, 1]} : vector<3x2x[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[VEC_FLT_0_B:.*]] = vector.broadcast %[[VEC_FLT_0]] : vector<[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[FMA_1:.*]] = vector.fma %[[VEC_IN_0]], %[[VEC_FLT_0_B]], %[[VEC_OUT_0]] : vector<3x2x[4]xf32>
// CHECK:           %[[VEC_FLT_1_B:.*]] = vector.broadcast %[[VEC_FLT_1]] : vector<[4]xf32> to vector<3x2x[4]xf32>
// CHECK:           %[[FMA_2:.*]] = vector.fma %[[VEC_IN_1]], %[[VEC_FLT_1_B]], %[[FMA_1]] : vector<3x2x[4]xf32>
// CHECK:           %[[VEC_OUT_1:.*]] = vector.insert_strided_slice %[[FMA_2]], %[[VEC_OUT]] {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<3x2x[4]xf32> into vector<3x2x[4]xf32>

/// Create a mask for the output tensor
// CHECK:           %[[VAL_39:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_40:.*]] = memref.dim %[[OUTPUT]], %[[VAL_39]] : memref<3x2x?xf32>
// CHECK:           %[[VAL_41:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_42:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_43:.*]] = vector.create_mask %[[VAL_41]], %[[VAL_42]], %[[VAL_40]] : vector<3x2x[4]xi1>
/// Write the output tensor
// CHECK:           vector.mask %[[VAL_43]] { vector.transfer_write %[[VEC_OUT_1]], %[[OUTPUT]]{{\[}}%[[VAL_8]], %[[VAL_8]], %[[VAL_8]]] : vector<3x2x[4]xf32>, memref<3x2x?xf32> } : vector<3x2x[4]xi1>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.depthwise_conv_1d_nwc_wc"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [3, 2, [4], 2] : !transform.any_op
    transform.yield
  }
}
