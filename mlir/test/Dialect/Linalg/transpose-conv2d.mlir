// RUN: mlir-opt %s -transform-interpreter | FileCheck %s

// CHECK-LABEL: @conv_2d_nhwc_fhwc_f64
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x4x4x6xf64>, %[[FILTER:.+]]: tensor<8x2x2x6xf64>, %[[INIT:.+]]: tensor<1x2x2x8xf64>) -> tensor<1x2x2x8xf64> {
// CHECK-DAG:    %[[NEWF:.+]] = tensor.empty() : tensor<2x2x6x8xf64>
// CHECK:    %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[FILTER]] : tensor<8x2x2x6xf64>) outs(%[[NEWF]] : tensor<2x2x6x8xf64>) permutation = [1, 2, 3, 0]
// CHECK:    %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%[[INPUT]], %[[TRANSPOSE]] : tensor<1x4x4x6xf64>, tensor<2x2x6x8xf64>) outs(%[[INIT]] : tensor<1x2x2x8xf64>) -> tensor<1x2x2x8xf64>
// CHECK:    return %[[CONV]] : tensor<1x2x2x8xf64>
func.func @conv_2d_nhwc_fhwc_f64(%input: tensor<1x4x4x6xf64>, %filter: tensor<8x2x2x6xf64>, %init: tensor<1x2x2x8xf64>) -> tensor<1x2x2x8xf64> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<2> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x4x4x6xf64>, tensor<8x2x2x6xf64>)
    outs (%init: tensor<1x2x2x8xf64>) -> tensor<1x2x2x8xf64>
  return %0 : tensor<1x2x2x8xf64>
}

// CHECK-LABEL: @conv_2d_nhwc_fhwc_f32
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x4x4x6xf32>, %[[FILTER:.+]]: tensor<8x2x2x6xf32>, %[[INIT:.+]]: tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32> {
// CHECK-DAG:    %[[NEWF:.+]] = tensor.empty() : tensor<2x2x6x8xf32>
// CHECK:    %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[FILTER]] : tensor<8x2x2x6xf32>) outs(%[[NEWF]] : tensor<2x2x6x8xf32>) permutation = [1, 2, 3, 0]
// CHECK:    %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%[[INPUT]], %[[TRANSPOSE]] : tensor<1x4x4x6xf32>, tensor<2x2x6x8xf32>) outs(%[[INIT]] : tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
// CHECK:    return %[[CONV]] : tensor<1x2x2x8xf32>
func.func @conv_2d_nhwc_fhwc_f32(%input: tensor<1x4x4x6xf32>, %filter: tensor<8x2x2x6xf32>, %init: tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<2> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x4x4x6xf32>, tensor<8x2x2x6xf32>)
    outs (%init: tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
  return %0 : tensor<1x2x2x8xf32>
}

// CHECK-LABEL: @conv_2d_nhwc_fhwc_f16
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x4x4x6xf16>, %[[FILTER:.+]]: tensor<8x2x2x6xf16>, %[[INIT:.+]]: tensor<1x2x2x8xf16>) -> tensor<1x2x2x8xf16> {
// CHECK-DAG:    %[[NEWF:.+]] = tensor.empty() : tensor<2x2x6x8xf16>
// CHECK:    %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[FILTER]] : tensor<8x2x2x6xf16>) outs(%[[NEWF]] : tensor<2x2x6x8xf16>) permutation = [1, 2, 3, 0]
// CHECK:    %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%[[INPUT]], %[[TRANSPOSE]] : tensor<1x4x4x6xf16>, tensor<2x2x6x8xf16>) outs(%[[INIT]] : tensor<1x2x2x8xf16>) -> tensor<1x2x2x8xf16>
// CHECK:    return %[[CONV]] : tensor<1x2x2x8xf16>
func.func @conv_2d_nhwc_fhwc_f16(%input: tensor<1x4x4x6xf16>, %filter: tensor<8x2x2x6xf16>, %init: tensor<1x2x2x8xf16>) -> tensor<1x2x2x8xf16> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<2> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x4x4x6xf16>, tensor<8x2x2x6xf16>)
    outs (%init: tensor<1x2x2x8xf16>) -> tensor<1x2x2x8xf16>
  return %0 : tensor<1x2x2x8xf16>
}

// CHECK-LABEL: @conv_2d_nhwc_fhwc_b16
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x4x4x6xbf16>, %[[FILTER:.+]]: tensor<8x2x2x6xbf16>, %[[INIT:.+]]: tensor<1x2x2x8xbf16>) -> tensor<1x2x2x8xbf16> {
// CHECK-DAG:    %[[NEWF:.+]] = tensor.empty() : tensor<2x2x6x8xbf16>
// CHECK:    %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[FILTER]] : tensor<8x2x2x6xbf16>) outs(%[[NEWF]] : tensor<2x2x6x8xbf16>) permutation = [1, 2, 3, 0]
// CHECK:    %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%[[INPUT]], %[[TRANSPOSE]] : tensor<1x4x4x6xbf16>, tensor<2x2x6x8xbf16>) outs(%[[INIT]] : tensor<1x2x2x8xbf16>) -> tensor<1x2x2x8xbf16>
// CHECK:    return %[[CONV]] : tensor<1x2x2x8xbf16>
func.func @conv_2d_nhwc_fhwc_b16(%input: tensor<1x4x4x6xbf16>, %filter: tensor<8x2x2x6xbf16>, %init: tensor<1x2x2x8xbf16>) -> tensor<1x2x2x8xbf16> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<2> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x4x4x6xbf16>, tensor<8x2x2x6xbf16>)
    outs (%init: tensor<1x2x2x8xbf16>) -> tensor<1x2x2x8xbf16>
  return %0 : tensor<1x2x2x8xbf16>
}

// CHECK-LABEL: @conv_2d_nhwc_fhwc
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x4x4x6xi64>, %[[FILTER:.+]]: tensor<8x2x2x6xi64>, %[[INIT:.+]]: tensor<1x2x2x8xi64>) -> tensor<1x2x2x8xi64> {
// CHECK-DAG:    %[[NEWF:.+]] = tensor.empty() : tensor<2x2x6x8xi64>
// CHECK:    %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[FILTER]] : tensor<8x2x2x6xi64>) outs(%[[NEWF]] : tensor<2x2x6x8xi64>) permutation = [1, 2, 3, 0]
// CHECK:    %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%[[INPUT]], %[[TRANSPOSE]] : tensor<1x4x4x6xi64>, tensor<2x2x6x8xi64>) outs(%[[INIT]] : tensor<1x2x2x8xi64>) -> tensor<1x2x2x8xi64>
// CHECK:    return %[[CONV]] : tensor<1x2x2x8xi64>
func.func @conv_2d_nhwc_fhwc_i64(%input: tensor<1x4x4x6xi64>, %filter: tensor<8x2x2x6xi64>, %init: tensor<1x2x2x8xi64>) -> tensor<1x2x2x8xi64> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<2> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x4x4x6xi64>, tensor<8x2x2x6xi64>)
    outs (%init: tensor<1x2x2x8xi64>) -> tensor<1x2x2x8xi64>
  return %0 : tensor<1x2x2x8xi64>
}

// CHECK-LABEL: @conv_2d_nhwc_fhwc_i32
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x4x4x6xi32>, %[[FILTER:.+]]: tensor<8x2x2x6xi32>, %[[INIT:.+]]: tensor<1x2x2x8xi32>) -> tensor<1x2x2x8xi32> {
// CHECK-DAG:    %[[NEWF:.+]] = tensor.empty() : tensor<2x2x6x8xi32>
// CHECK:    %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[FILTER]] : tensor<8x2x2x6xi32>) outs(%[[NEWF]] : tensor<2x2x6x8xi32>) permutation = [1, 2, 3, 0]
// CHECK:    %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%[[INPUT]], %[[TRANSPOSE]] : tensor<1x4x4x6xi32>, tensor<2x2x6x8xi32>) outs(%[[INIT]] : tensor<1x2x2x8xi32>) -> tensor<1x2x2x8xi32>
// CHECK:    return %[[CONV]] : tensor<1x2x2x8xi32>
func.func @conv_2d_nhwc_fhwc_i32(%input: tensor<1x4x4x6xi32>, %filter: tensor<8x2x2x6xi32>, %init: tensor<1x2x2x8xi32>) -> tensor<1x2x2x8xi32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<2> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x4x4x6xi32>, tensor<8x2x2x6xi32>)
    outs (%init: tensor<1x2x2x8xi32>) -> tensor<1x2x2x8xi32>
  return %0 : tensor<1x2x2x8xi32>
}

// CHECK-LABEL: @conv_2d_nhwc_fhwc_i16
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x4x4x6xi16>, %[[FILTER:.+]]: tensor<8x2x2x6xi16>, %[[INIT:.+]]: tensor<1x2x2x8xi16>) -> tensor<1x2x2x8xi16> {
// CHECK-DAG:    %[[NEWF:.+]] = tensor.empty() : tensor<2x2x6x8xi16>
// CHECK:    %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[FILTER]] : tensor<8x2x2x6xi16>) outs(%[[NEWF]] : tensor<2x2x6x8xi16>) permutation = [1, 2, 3, 0]
// CHECK:    %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%[[INPUT]], %[[TRANSPOSE]] : tensor<1x4x4x6xi16>, tensor<2x2x6x8xi16>) outs(%[[INIT]] : tensor<1x2x2x8xi16>) -> tensor<1x2x2x8xi16>
// CHECK:    return %[[CONV]] : tensor<1x2x2x8xi16>
func.func @conv_2d_nhwc_fhwc_i16(%input: tensor<1x4x4x6xi16>, %filter: tensor<8x2x2x6xi16>, %init: tensor<1x2x2x8xi16>) -> tensor<1x2x2x8xi16> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<2> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x4x4x6xi16>, tensor<8x2x2x6xi16>)
    outs (%init: tensor<1x2x2x8xi16>) -> tensor<1x2x2x8xi16>
  return %0 : tensor<1x2x2x8xi16>
}

// CHECK-LABEL: @conv_2d_nhwc_fhwc_i8
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x4x4x6xi8>, %[[FILTER:.+]]: tensor<8x2x2x6xi8>, %[[INIT:.+]]: tensor<1x2x2x8xi8>) -> tensor<1x2x2x8xi8> {
// CHECK-DAG:    %[[NEWF:.+]] = tensor.empty() : tensor<2x2x6x8xi8>
// CHECK:    %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[FILTER]] : tensor<8x2x2x6xi8>) outs(%[[NEWF]] : tensor<2x2x6x8xi8>) permutation = [1, 2, 3, 0]
// CHECK:    %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%[[INPUT]], %[[TRANSPOSE]] : tensor<1x4x4x6xi8>, tensor<2x2x6x8xi8>) outs(%[[INIT]] : tensor<1x2x2x8xi8>) -> tensor<1x2x2x8xi8>
// CHECK:    return %[[CONV]] : tensor<1x2x2x8xi8>
func.func @conv_2d_nhwc_fhwc_i8(%input: tensor<1x4x4x6xi8>, %filter: tensor<8x2x2x6xi8>, %init: tensor<1x2x2x8xi8>) -> tensor<1x2x2x8xi8> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<2> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x4x4x6xi8>, tensor<8x2x2x6xi8>)
    outs (%init: tensor<1x2x2x8xi8>) -> tensor<1x2x2x8xi8>
  return %0 : tensor<1x2x2x8xi8>
}

// CHECK-LABEL: @conv_2d_nhwc_fhwc_q
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x4x4x6xf32>, %[[FILTER:.+]]: tensor<8x2x2x6xf32>, %[[INIT:.+]]: tensor<1x2x2x8xf32>, %[[A:.+]]: i32, %[[B:.+]]: i32) -> tensor<1x2x2x8xf32> {
// CHECK-DAG:    %[[NEWF:.+]] = tensor.empty() : tensor<2x2x6x8xf32>
// CHECK:    %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[FILTER]] : tensor<8x2x2x6xf32>) outs(%[[NEWF]] : tensor<2x2x6x8xf32>) permutation = [1, 2, 3, 0]
// CHECK:    %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%[[INPUT]], %[[TRANSPOSE]], %[[A]], %[[B]] : tensor<1x4x4x6xf32>, tensor<2x2x6x8xf32>, i32, i32) outs(%[[INIT]] : tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
// CHECK:    return %[[CONV]] : tensor<1x2x2x8xf32>
  func.func @conv_2d_nhwc_fhwc_q(%input: tensor<1x4x4x6xf32>, %filter: tensor<8x2x2x6xf32>, %init: tensor<1x2x2x8xf32>, %a: i32, %b: i32) -> tensor<1x2x2x8xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc_q {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<2> : tensor<2xi64>}
     ins (%input, %filter, %a, %b: tensor<1x4x4x6xf32>, tensor<8x2x2x6xf32>, i32, i32)
    outs (%init: tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
  return %0 : tensor<1x2x2x8xf32>
}

// CHECK-LABEL: @conv_2d_nhwc_fhwc_f32_unit_stride
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x4x4x6xf32>, %[[FILTER:.+]]: tensor<8x2x2x6xf32>, %[[INIT:.+]]: tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32> {
// CHECK-DAG:    %[[NEWF:.+]] = tensor.empty() : tensor<2x2x6x8xf32>
// CHECK:    %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[FILTER]] : tensor<8x2x2x6xf32>) outs(%[[NEWF]] : tensor<2x2x6x8xf32>) permutation = [1, 2, 3, 0]
// CHECK:    %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%[[INPUT]], %[[TRANSPOSE]] : tensor<1x4x4x6xf32>, tensor<2x2x6x8xf32>) outs(%[[INIT]] : tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32>
// CHECK:    return %[[CONV]] : tensor<1x3x3x8xf32>
func.func @conv_2d_nhwc_fhwc_f32_unit_stride(%input: tensor<1x4x4x6xf32>, %filter: tensor<8x2x2x6xf32>, %init: tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                                              strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x4x4x6xf32>, tensor<8x2x2x6xf32>)
    outs (%init: tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32>
  return %0 : tensor<1x3x3x8xf32>
}

// CHECK-LABEL: @conv_2d_nhwc_fhwc_f32_2_dialation
// CHECK-SAME: (%[[INPUT:.+]]: tensor<1x4x4x6xf32>, %[[FILTER:.+]]: tensor<8x2x2x6xf32>, %[[INIT:.+]]: tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32> {
// CHECK-DAG:    %[[NEWF:.+]] = tensor.empty() : tensor<2x2x6x8xf32>
// CHECK:    %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[FILTER]] : tensor<8x2x2x6xf32>) outs(%[[NEWF]] : tensor<2x2x6x8xf32>) permutation = [1, 2, 3, 0]
// CHECK:    %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%[[INPUT]], %[[TRANSPOSE]] : tensor<1x4x4x6xf32>, tensor<2x2x6x8xf32>) outs(%[[INIT]] : tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
// CHECK:    return %[[CONV]] : tensor<1x2x2x8xf32>
func.func @conv_2d_nhwc_fhwc_f32_2_dialation(%input: tensor<1x4x4x6xf32>, %filter: tensor<8x2x2x6xf32>, %init: tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32> {
  %0 = linalg.conv_2d_nhwc_fhwc {dilations = dense<2> : tensor<2xi64>,
                                              strides = dense<1> : tensor<2xi64>}
     ins (%input, %filter: tensor<1x4x4x6xf32>, tensor<8x2x2x6xf32>)
    outs (%init: tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>
  return %0 : tensor<1x2x2x8xf32>
}

// CHECK-LABEL: @conv_2d_nhwc_fhwc_memref
// CHECK-SAME: (%[[INPUT:.+]]: memref<1x4x4x6xf32>, %[[FILTER:.+]]: memref<8x2x2x6xf32>, %[[INIT:.+]]: memref<1x2x2x8xf32>) -> memref<1x2x2x8xf32> {
// CHECK-DAG:    %[[NEWF:.+]] = memref.alloc() : memref<2x2x6x8xf32>
// CHECK:    linalg.transpose ins(%[[FILTER]] : memref<8x2x2x6xf32>) outs(%[[NEWF]] : memref<2x2x6x8xf32>) permutation = [1, 2, 3, 0]
// CHECK:    linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%[[INPUT]], %[[NEWF]] : memref<1x4x4x6xf32>, memref<2x2x6x8xf32>) outs(%[[INIT]] : memref<1x2x2x8xf32>)
// CHECK:    return %[[INIT]] : memref<1x2x2x8xf32>
func.func @conv_2d_nhwc_fhwc_memref(%input: memref<1x4x4x6xf32>, %filter: memref<8x2x2x6xf32>, %init: memref<1x2x2x8xf32>) -> memref<1x2x2x8xf32> {
  linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>,
                            strides = dense<2> : tensor<2xi64>}
     ins (%input, %filter: memref<1x4x4x6xf32>, memref<8x2x2x6xf32>)
    outs (%init: memref<1x2x2x8xf32>)
  return %init : memref<1x2x2x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_fhwc", "linalg.conv_2d_nhwc_fhwc_q"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.transpose_conv2d %0 : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
