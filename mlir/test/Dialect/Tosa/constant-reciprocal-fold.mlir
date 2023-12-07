// RUN: mlir-opt --split-input-file --tosa-layerwise-constant-fold %s | FileCheck %s

// CHECK-LABEL: @reciprocal_fold_single_valued
func.func @reciprocal_fold_single_valued() -> tensor<f32> {
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}2.5{{0*}}e-01{{.*}}tensor<f32>
  // CHECK-NOT: tosa.reciprocal
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<4.0> : tensor<f32>} : () -> tensor<f32>
  %1 = "tosa.reciprocal"(%0) : (tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: @reciprocal_fold_splat
func.func @reciprocal_fold_splat() -> tensor<12x7xf32> {
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}2.5{{0*}}e-01{{.*}}tensor<12x7xf32>
  // CHECK-NOT: tosa.reciprocal
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<4.0> : tensor<12x7xf32>} : () -> tensor<12x7xf32>
  %1 = "tosa.reciprocal"(%0) : (tensor<12x7xf32>) -> tensor<12x7xf32>
  return %1 : tensor<12x7xf32>
}

// CHECK-LABEL: @reciprocal_div_zero
func.func @reciprocal_div_zero() -> tensor<f32> {
  // 0x7F800000 is the value for +infinity
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}0x7F800000
  // CHECK-NOT: tosa.reciprocal
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %1 = "tosa.reciprocal"(%0) : (tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: @reciprocal_div_neg_zero
func.func @reciprocal_div_neg_zero() -> tensor<f32> {
  // 0xFF800000 is the value for -infinity
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}0xFF800000
  // CHECK-NOT: tosa.reciprocal
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<-0.0> : tensor<f32>} : () -> tensor<f32>
  %1 = "tosa.reciprocal"(%0) : (tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: @reciprocal_div_nan
func.func @reciprocal_div_nan() -> tensor<f32> {
  // 0x7FC00000 is the value for NAN
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}0x7FC00000
  // CHECK-NOT: tosa.reciprocal
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<0x7FC00000> : tensor<f32>} : () -> tensor<f32>
  %1 = "tosa.reciprocal"(%0) : (tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: @reciprocal_div_infinity
func.func @reciprocal_div_infinity() -> tensor<f32> {
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}<0.{{0*}}e+00>
  // CHECK-NOT: tosa.reciprocal
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<0x7F800000> : tensor<f32>} : () -> tensor<f32>
  %1 = "tosa.reciprocal"(%0) : (tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: @reciprocal_div_neg_infinity
func.func @reciprocal_div_neg_infinity() -> tensor<f32> {
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}<-0.{{0*}}e+00>
  // CHECK-NOT: tosa.reciprocal
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<f32>
  %1 = "tosa.reciprocal"(%0) : (tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: @reciprocal_div_underflow
func.func @reciprocal_div_underflow() -> tensor<2xf16> {
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}-0.{{0*}}e+00, 0.{{0*}}e+00
  // CHECK-NOT: tosa.reciprocal
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<[-6.0e+15, 6.0e+15]> : tensor<2xf16>} : () -> tensor<2xf16>
  %1 = "tosa.reciprocal"(%0) : (tensor<2xf16>) -> tensor<2xf16>
  return %1 : tensor<2xf16>
}

// CHECK-LABEL: @reciprocal_div_overflow
func.func @reciprocal_div_overflow() -> tensor<2xf16> {
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}0x7C00, 0xFC00
  // CHECK-NOT: tosa.reciprocal
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<[0.0000001, -0.0000001]> : tensor<2xf16>} : () -> tensor<2xf16>
  %1 = "tosa.reciprocal"(%0) : (tensor<2xf16>) -> tensor<2xf16>
  return %1 : tensor<2xf16>
}

// CHECK-LABEL: @reciprocal_no_fold
// The folding optimization works only intra-procedurally, so we won't be able
// to fold anything here
func.func @reciprocal_no_fold(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: tosa.reciprocal
  // CHECK-NEXT: return
  %0 = "tosa.reciprocal"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @reciprocal_fold
func.func @reciprocal_fold() -> tensor<4x6xf32> {
  // CHECK: [[RES:]] ={{.*}}tosa.const
  // CHECK-SAME{LITERAL}: [[5.68828249, 11.4416485, 1.6880486, 0.680272102, -0.875350117, 0.342313349],
  // CHECK-SAME{LITERAL}:  [-4.81231928, 0.698080301, 0.65432179, -82.6446304, -4.33651352, -0.747551739],
  // CHECK-SAME{LITERAL}:  [-12.4378109, 13.140605, 1.89501607, 0.885582745, 4.08830738, 1.4396776],
  // CHECK-SAME{LITERAL}:  [2.02880907, -1.53280187, 0.552730501, 7.15819644, 0.64495325, -0.973709881]]
  // CHECK-NOT: tosa.reciprocal
  // CHECK: return [[RES]]
  %0 = "tosa.const"() { value = dense<[
                        [ 0.1758,  0.0874,  0.5924,  1.4700, -1.1424,  2.9213],
                        [-0.2078,  1.4325,  1.5283, -0.0121, -0.2306, -1.3377],
                        [-0.0804,  0.0761,  0.5277,  1.1292,  0.2446,  0.6946],
                        [ 0.4929, -0.6524,  1.8092,  0.1397,  1.5505, -1.0270]]>
                        : tensor<4x6xf32>
                      } : () -> tensor<4x6xf32>
  %1 = "tosa.reciprocal"(%0) : (tensor<4x6xf32>) -> tensor<4x6xf32>
  return %1 : tensor<4x6xf32>
}

// CHECK-LABEL: @reciprocal_of_const_sparse
// Sparse tensors are currently not supported
func.func @reciprocal_of_const_sparse() -> tensor<32xbf16> {
  // CHECK: tosa.const
  // CHECK: tosa.reciprocal
    %0 = "tosa.const"() { value = sparse<
          [[0], [3], [11], [17], [20], [23], [25], [30], [31]],
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]>
          : tensor<32xbf16> } : () -> tensor<32xbf16>
    %1 = "tosa.reciprocal"(%0) : (tensor<32xbf16>) -> tensor<32xbf16>
    return %1 : tensor<32xbf16>
}
