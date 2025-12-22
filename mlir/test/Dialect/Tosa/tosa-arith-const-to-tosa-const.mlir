// RUN: mlir-opt %s --tosa-arith-const-to-tosa-const --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @rewrite_f32_tensor
// CHECK: %[[CST:.*]] = "tosa.const"() <{values = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>}> : () -> tensor<2xf32>
// CHECK: return %[[CST]]
func.func @rewrite_f32_tensor() -> tensor<2xf32> {
  %c = arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  return %c : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @rewrite_i32_tensor
// CHECK: %[[CST:.*]] = "tosa.const"() <{values = dense<[1, 0, -1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK: return %[[CST]]
func.func @rewrite_i32_tensor() -> tensor<3xi32> {
  %c = arith.constant dense<[1, 0, -1]> : tensor<3xi32>
  return %c : tensor<3xi32>
}

// -----

// CHECK-LABEL: func.func @rewrite_i1_tensor
// CHECK: %[[CST:.*]] = "tosa.const"() <{values = dense<[true, false]> : tensor<2xi1>}> : () -> tensor<2xi1>
func.func @rewrite_i1_tensor() -> tensor<2xi1> {
  %c = arith.constant dense<[true, false]> : tensor<2xi1>
  return %c : tensor<2xi1>
}

// -----

// CHECK-LABEL: func.func @rewrite_rank0_tensor
// CHECK: %[[CST:.*]] = "tosa.const"() <{values = dense<1.234500e+00> : tensor<f32>}> : () -> tensor<f32>
func.func @rewrite_rank0_tensor() -> tensor<f32> {
  %c = arith.constant dense<1.234500e+00> : tensor<f32>
  return %c : tensor<f32>
}

// -----

// CHECK-LABEL: func.func @preserve_scalar_i32
// CHECK: %[[CST:.*]] = arith.constant 42 : i32
func.func @preserve_scalar_i32() -> i32 {
  %c = arith.constant 42 : i32
  return %c : i32
}

// -----

// CHECK-LABEL: func.func @preserve_index_tensor
// CHECK: %[[CST:.*]] = arith.constant dense<[0, 1]> : tensor<2xindex>
func.func @preserve_index_tensor() -> tensor<2xindex> {
  %c = arith.constant dense<[0, 1]> : tensor<2xindex>
  return %c : tensor<2xindex>
}

// -----

// CHECK-LABEL: func.func @rewrite_resource_tensor
// CHECK: %[[CST:.*]] = "tosa.const"() <{values = dense_resource<blob1> : tensor<4xf32>}> : () -> tensor<4xf32>
func.func @rewrite_resource_tensor() -> tensor<4xf32> {
  %c = arith.constant dense_resource<"blob1"> : tensor<4xf32>
  return %c : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @rewrite_quant_tensor
// CHECK: %[[CST:.*]] = "tosa.const"() <{values = dense<[10, 20]> : tensor<2xui8>}> : () -> tensor<2xui8>
func.func @rewrite_quant_tensor() -> tensor<2xui8> {
  %c = arith.constant dense<[10, 20]> : tensor<2xui8>
  return %c : tensor<2xui8>
}

// -----

// CHECK-LABEL: func.func @rewrite_quant_uniform_tensor
// CHECK: %[[CST:.*]] = "tosa.const"() <{values = dense<["10", "20"]> : tensor<2x!quant.uniform<i8:f32, 5.000000e-01>>}> : () -> tensor<2x!quant.uniform<i8:f32, 5.000000e-01>>
func.func @rewrite_quant_uniform_tensor() -> tensor<2x!quant.uniform<i8:f32, 0.5:0>> {
  %c = arith.constant dense<["10", "20"]> : tensor<2x!quant.uniform<i8:f32, 0.5:0>>
  return %c : tensor<2x!quant.uniform<i8:f32, 0.5:0>>
}

// -----

// CHECK-LABEL: func.func @rewrite_fp8_tensor
// CHECK: %[[CST:.*]] = "tosa.const"() <{values = dense<[1.000000e+00, -5.000000e-01]> : tensor<2xf8E4M3FN>}> : () -> tensor<2xf8E4M3FN>
func.func @rewrite_fp8_tensor() -> tensor<2xf8E4M3FN> {
  %c = arith.constant dense<[1.0, -0.5]> : tensor<2xf8E4M3FN>
  return %c : tensor<2xf8E4M3FN>
}

// -----

// CHECK-LABEL: func.func @rewrite_mxint8_tensor
// CHECK: %[[CST:.*]] = "tosa.const"() <{values = dense<["0x00", "0x7F"]> : tensor<2x!tosa.mxint8>}> : () -> tensor<2x!tosa.mxint8>
func.func @rewrite_mxint8_tensor() -> tensor<2x!tosa.mxint8> {
  %c = arith.constant dense<["0x00", "0x7F"]> : tensor<2x!tosa.mxint8>
  return %c : tensor<2x!tosa.mxint8>
}
