// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-linalg))" %s -verify-diagnostics

// CHECK-LABEL: @avg_pool2d_with_unsupported_quant_type
func.func @avg_pool2d_with_unsupported_quant_type(%arg0: tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>) -> tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>> {
  // expected-error@+1 {{failed to legalize operation 'tosa.avg_pool2d'}}
  %0 = "tosa.avg_pool2d"(%arg0) {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>) -> tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>
  return %0 : tensor<1x7x7x9x!quant.uniform<i8:f32, 0.01>>
}

// -----

// CHECK-LABEL: @tensor_with_unknown_rank
func.func @tensor_with_unknown_rank(%arg0: tensor<*xi8>) -> tensor<*xi8> {
  // expected-error@+1 {{failed to legalize operation 'tosa.abs'}}
  %0 = "tosa.abs"(%arg0) : (tensor<*xi8>) -> tensor<*xi8>
  return %0 : tensor<*xi8>
}
