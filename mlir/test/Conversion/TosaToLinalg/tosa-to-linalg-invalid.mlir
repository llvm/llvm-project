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

// -----

// CHECK-LABEL: @unranked_add
func.func @unranked_add(%arg0 : tensor<10x10xf32> , %arg1 : tensor<10x10xf32>, %arg2 : tensor<*xf32>) -> (tensor<10x10xf32>) {
  // expected-error@+3 {{failed to legalize operation 'tosa.add'}}
  %reduce = tosa.reduce_max %arg0 {axis = 1 : i32} : (tensor<10x10xf32>) -> tensor<10x1xf32>
  %1 = tosa.add %reduce, %arg1 : (tensor<10x1xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = tosa.add %1, %arg2 : (tensor<10x10xf32>, tensor<*xf32>) -> tensor<*xf32>
  %2 = tosa.reshape %0 {new_shape = array<i64: 10, 10>} : (tensor<*xf32>) -> tensor<10x10xf32>
  return %2 : tensor<10x10xf32>
}
