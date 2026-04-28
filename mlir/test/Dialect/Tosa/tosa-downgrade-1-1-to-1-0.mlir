// RUN: mlir-opt --split-input-file --tosa-downgrade-1-1-to-1-0 %s | FileCheck %s

// CHECK-LABEL: @test_max_pool
// CHECK: %[[POOL:.+]] = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, nan_mode = IGNORE, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
// CHECK: return %[[POOL]]
func.func @test_max_pool(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  %kernel = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %stride = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %pad = tosa.const_shape {values = dense<[0, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %0 = tosa.max_pool2d_adaptive %arg0, %kernel, %stride, %pad {nan_mode = IGNORE} :
         (tensor<1x32x32x8xf32>, !tosa.shape<2>, !tosa.shape<2>, !tosa.shape<4>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----

// CHECK-LABEL: @test_avg_pool
// CHECK: %[[POOL:.+]] = tosa.avg_pool2d %arg0, %{{.*}}, %{{.*}} {acc_type = f32, kernel = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x7x7x9xf32>
// CHECK: return %[[POOL]]
func.func @test_avg_pool(%arg0: tensor<1x7x7x9xf32>, %input_zp: tensor<1xf32>, %output_zp: tensor<1xf32>) -> tensor<1x7x7x9xf32> {
  %kernel = tosa.const_shape {values = dense<[3, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %stride = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %pad = tosa.const_shape {values = dense<[1, 1, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %0 = tosa.avg_pool2d_adaptive %arg0, %input_zp, %output_zp, %kernel, %stride, %pad {acc_type = f32} :
         (tensor<1x7x7x9xf32>, tensor<1xf32>, tensor<1xf32>, !tosa.shape<2>, !tosa.shape<2>, !tosa.shape<4>) -> tensor<1x7x7x9xf32>
  return %0 : tensor<1x7x7x9xf32>
}

// -----

// CHECK-LABEL: @test_bool_to_fp32
// CHECK: %[[BOOL_TO_I8:.+]] = tosa.cast %arg0 : (tensor<13x21x3xi1>) -> tensor<13x21x3xi8>
// CHECK: %[[I8_TO_F32:.+]] = tosa.cast %[[BOOL_TO_I8]] : (tensor<13x21x3xi8>) -> tensor<13x21x3xf32>
// CHECK: return %[[I8_TO_F32]]
func.func @test_bool_to_fp32(%arg0: tensor<13x21x3xi1>) -> tensor<13x21x3xf32> {
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi1>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: @test_bool_to_fp32_unranked
// CHECK: %[[BOOL_TO_I8:.+]] = tosa.cast %arg0 : (tensor<*xi1>) -> tensor<*xi8>
// CHECK: %[[I8_TO_F32:.+]] = tosa.cast %[[BOOL_TO_I8]] : (tensor<*xi8>) -> tensor<*xf32>
// CHECK: return %[[I8_TO_F32]]
func.func @test_bool_to_fp32_unranked(%arg0: tensor<*xi1>) -> tensor<*xf32> {
  %0 = tosa.cast %arg0 : (tensor<*xi1>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_fp32_to_bool_ranked_dynamic
// CHECK: %[[FP32_TO_I8:.+]] = tosa.cast %arg0 : (tensor<13x?x3xf32>) -> tensor<13x?x3xi8>
// CHECK: %[[I8_TO_BOOL:.+]] = tosa.cast %[[FP32_TO_I8]] : (tensor<13x?x3xi8>) -> tensor<13x?x3xi1>
// CHECK: return %[[I8_TO_BOOL]]
func.func @test_fp32_to_bool_ranked_dynamic(%arg0: tensor<13x?x3xf32>) -> tensor<13x?x3xi1> {
  %0 = tosa.cast %arg0 : (tensor<13x?x3xf32>) -> tensor<13x?x3xi1>
  return %0 : tensor<13x?x3xi1>
}

// -----

// CHECK-LABEL: @test_unranked_fp32_to_bool
// CHECK: %[[FP32_TO_I8:.+]] = tosa.cast %arg0 : (tensor<*xf32>) -> tensor<*xi8>
// CHECK: %[[I8_TO_BOOL:.+]] = tosa.cast %[[FP32_TO_I8]] : (tensor<*xi8>) -> tensor<*xi1>
// CHECK: return %[[I8_TO_BOOL]]
func.func @test_unranked_fp32_to_bool(%arg0: tensor<*xf32>) -> tensor<*xi1> {
  %0 = tosa.cast %arg0 : (tensor<*xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>
}

// -----

// CHECK-LABEL: @test_preserve_bool_to_i8
// CHECK: %[[CAST:.+]] = tosa.cast %arg0 : (tensor<13x21x3xi1>) -> tensor<13x21x3xi8>
// CHECK: return %[[CAST]]
func.func @test_preserve_bool_to_i8(%arg0: tensor<13x21x3xi1>) -> tensor<13x21x3xi8> {
  %0 = tosa.cast %arg0 : (tensor<13x21x3xi1>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}

// -----

// CHECK-LABEL: @test_gather_bool_i32
// CHECK: %[[VALUES_TO_I8:.+]] = tosa.cast %arg0 : (tensor<13x21x3xi1>) -> tensor<13x21x3xi8>
// CHECK: %[[GATHER_I8:.+]] = tosa.gather %[[VALUES_TO_I8]], %arg1 : (tensor<13x21x3xi8>, tensor<13x26xi32>) -> tensor<13x26x3xi8>
// CHECK: %[[I8_TO_BOOL:.+]] = tosa.cast %[[GATHER_I8]] : (tensor<13x26x3xi8>) -> tensor<13x26x3xi1>
// CHECK: return %[[I8_TO_BOOL]]
func.func @test_gather_bool_i32(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x26xi32>) -> tensor<13x26x3xi1> {
  %0 = tosa.gather %arg0, %arg1 : (tensor<13x21x3xi1>, tensor<13x26xi32>) -> tensor<13x26x3xi1>
  return %0 : tensor<13x26x3xi1>
}

// -----

// CHECK-LABEL: @test_preserve_gather_bool_i64
// CHECK: %[[GATHER:.+]] = tosa.gather %arg0, %arg1 : (tensor<13x21x3xi1>, tensor<13x26xi64>) -> tensor<13x26x3xi1>
// CHECK: return %[[GATHER]]
func.func @test_preserve_gather_bool_i64(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x26xi64>) -> tensor<13x26x3xi1> {
  %0 = tosa.gather %arg0, %arg1 : (tensor<13x21x3xi1>, tensor<13x26xi64>) -> tensor<13x26x3xi1>
  return %0 : tensor<13x26x3xi1>
}

// -----

// CHECK-LABEL: @test_preserve_gather_i8_i32
// CHECK: %[[GATHER:.+]] = tosa.gather %arg0, %arg1 : (tensor<13x21x3xi8>, tensor<13x26xi32>) -> tensor<13x26x3xi8>
// CHECK: return %[[GATHER]]
func.func @test_preserve_gather_i8_i32(%arg0: tensor<13x21x3xi8>, %arg1: tensor<13x26xi32>) -> tensor<13x26x3xi8> {
  %0 = tosa.gather %arg0, %arg1 : (tensor<13x21x3xi8>, tensor<13x26xi32>) -> tensor<13x26x3xi8>
  return %0 : tensor<13x26x3xi8>
}

// -----

// CHECK-LABEL: @test_scatter_bool_i32
// CHECK: %[[VALUES_IN_TO_I8:.+]] = tosa.cast %arg0 : (tensor<13x52x3xi1>) -> tensor<13x52x3xi8>
// CHECK: %[[INPUT_TO_I8:.+]] = tosa.cast %arg2 : (tensor<13x26x3xi1>) -> tensor<13x26x3xi8>
// CHECK: %[[SCATTER_I8:.+]] = tosa.scatter %[[VALUES_IN_TO_I8]], %arg1, %[[INPUT_TO_I8]] : (tensor<13x52x3xi8>, tensor<13x26xi32>, tensor<13x26x3xi8>) -> tensor<13x52x3xi8>
// CHECK: %[[I8_TO_BOOL:.+]] = tosa.cast %[[SCATTER_I8]] : (tensor<13x52x3xi8>) -> tensor<13x52x3xi1>
// CHECK: return %[[I8_TO_BOOL]]
func.func @test_scatter_bool_i32(%arg0: tensor<13x52x3xi1>, %arg1: tensor<13x26xi32>, %arg2: tensor<13x26x3xi1>) -> tensor<13x52x3xi1> {
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x52x3xi1>, tensor<13x26xi32>, tensor<13x26x3xi1>) -> tensor<13x52x3xi1>
  return %0 : tensor<13x52x3xi1>
}

// -----

// CHECK-LABEL: @test_preserve_scatter_bool_i64
// CHECK: %[[SCATTER:.+]] = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x52x3xi1>, tensor<13x26xi64>, tensor<13x26x3xi1>) -> tensor<13x52x3xi1>
// CHECK: return %[[SCATTER]]
func.func @test_preserve_scatter_bool_i64(%arg0: tensor<13x52x3xi1>, %arg1: tensor<13x26xi64>, %arg2: tensor<13x26x3xi1>) -> tensor<13x52x3xi1> {
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x52x3xi1>, tensor<13x26xi64>, tensor<13x26x3xi1>) -> tensor<13x52x3xi1>
  return %0 : tensor<13x52x3xi1>
}

// -----

// CHECK-LABEL: @test_preserve_scatter_i8_i32
// CHECK: %[[SCATTER:.+]] = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x52x3xi8>, tensor<13x26xi32>, tensor<13x26x3xi8>) -> tensor<13x52x3xi8>
// CHECK: return %[[SCATTER]]
func.func @test_preserve_scatter_i8_i32(%arg0: tensor<13x52x3xi8>, %arg1: tensor<13x26xi32>, %arg2: tensor<13x26x3xi8>) -> tensor<13x52x3xi8> {
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x52x3xi8>, tensor<13x26xi32>, tensor<13x26x3xi8>) -> tensor<13x52x3xi8>
  return %0 : tensor<13x52x3xi8>
}

// -----

// CHECK-LABEL: @test_preserve_non_const_adaptive
// CHECK: tosa.max_pool2d_adaptive
func.func @test_preserve_non_const_adaptive(%arg0: tensor<1x?x?x8xf32>) -> tensor<1x?x?x8xf32> {
  %dim1 = tosa.dim %arg0 {axis = 1 : i32} : (tensor<1x?x?x8xf32>) -> !tosa.shape<1>
  %dim2 = tosa.dim %arg0 {axis = 2 : i32} : (tensor<1x?x?x8xf32>) -> !tosa.shape<1>
  %kernel = tosa.concat_shape %dim1, %dim2 : (!tosa.shape<1>, !tosa.shape<1>) -> !tosa.shape<2>
  %stride = tosa.const_shape {values = dense<[1, 1]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %pad = tosa.const_shape {values = dense<[0, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %0 = tosa.max_pool2d_adaptive %arg0, %kernel, %stride, %pad {nan_mode = IGNORE} :
          (tensor<1x?x?x8xf32>, !tosa.shape<2>, !tosa.shape<2>, !tosa.shape<4>) -> tensor<1x?x?x8xf32>
  return %0 : tensor<1x?x?x8xf32>
}
