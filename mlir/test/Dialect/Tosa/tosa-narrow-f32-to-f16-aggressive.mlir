// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-narrow-f32-to-f16="aggressive-rewrite=1" %s | FileCheck %s --allow-unused-prefixes --check-prefixes=COMMON,DEFAULT
// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-narrow-f32-to-f16="aggressive-rewrite=1 convert-function-boundaries=1" %s | FileCheck %s --allow-unused-prefixes --check-prefixes=COMMON,FUNCBOUND
// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-narrow-f32-to-f16="aggressive-rewrite=1 convert-accumulator-type=1" %s | FileCheck %s --allow-unused-prefixes --check-prefixes=COMMON,CONVERT-ACC

// -----

// CHECK-LABEL: test_f32_add
// DEFAULT: %[[IN0:.*]]: tensor<13x21x1xf32>, %[[IN1:.*]]: tensor<13x21x3xf32>
// FUNCBOUND: %[[IN0:.*]]: tensor<13x21x1xf16>, %[[IN1:.*]]: tensor<13x21x3xf16>
func.func @test_f32_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // DEFAULT-DAG: %[[CAST0:.*]] = tosa.cast %[[IN0]] : (tensor<13x21x1xf32>) -> tensor<13x21x1xf16>
  // DEFAULT-DAG: %[[CAST1:.*]] = tosa.cast %[[IN1]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xf16>
  // COMMON: %[[ADD:.*]] = tosa.add %{{.*}}, %{{.*}} : (tensor<13x21x1xf16>, tensor<13x21x3xf16>) -> tensor<13x21x3xf16>
  %0 = tosa.add %arg0, %arg1 : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[ADD]] : (tensor<13x21x3xf16>) -> tensor<13x21x3xf32>
  // DEFAULT: return %[[OUT]] : tensor<13x21x3xf32>
  // FUNCBOUND: return %[[ADD]] : tensor<13x21x3xf16>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_f32_regions
// DEFAULT: %[[IN0:.*]]: tensor<1xf32>, %[[IN1:.*]]: tensor<1xf32>
func.func @test_f32_regions(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<i1>) -> tensor<1xf32> {
  // DEFAULT-DAG: %[[CAST0:.*]] = tosa.cast %[[IN0]] : (tensor<1xf32>) -> tensor<1xf16>
  // DEFAULT-DAG: %[[CAST1:.*]] = tosa.cast %[[IN1]] : (tensor<1xf32>) -> tensor<1xf16>
  // COMMON: %[[IF:.*]] = tosa.cond_if %arg2 : tensor<i1> -> tensor<1xf16>
  %0 = tosa.cond_if %arg2 : tensor<i1> -> tensor<1xf32> {
    // COMMON: %[[ADD:.*]] = tosa.add %{{.*}}, %{{.*}} : (tensor<1xf16>, tensor<1xf16>) -> tensor<1xf16>
    %1 = tosa.add %arg0, %arg1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    tosa.yield %1 : tensor<1xf32>
  } else {
    // COMMON: %[[SUB:.*]] = tosa.sub %{{.*}}, %{{.*}} : (tensor<1xf16>, tensor<1xf16>) -> tensor<1xf16>
    %1 = tosa.sub %arg0, %arg1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    tosa.yield %1 : tensor<1xf32>
  }
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[IF]] : (tensor<1xf16>) -> tensor<1xf32>
  // DEFAULT: return %[[OUT]] : tensor<1xf32>
  // FUNCBOUND: return %[[IF]] : tensor<1xf16>
  return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: test_convert_input_parameters
// DEFAULT: %[[IN:.*]]: tensor<1x3xf32>
// FUNCBOUND: %[[IN:.*]]: tensor<1x3xf16>
func.func @test_convert_input_parameters(%arg0: tensor<1x3xf32>) -> tensor<1x3xf16> {
  // DEFAULT: %[[CAST_IN:.*]] = tosa.cast %[[IN]] : (tensor<1x3xf32>) -> tensor<1x3xf16>
  // DEFAULT: %[[IDENTITY:.*]] = tosa.identity %[[CAST_IN]] : (tensor<1x3xf16>) -> tensor<1x3xf16>
  // FUNCBOUND: %[[IDENTITY:.*]] = tosa.identity %[[IN]] : (tensor<1x3xf16>) -> tensor<1x3xf16>
  %0 = tosa.identity %arg0 : (tensor<1x3xf32>) -> tensor<1x3xf32>
  // COMMON: %[[TO_F16:.*]] = tosa.cast %{{.*}} {{.*}}: (tensor<1x3xf16>) -> tensor<1x3xf16>
  %1 = tosa.cast %0 {input_unsigned = false} : (tensor<1x3xf32>) -> tensor<1x3xf16>
  // DEFAULT: return %[[TO_F16]] : tensor<1x3xf16>
  // FUNCBOUND: return %[[TO_F16]] : tensor<1x3xf16>
  return %1 : tensor<1x3xf16>
}

// -----

// CHECK-LABEL: test_f32_const
func.func @test_f32_const() -> tensor<2xf32> {
  // COMMON: %[[CONST:.*]] = "tosa.const"() <{values = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf16>}> : () -> tensor<2xf16>
  %0 = "tosa.const"() <{values = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>}> : () -> tensor<2xf32>
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[CONST]] : (tensor<2xf16>) -> tensor<2xf32>
  // DEFAULT: return %[[OUT]] : tensor<2xf32>
  // FUNCBOUND: return %[[CONST]] : tensor<2xf16>
  return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: test_f32_accumulator
func.func @test_f32_accumulator(%arg0: tensor<1x4x4x1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1x3x3x1xf32> {
  // COMMON: tosa.avg_pool2d
  // DEFAULT-SAME: acc_type = f32
  // FUNCBOUND-SAME: acc_type = f32
  // CONVERT-ACC-SAME: acc_type = f16
  // COMMON-SAME: (tensor<1x4x4x1xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x3x3x1xf16>
  %0 = tosa.avg_pool2d %arg0, %arg1, %arg2 {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x3x3x1xf32>
  return %0 : tensor<1x3x3x1xf32>
}

// -----

// CHECK-LABEL: test_f32_conv2d_accumulator
func.func @test_f32_conv2d_accumulator(%arg0: tensor<1x4x4x1xf32>, %arg1: tensor<1x2x2x1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<1x3x3x1xf32> {
  // COMMON: tosa.conv2d
  // DEFAULT-SAME: acc_type = f32
  // FUNCBOUND-SAME: acc_type = f32
  // CONVERT-ACC-SAME: acc_type = f16
  // COMMON-SAME: (tensor<1x4x4x1xf16>, tensor<1x2x2x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x3x3x1xf16>
  %0 = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x1xf32>, tensor<1x2x2x1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x3x3x1xf32>
  return %0 : tensor<1x3x3x1xf32>
}

// -----

// CHECK-LABEL: test_accumulator_only_f32
func.func @test_accumulator_only_f32(%arg0: tensor<1x4x4x1xf16>, %arg1: tensor<1xf16>, %arg2: tensor<1xf16>) -> tensor<1x3x3x1xf16> {
  // COMMON: tosa.avg_pool2d
  // DEFAULT-SAME: acc_type = f32
  // FUNCBOUND-SAME: acc_type = f32
  // CONVERT-ACC-SAME: acc_type = f16
  %0 = tosa.avg_pool2d %arg0, %arg1, %arg2 {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x1xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x3x3x1xf16>
  return %0 : tensor<1x3x3x1xf16>
}

// -----

// CHECK-LABEL: test_dense_resource_f32
func.func @test_dense_resource_f32() -> tensor<1x2xf32> {
  // COMMON: %[[CONST:.*]] = "tosa.const"() <{values = dense_resource<resource> : tensor<1x2xf16>}> : () -> tensor<1x2xf16>
  %0 = "tosa.const"() <{values = dense_resource<resource> : tensor<1x2xf32>}> : () -> tensor<1x2xf32>
  // DEFAULT: %[[OUT_CAST:.*]] = tosa.cast %[[CONST]] : (tensor<1x2xf16>) -> tensor<1x2xf32>
  // DEFAULT: return %[[OUT_CAST]] : tensor<1x2xf32>
  // FUNCBOUND: return %[[CONST]] : tensor<1x2xf16>
  return %0 : tensor<1x2xf32>
}
{-#
  dialect_resources: {
    builtin: {
      // COMMON: resource: "0x0200000000000000"
      resource: "0x040000000000000000000000"
    }
  }
#-}
