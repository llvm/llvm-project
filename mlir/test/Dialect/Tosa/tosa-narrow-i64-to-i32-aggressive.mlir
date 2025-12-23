// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-narrow-i64-to-i32="aggressive-rewrite=1" %s | FileCheck %s --allow-unused-prefixes --check-prefixes=COMMON,DEFAULT
// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-narrow-i64-to-i32="aggressive-rewrite=1 convert-function-boundaries=1" %s | FileCheck %s --allow-unused-prefixes --check-prefixes=COMMON,FUNCBOUND

// CHECK-LABEL: test_i64_argmax_large_axis_dim
func.func @test_i64_argmax_large_axis_dim(%arg0: tensor<1x513x513x2147483650xi8>) -> tensor<1x513x513xi64> {
  // DEFAULT: tosa.argmax %arg0 {axis = 3 : i32} : (tensor<1x513x513x2147483650xi8>) -> tensor<1x513x513xi32>
  %0 = tosa.argmax %arg0 {axis = 3 : i32} : (tensor<1x513x513x2147483650xi8>) -> tensor<1x513x513xi64>
  return %0 : tensor<1x513x513xi64>
}

// -----

// CHECK-LABEL: test_convert_input_parameters
// DEFAULT: %[[IN:.*]]: tensor<1x513x513x3xi64>
// FUNCBOUND: %[[IN:.*]]: tensor<1x513x513x3xi32>
func.func @test_convert_input_parameters(%arg0: tensor<1x513x513x3xi64>) -> tensor<1x513x513x3xf32> {
  // DEFAULT: %[[FUNC_BOUND_CAST:.*]] = tosa.cast %[[IN]] : (tensor<1x513x513x3xi64>) -> tensor<1x513x513x3xi32>
  // DEFAULT: %[[CAST1:.*]] = tosa.cast %[[FUNC_BOUND_CAST]] : (tensor<1x513x513x3xi32>) -> tensor<1x513x513x3xi32>
  // FUNCBOUND: %[[CAST1:.*]] = tosa.cast %[[IN]] : (tensor<1x513x513x3xi32>) -> tensor<1x513x513x3xi32>
  %0 = tosa.cast %arg0 : (tensor<1x513x513x3xi64>) -> tensor<1x513x513x3xi32>

  // COMMON: %[[CAST2:.*]] = tosa.cast %[[CAST1]] : (tensor<1x513x513x3xi32>) -> tensor<1x513x513x3xf32>
  %1 = tosa.cast %0 : (tensor<1x513x513x3xi32>) -> tensor<1x513x513x3xf32>
  return %1 : tensor<1x513x513x3xf32>
}

// -----

// CHECK-LABEL: test_add
// DEFAULT: %[[IN0:.*]]: tensor<13x21x1xi64>, %[[IN1:.*]]: tensor<13x21x3xi64>
// FUNCBOUND: %[[IN0:.*]]: tensor<13x21x1xi32>, %[[IN1:.*]]: tensor<13x21x3xi32>
func.func @test_add(%arg0: tensor<13x21x1xi64>, %arg1: tensor<13x21x3xi64>) -> tensor<13x21x3xi64> {
  // DEFAULT-DAG: %[[FUNC_BOUND_CAST0:.*]] = tosa.cast %[[IN0]] : (tensor<13x21x1xi64>) -> tensor<13x21x1xi32>
  // DEFAULT-DAG: %[[FUNC_BOUND_CAST1:.*]] = tosa.cast %[[IN1]] : (tensor<13x21x3xi64>) -> tensor<13x21x3xi32>
  // DEFAULT: %[[ADD:.*]] = tosa.add %[[FUNC_BOUND_CAST0]], %[[FUNC_BOUND_CAST1]] : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  // DEFAULT: %[[CAST:.*]] = tosa.cast %[[ADD]] : (tensor<13x21x3xi32>) -> tensor<13x21x3xi64>
  // DEFAULT: return %[[CAST]] : tensor<13x21x3xi64>
  // FUNCBOUND: %[[ADD:.*]] = tosa.add %[[IN0]], %[[IN1]] : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  // FUNCBOUND: return %[[ADD]] : tensor<13x21x3xi32>
  %0 = tosa.add %arg0, %arg1 : (tensor<13x21x1xi64>, tensor<13x21x3xi64>) -> tensor<13x21x3xi64>
  return %0 : tensor<13x21x3xi64>
}

// -----

// CHECK-LABEL: test_regions
// DEFAULT: %[[IN0:.*]]: tensor<i64>, %[[IN1:.*]]: tensor<i64>
func.func @test_regions(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i1>) -> tensor<i64> {
  // DEFAULT-DAG: %[[CAST0:.*]] = tosa.cast %[[IN0]] : (tensor<i64>) -> tensor<i32>
  // DEFAULT-DAG: %[[CAST1:.*]] = tosa.cast %[[IN1]] : (tensor<i64>) -> tensor<i32>
  // COMMON: %[[IF_RESULT:.*]] = tosa.cond_if
  %0 = tosa.cond_if %arg2 : tensor<i1> -> (tensor<i64>) {
    // DEFAULT: %[[ADD:.*]] = tosa.add %[[CAST0]], %[[CAST1]] : (tensor<i32>, tensor<i32>) -> tensor<i32>
    // FUNCBOUND: %[[ADD:.*]] = tosa.add %[[IN0]], %[[IN1]] : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = tosa.add %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    // COMMON: tosa.yield %[[ADD]] : tensor<i32>
    tosa.yield %1 : tensor<i64>
  } else {
    // DEFAULT: %[[SUB:.*]] = tosa.sub %[[CAST0]], %[[CAST1]] : (tensor<i32>, tensor<i32>) -> tensor<i32>
    // FUNCBOUND: %[[SUB:.*]] = tosa.sub %[[IN0]], %[[IN1]] : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = tosa.sub %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    // COMMON: tosa.yield %[[SUB]] : tensor<i32>
    tosa.yield %1 : tensor<i64>
  }
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[IF_RESULT]] : (tensor<i32>) -> tensor<i64>
  // DEFAULT: return %[[OUT]] : tensor<i64>
  // FUNCBOUND: return %[[IF_RESULT]] : tensor<i32>
  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: test_const
func.func @test_const() -> tensor<2xi64> {
  // COMMON: %[[CONST:.*]] = "tosa.const"() <{values = dense<[1, 2]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %0 = "tosa.const"() <{values = dense<[1, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[CONST]] : (tensor<2xi32>) -> tensor<2xi64>
  // DEFAULT: return %[[OUT]] : tensor<2xi64>
  // FUNCBOUND: return %[[CONST]] : tensor<2xi32>
  return %0 : tensor<2xi64>
}
