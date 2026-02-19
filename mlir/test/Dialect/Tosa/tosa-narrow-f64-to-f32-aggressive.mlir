// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-narrow-f64-to-f32="aggressive-rewrite=1" %s | FileCheck %s --allow-unused-prefixes --check-prefixes=COMMON,DEFAULT
// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-narrow-f64-to-f32="aggressive-rewrite=1 convert-function-boundaries=1" %s | FileCheck %s --allow-unused-prefixes --check-prefixes=COMMON,FUNCBOUND

// -----

// CHECK-LABEL: test_f64_add
// DEFAULT: %[[IN0:.*]]: tensor<13x21x1xf64>, %[[IN1:.*]]: tensor<13x21x3xf64>
// FUNCBOUND: %[[IN0:.*]]: tensor<13x21x1xf32>, %[[IN1:.*]]: tensor<13x21x3xf32>
func.func @test_f64_add(%arg0: tensor<13x21x1xf64>, %arg1: tensor<13x21x3xf64>) -> tensor<13x21x3xf64> {
  // DEFAULT-DAG: %[[CAST0:.*]] = tosa.cast %[[IN0]] : (tensor<13x21x1xf64>) -> tensor<13x21x1xf32>
  // DEFAULT-DAG: %[[CAST1:.*]] = tosa.cast %[[IN1]] : (tensor<13x21x3xf64>) -> tensor<13x21x3xf32>
  // COMMON: %[[ADD:.*]] = tosa.add %{{.*}}, %{{.*}} : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = tosa.add %arg0, %arg1 : (tensor<13x21x1xf64>, tensor<13x21x3xf64>) -> tensor<13x21x3xf64>
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[ADD]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xf64>
  // DEFAULT: return %[[OUT]] : tensor<13x21x3xf64>
  // FUNCBOUND: return %[[ADD]] : tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf64>
}

// -----

// CHECK-LABEL: test_f64_regions
// DEFAULT: %[[IN0:.*]]: tensor<1xf64>, %[[IN1:.*]]: tensor<1xf64>
func.func @test_f64_regions(%arg0: tensor<1xf64>, %arg1: tensor<1xf64>, %arg2: tensor<i1>) -> tensor<1xf64> {
  // DEFAULT-DAG: %[[CAST0:.*]] = tosa.cast %[[IN0]] : (tensor<1xf64>) -> tensor<1xf32>
  // DEFAULT-DAG: %[[CAST1:.*]] = tosa.cast %[[IN1]] : (tensor<1xf64>) -> tensor<1xf32>
  // COMMON: %[[IF:.*]] = tosa.cond_if %arg2 : tensor<i1> -> tensor<1xf32>
  %0 = tosa.cond_if %arg2 : tensor<i1> -> tensor<1xf64> {
    // COMMON: %[[ADD:.*]] = tosa.add %{{.*}}, %{{.*}} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %1 = tosa.add %arg0, %arg1 : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    tosa.yield %1 : tensor<1xf64>
  } else {
    // COMMON: %[[SUB:.*]] = tosa.sub %{{.*}}, %{{.*}} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %1 = tosa.sub %arg0, %arg1 : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    tosa.yield %1 : tensor<1xf64>
  }
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[IF]] : (tensor<1xf32>) -> tensor<1xf64>
  // DEFAULT: return %[[OUT]] : tensor<1xf64>
  // FUNCBOUND: return %[[IF]] : tensor<1xf32>
  return %0 : tensor<1xf64>
}

// -----

// CHECK-LABEL: test_convert_input_parameters
// DEFAULT: %[[IN:.*]]: tensor<1x3xf64>
// FUNCBOUND: %[[IN:.*]]: tensor<1x3xf32>
func.func @test_convert_input_parameters(%arg0: tensor<1x3xf64>) -> tensor<1x3xf32> {
  // DEFAULT: %[[CAST_IN:.*]] = tosa.cast %[[IN]] : (tensor<1x3xf64>) -> tensor<1x3xf32>
  // DEFAULT: %[[IDENTITY:.*]] = tosa.identity %[[CAST_IN]] : (tensor<1x3xf32>) -> tensor<1x3xf32>
  // FUNCBOUND: %[[IDENTITY:.*]] = tosa.identity %[[IN]] : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %0 = tosa.identity %arg0 : (tensor<1x3xf64>) -> tensor<1x3xf64>
  // COMMON: %[[TO_F32:.*]] = tosa.cast %[[IDENTITY]] : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %1 = tosa.cast %0 : (tensor<1x3xf64>) -> tensor<1x3xf32>
  // DEFAULT: return %[[TO_F32]] : tensor<1x3xf32>
  // FUNCBOUND: return %[[TO_F32]] : tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}

// -----

// CHECK-LABEL: test_f64_const
func.func @test_f64_const() -> tensor<2xf64> {
  // COMMON: %[[CONST:.*]] = "tosa.const"() <{values = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>}> : () -> tensor<2xf32>
  %0 = "tosa.const"() <{values = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>}> : () -> tensor<2xf64>
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[CONST]] : (tensor<2xf32>) -> tensor<2xf64>
  // DEFAULT: return %[[OUT]] : tensor<2xf64>
  // FUNCBOUND: return %[[CONST]] : tensor<2xf32>
  return %0 : tensor<2xf64>
}

// -----

// CHECK-LABEL: test_dense_ressource_f64
func.func @test_dense_ressource_f64() -> tensor<1x2xf64> {
  // COMMON: %[[CONST:.*]] = "tosa.const"() <{values = dense_resource<resource> : tensor<1x2xf32>}> : () -> tensor<1x2xf32>
  %0 = "tosa.const"() <{values = dense_resource<resource> : tensor<1x2xf64>}> : () -> tensor<1x2xf64>
  // DEFAULT: %[[OUT_CAST:.*]] = tosa.cast %[[CONST]] : (tensor<1x2xf32>) -> tensor<1x2xf64>
  // DEFAULT: return %[[OUT_CAST]] : tensor<1x2xf64>
  // FUNCBOUND: return %[[CONST]] : tensor<1x2xf32>
  return %0 : tensor<1x2xf64>
}
{-#
  dialect_resources: {
    builtin: {
      // COMMON: resource: "0x04000000DB0F4940EAD6FCBD"
      resource: "0x08000000182D4454FB21094059F64637DD9ABFBF"
    }
  }
#-}
