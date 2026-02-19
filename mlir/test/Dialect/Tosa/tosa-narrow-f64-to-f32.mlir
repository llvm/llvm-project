// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-narrow-f64-to-f32="convert-function-boundaries=0" %s | FileCheck %s --allow-unused-prefixes --check-prefixes=COMMON,DEFAULT
// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-narrow-f64-to-f32="convert-function-boundaries=1" %s | FileCheck %s --allow-unused-prefixes --check-prefixes=COMMON,FUNCBOUND

// -----

// CHECK-LABEL: test_f64_identity_chain
func.func @test_f64_identity_chain(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // DEFAULT: %[[CAST_IN:.*]] = tosa.cast %arg0 : (tensor<1xf64>) -> tensor<1xf32>
  // DEFAULT: %[[ID1:.*]] = tosa.identity %[[CAST_IN]] : (tensor<1xf32>) -> tensor<1xf32>
  // FUNCBOUND: %[[ID1:.*]] = tosa.identity %arg0 : (tensor<1xf32>) -> tensor<1xf32>
  %0 = tosa.identity %arg0 : (tensor<1xf64>) -> tensor<1xf64>
  // COMMON: %[[ID2:.*]] = tosa.identity %[[ID1]] : (tensor<1xf32>) -> tensor<1xf32>
  %1 = tosa.identity %0 : (tensor<1xf64>) -> tensor<1xf64>
  // DEFAULT: %[[CAST_OUT:.*]] = tosa.cast %[[ID2]] : (tensor<1xf32>) -> tensor<1xf64>
  // DEFAULT: return %[[CAST_OUT]] : tensor<1xf64>
  // FUNCBOUND: return %[[ID2]] : tensor<1xf32>
  return %1 : tensor<1xf64>
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

// CHECK-LABEL: test_f64_const_precision_loss
func.func @test_f64_const_precision_loss() -> tensor<1xf64> {
  // expected-error @+2 {{failed to legalize operation 'tosa.const'}}
  // 2^24 + 1 fits in f64 but rounds to 2^24 in f32.
  %0 = "tosa.const"() <{values = dense<16777217.0> : tensor<1xf64>}> : () -> tensor<1xf64>
  return %0 : tensor<1xf64>
}

// -----

// CHECK-LABEL: test_f64_const_precision_loss_small
func.func @test_f64_const_precision_loss_small() -> tensor<1xf64> {
  // expected-error @+2 {{failed to legalize operation 'tosa.const'}}
  // Too small: underflows to zero when narrowed to f32.
  %0 = "tosa.const"() <{values = dense<1.0e-46> : tensor<1xf64>}> : () -> tensor<1xf64>
  return %0 : tensor<1xf64>
}

// -----

// CHECK-LABEL: test_f64_concat
// DEFAULT: %[[A0:.*]]: tensor<13x21x3xf64>, %[[A1:.*]]: tensor<13x21x3xf64>
// FUNCBOUND: %[[A0:.*]]: tensor<13x21x3xf32>, %[[A1:.*]]: tensor<13x21x3xf32>
func.func @test_f64_concat(%arg0: tensor<13x21x3xf64>, %arg1: tensor<13x21x3xf64>) -> tensor<26x21x3xf64> {
  // DEFAULT-DAG: %[[CAST0:.*]] = tosa.cast %[[A0]] : (tensor<13x21x3xf64>) -> tensor<13x21x3xf32>
  // DEFAULT-DAG: %[[CAST1:.*]] = tosa.cast %[[A1]] : (tensor<13x21x3xf64>) -> tensor<13x21x3xf32>
  // COMMON: %[[CONCAT:.*]] = tosa.concat %{{.*}}, %{{.*}} {axis = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<26x21x3xf32>
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<13x21x3xf64>, tensor<13x21x3xf64>) -> tensor<26x21x3xf64>
  // DEFAULT: %[[CAST_OUT:.*]] = tosa.cast %[[CONCAT]] : (tensor<26x21x3xf32>) -> tensor<26x21x3xf64>
  // DEFAULT: return %[[CAST_OUT]] : tensor<26x21x3xf64>
  // FUNCBOUND: return %[[CONCAT]] : tensor<26x21x3xf32>
  return %0 : tensor<26x21x3xf64>
}

// -----

// CHECK-LABEL: test_f64_pad
func.func @test_f64_pad(%arg0: tensor<13x21x3xf64>, %arg1: tensor<1xf64>) -> tensor<15x23x5xf64> {
  %padding = tosa.const_shape {values = dense<1> : tensor<6xindex>} : () -> !tosa.shape<6>
  // DEFAULT-DAG: %[[IN_CAST:.*]] = tosa.cast %arg0 : (tensor<13x21x3xf64>) -> tensor<13x21x3xf32>
  // DEFAULT-DAG: %[[PAD_CAST:.*]] = tosa.cast %arg1 : (tensor<1xf64>) -> tensor<1xf32>
  // COMMON: %[[PAD:.*]] = tosa.pad %{{.*}}, %{{.*}}, %{{.*}} : (tensor<13x21x3xf32>, !tosa.shape<6>, tensor<1xf32>) -> tensor<15x23x5xf32>
  %1 = tosa.pad %arg0, %padding, %arg1 : (tensor<13x21x3xf64>, !tosa.shape<6>, tensor<1xf64>) -> tensor<15x23x5xf64>
  // DEFAULT: %[[OUT_CAST:.*]] = tosa.cast %[[PAD]] : (tensor<15x23x5xf32>) -> tensor<15x23x5xf64>
  // DEFAULT: return %[[OUT_CAST]] : tensor<15x23x5xf64>
  // FUNCBOUND: return %[[PAD]] : tensor<15x23x5xf32>
  return %1 : tensor<15x23x5xf64>
}

// -----

// CHECK-LABEL: test_f64_reshape
func.func @test_f64_reshape(%arg0: tensor<13x21x3xf64>) -> tensor<1x819xf64> {
  %shape = tosa.const_shape {values = dense<[1, 819]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // COMMON: %[[RESHAPE:.*]] = tosa.reshape %{{.*}}, %{{.*}} : (tensor<13x21x3xf32>, !tosa.shape<2>) -> tensor<1x819xf32>
  %0 = tosa.reshape %arg0, %shape : (tensor<13x21x3xf64>, !tosa.shape<2>) -> tensor<1x819xf64>
  // DEFAULT: %[[OUT_CAST:.*]] = tosa.cast %[[RESHAPE]] : (tensor<1x819xf32>) -> tensor<1x819xf64>
  // DEFAULT: return %[[OUT_CAST]] : tensor<1x819xf64>
  // FUNCBOUND: return %[[RESHAPE]] : tensor<1x819xf32>
  return %0 : tensor<1x819xf64>
}

// -----

// CHECK-LABEL: test_f64_reverse
func.func @test_f64_reverse(%arg0: tensor<13x21x3xf64>) -> tensor<13x21x3xf64> {
  // COMMON: %[[REV:.*]] = tosa.reverse %{{.*}} {axis = 0 : i32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<13x21x3xf64>) -> tensor<13x21x3xf64>
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[REV]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xf64>
  // DEFAULT: return %[[OUT]] : tensor<13x21x3xf64>
  // FUNCBOUND: return %[[REV]] : tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf64>
}

// -----

// CHECK-LABEL: test_f64_slice
func.func @test_f64_slice(%arg0: tensor<13x21x3xf64>) -> tensor<4x11x1xf64> {
  %size = tosa.const_shape {values = dense<[4, 11, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %start = tosa.const_shape {values = dense<[6, 8, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  // COMMON: %[[SLICE:.*]] = tosa.slice %{{.*}}, %{{.*}}, %{{.*}} : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xf32>
  %0 = tosa.slice %arg0, %size, %start : (tensor<13x21x3xf64>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xf64>
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[SLICE]] : (tensor<4x11x1xf32>) -> tensor<4x11x1xf64>
  // DEFAULT: return %[[OUT]] : tensor<4x11x1xf64>
  // FUNCBOUND: return %[[SLICE]] : tensor<4x11x1xf32>
  return %0 : tensor<4x11x1xf64>
}

// -----

// CHECK-LABEL: test_f64_tile
func.func @test_f64_tile(%arg0: tensor<13x21x3xf64>) -> tensor<39x21x6xf64> {
  %multipliers = tosa.const_shape { values = dense<[3, 1, 2]> : tensor<3xindex> } : () -> !tosa.shape<3>
  // COMMON: %[[TILE:.*]] = tosa.tile %{{.*}}, %{{.*}} : (tensor<13x21x3xf32>, !tosa.shape<3>) -> tensor<39x21x6xf32>
  %0 = tosa.tile %arg0, %multipliers : (tensor<13x21x3xf64>, !tosa.shape<3>) -> tensor<39x21x6xf64>
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[TILE]] : (tensor<39x21x6xf32>) -> tensor<39x21x6xf64>
  // DEFAULT: return %[[OUT]] : tensor<39x21x6xf64>
  // FUNCBOUND: return %[[TILE]] : tensor<39x21x6xf32>
  return %0 : tensor<39x21x6xf64>
}

// -----

// CHECK-LABEL: test_f64_transpose
func.func @test_f64_transpose(%arg0: tensor<13x21x3xf64>) -> tensor<3x13x21xf64> {
  // COMMON: %[[TRANSPOSE:.*]] = tosa.transpose %{{.*}} {perms = array<i32: 2, 0, 1>} : (tensor<13x21x3xf32>) -> tensor<3x13x21xf32>
  %0 = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>} : (tensor<13x21x3xf64>) -> tensor<3x13x21xf64>
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[TRANSPOSE]] : (tensor<3x13x21xf32>) -> tensor<3x13x21xf64>
  // DEFAULT: return %[[OUT]] : tensor<3x13x21xf64>
  // FUNCBOUND: return %[[TRANSPOSE]] : tensor<3x13x21xf32>
  return %0 : tensor<3x13x21xf64>
}

// -----

module {
// CHECK-LABEL: test_f64_regions
func.func @test_f64_regions(%arg0: tensor<1xf64>, %arg1: tensor<1xf64>, %arg2: tensor<i1>) -> tensor<1xf64> {
  // COMMON: %[[IF_RESULT:.*]] = tosa.cond_if %arg2 : tensor<i1> -> tensor<1xf32>
  %0 = tosa.cond_if %arg2 : tensor<i1> -> tensor<1xf64> {
    // COMMON: %[[ID0:.*]] = tosa.identity %{{.*}} : (tensor<1xf32>) -> tensor<1xf32>
    %1 = tosa.identity %arg0 : (tensor<1xf64>) -> tensor<1xf64>
    // COMMON: tosa.yield %[[ID0]] : tensor<1xf32>
    tosa.yield %1 : tensor<1xf64>
  } else {
    // COMMON: %[[ID1:.*]] = tosa.identity %{{.*}} : (tensor<1xf32>) -> tensor<1xf32>
    %1 = tosa.identity %arg1 : (tensor<1xf64>) -> tensor<1xf64>
    // COMMON: tosa.yield %[[ID1]] : tensor<1xf32>
    tosa.yield %1 : tensor<1xf64>
  }
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[IF_RESULT]] : (tensor<1xf32>) -> tensor<1xf64>
  // DEFAULT: return %[[OUT]] : tensor<1xf64>
  // FUNCBOUND: return %[[IF_RESULT]] : tensor<1xf32>
  return %0 : tensor<1xf64>
}
}

// -----

module {
// CHECK-LABEL: test_f64_add_diagnostic
func.func @test_f64_add_diagnostic(%arg0: tensor<13x21x1xf64>, %arg1: tensor<13x21x3xf64>) -> tensor<13x21x3xf64> {
  // expected-error @+1 {{failed to legalize operation 'tosa.add'}}
  %0 = tosa.add %arg0, %arg1 : (tensor<13x21x1xf64>, tensor<13x21x3xf64>) -> tensor<13x21x3xf64>
  return %0 : tensor<13x21x3xf64>
}
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
      // COMMON: resource: "0x040000000000000000000000"
      resource: "0x0800000000000000000000000000000000000000"
    }
  }
#-}
