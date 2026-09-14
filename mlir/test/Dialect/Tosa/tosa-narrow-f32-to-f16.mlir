// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-narrow-f32-to-f16="convert-function-boundaries=0" %s | FileCheck %s --allow-unused-prefixes --check-prefixes=COMMON,DEFAULT,PRESERVE-ACC
// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-narrow-f32-to-f16="convert-function-boundaries=1" %s | FileCheck %s --allow-unused-prefixes --check-prefixes=COMMON,FUNCBOUND,PRESERVE-ACC
// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-narrow-f32-to-f16="convert-accumulator-type=1" %s | FileCheck %s --allow-unused-prefixes --check-prefixes=COMMON,DEFAULT,CONVERT-ACC

// -----

// CHECK-LABEL: test_accumulator_only_f32
func.func @test_accumulator_only_f32(%arg0: tensor<1x4x4x1xf16>, %arg1: tensor<1xf16>, %arg2: tensor<1xf16>) -> tensor<1x3x3x1xf16> {
  // PRESERVE-ACC: tosa.avg_pool2d
  // PRESERVE-ACC-SAME: acc_type = f32
  // CONVERT-ACC: tosa.avg_pool2d
  // CONVERT-ACC-SAME: acc_type = f16
  %0 = tosa.avg_pool2d %arg0, %arg1, %arg2 {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x1xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x3x3x1xf16>
  return %0 : tensor<1x3x3x1xf16>
}

// -----

// CHECK-LABEL: test_f32_identity_chain
func.func @test_f32_identity_chain(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  // DEFAULT: %[[CAST_IN:.*]] = tosa.cast %arg0 : (tensor<1xf32>) -> tensor<1xf16>
  // DEFAULT: %[[ID1:.*]] = tosa.identity %[[CAST_IN]] : (tensor<1xf16>) -> tensor<1xf16>
  // FUNCBOUND: %[[ID1:.*]] = tosa.identity %arg0 : (tensor<1xf16>) -> tensor<1xf16>
  %0 = tosa.identity %arg0 : (tensor<1xf32>) -> tensor<1xf32>
  // COMMON: %[[ID2:.*]] = tosa.identity %[[ID1]] : (tensor<1xf16>) -> tensor<1xf16>
  %1 = tosa.identity %0 : (tensor<1xf32>) -> tensor<1xf32>
  // DEFAULT: %[[CAST_OUT:.*]] = tosa.cast %[[ID2]] : (tensor<1xf16>) -> tensor<1xf32>
  // DEFAULT: return %[[CAST_OUT]] : tensor<1xf32>
  // FUNCBOUND: return %[[ID2]] : tensor<1xf16>
  return %1 : tensor<1xf32>
}

// -----

// CHECK-LABEL: test_f32_const
func.func @test_f32_const() -> tensor<2xf32> {
  // COMMON: %[[CONST:.*]] = "tosa.const"() <{values = dense<[-1.000000e+00, 2.000000e+00]> : tensor<2xf16>}> : () -> tensor<2xf16>
  %0 = "tosa.const"() <{values = dense<[-1.000000e+00, 2.000000e+00]> : tensor<2xf32>}> : () -> tensor<2xf32>
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[CONST]] : (tensor<2xf16>) -> tensor<2xf32>
  // DEFAULT: return %[[OUT]] : tensor<2xf32>
  // FUNCBOUND: return %[[CONST]] : tensor<2xf16>
  return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: test_f32_const_precision_loss
func.func @test_f32_const_precision_loss() -> tensor<1xf32> {
  // expected-error @+2 {{failed to legalize operation 'tosa.const'}}
  // The value exceeds the finite f16 range.
  %0 = "tosa.const"() <{values = dense<65536.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: test_f32_const_negative_precision_loss
func.func @test_f32_const_negative_precision_loss() -> tensor<1xf32> {
  // expected-error @+2 {{failed to legalize operation 'tosa.const'}}
  // The value exceeds the finite f16 range.
  %0 = "tosa.const"() <{values = dense<-65536.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: test_f32_const_precision_loss_small
func.func @test_f32_const_precision_loss_small() -> tensor<1xf32> {
  // expected-error @+2 {{failed to legalize operation 'tosa.const'}}
  // Too small: underflows to zero when narrowed to f16.
  %0 = "tosa.const"() <{values = dense<1.0e-40> : tensor<1xf32>}> : () -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

// CHECK-LABEL: test_f32_concat
// DEFAULT: %[[A0:.*]]: tensor<13x21x3xf32>, %[[A1:.*]]: tensor<13x21x3xf32>
// FUNCBOUND: %[[A0:.*]]: tensor<13x21x3xf16>, %[[A1:.*]]: tensor<13x21x3xf16>
func.func @test_f32_concat(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<26x21x3xf32> {
  // DEFAULT-DAG: %[[CAST0:.*]] = tosa.cast %[[A0]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xf16>
  // DEFAULT-DAG: %[[CAST1:.*]] = tosa.cast %[[A1]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xf16>
  // COMMON: %[[CONCAT:.*]] = tosa.concat %{{.*}}, %{{.*}} {axis = 0 : i32} : (tensor<13x21x3xf16>, tensor<13x21x3xf16>) -> tensor<26x21x3xf16>
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<26x21x3xf32>
  // DEFAULT: %[[CAST_OUT:.*]] = tosa.cast %[[CONCAT]] : (tensor<26x21x3xf16>) -> tensor<26x21x3xf32>
  // DEFAULT: return %[[CAST_OUT]] : tensor<26x21x3xf32>
  // FUNCBOUND: return %[[CONCAT]] : tensor<26x21x3xf16>
  return %0 : tensor<26x21x3xf32>
}

// -----

// CHECK-LABEL: test_f32_pad
func.func @test_f32_pad(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1xf32>) -> tensor<15x23x5xf32> {
  %padding = tosa.const_shape {values = dense<1> : tensor<6xindex>} : () -> !tosa.shape<6>
  // DEFAULT-DAG: %[[IN_CAST:.*]] = tosa.cast %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xf16>
  // DEFAULT-DAG: %[[PAD_CAST:.*]] = tosa.cast %arg1 : (tensor<1xf32>) -> tensor<1xf16>
  // COMMON: %[[PAD:.*]] = tosa.pad %{{.*}}, %{{.*}}, %{{.*}} : (tensor<13x21x3xf16>, !tosa.shape<6>, tensor<1xf16>) -> tensor<15x23x5xf16>
  %1 = tosa.pad %arg0, %padding, %arg1 : (tensor<13x21x3xf32>, !tosa.shape<6>, tensor<1xf32>) -> tensor<15x23x5xf32>
  // DEFAULT: %[[OUT_CAST:.*]] = tosa.cast %[[PAD]] : (tensor<15x23x5xf16>) -> tensor<15x23x5xf32>
  // DEFAULT: return %[[OUT_CAST]] : tensor<15x23x5xf32>
  // FUNCBOUND: return %[[PAD]] : tensor<15x23x5xf16>
  return %1 : tensor<15x23x5xf32>
}

// -----

// CHECK-LABEL: test_f32_reshape
func.func @test_f32_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<1x819xf32> {
  %shape = tosa.const_shape {values = dense<[1, 819]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // COMMON: %[[RESHAPE:.*]] = tosa.reshape %{{.*}}, %{{.*}} : (tensor<13x21x3xf16>, !tosa.shape<2>) -> tensor<1x819xf16>
  %0 = tosa.reshape %arg0, %shape : (tensor<13x21x3xf32>, !tosa.shape<2>) -> tensor<1x819xf32>
  // DEFAULT: %[[OUT_CAST:.*]] = tosa.cast %[[RESHAPE]] : (tensor<1x819xf16>) -> tensor<1x819xf32>
  // DEFAULT: return %[[OUT_CAST]] : tensor<1x819xf32>
  // FUNCBOUND: return %[[RESHAPE]] : tensor<1x819xf16>
  return %0 : tensor<1x819xf32>
}

// -----

// CHECK-LABEL: test_f32_reverse
func.func @test_f32_reverse(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // COMMON: %[[REV:.*]] = tosa.reverse %{{.*}} {axis = 0 : i32} : (tensor<13x21x3xf16>) -> tensor<13x21x3xf16>
  %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[REV]] : (tensor<13x21x3xf16>) -> tensor<13x21x3xf32>
  // DEFAULT: return %[[OUT]] : tensor<13x21x3xf32>
  // FUNCBOUND: return %[[REV]] : tensor<13x21x3xf16>
  return %0 : tensor<13x21x3xf32>
}

// -----

// CHECK-LABEL: test_f32_slice
func.func @test_f32_slice(%arg0: tensor<13x21x3xf32>) -> tensor<4x11x1xf32> {
  %size = tosa.const_shape {values = dense<[4, 11, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %start = tosa.const_shape {values = dense<[6, 8, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  // COMMON: %[[SLICE:.*]] = tosa.slice %{{.*}}, %{{.*}}, %{{.*}} : (tensor<13x21x3xf16>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xf16>
  %0 = tosa.slice %arg0, %start, %size : (tensor<13x21x3xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x1xf32>
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[SLICE]] : (tensor<4x11x1xf16>) -> tensor<4x11x1xf32>
  // DEFAULT: return %[[OUT]] : tensor<4x11x1xf32>
  // FUNCBOUND: return %[[SLICE]] : tensor<4x11x1xf16>
  return %0 : tensor<4x11x1xf32>
}

// -----

// CHECK-LABEL: test_f32_tile
func.func @test_f32_tile(%arg0: tensor<13x21x3xf32>) -> tensor<39x21x6xf32> {
  %multipliers = tosa.const_shape { values = dense<[3, 1, 2]> : tensor<3xindex> } : () -> !tosa.shape<3>
  // COMMON: %[[TILE:.*]] = tosa.tile %{{.*}}, %{{.*}} : (tensor<13x21x3xf16>, !tosa.shape<3>) -> tensor<39x21x6xf16>
  %0 = tosa.tile %arg0, %multipliers : (tensor<13x21x3xf32>, !tosa.shape<3>) -> tensor<39x21x6xf32>
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[TILE]] : (tensor<39x21x6xf16>) -> tensor<39x21x6xf32>
  // DEFAULT: return %[[OUT]] : tensor<39x21x6xf32>
  // FUNCBOUND: return %[[TILE]] : tensor<39x21x6xf16>
  return %0 : tensor<39x21x6xf32>
}

// -----

// CHECK-LABEL: test_f32_transpose
func.func @test_f32_transpose(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x21xf32> {
  // COMMON: %[[TRANSPOSE:.*]] = tosa.transpose %{{.*}} {perms = array<i32: 2, 0, 1>} : (tensor<13x21x3xf16>) -> tensor<3x13x21xf16>
  %0 = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1>} : (tensor<13x21x3xf32>) -> tensor<3x13x21xf32>
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[TRANSPOSE]] : (tensor<3x13x21xf16>) -> tensor<3x13x21xf32>
  // DEFAULT: return %[[OUT]] : tensor<3x13x21xf32>
  // FUNCBOUND: return %[[TRANSPOSE]] : tensor<3x13x21xf16>
  return %0 : tensor<3x13x21xf32>
}

// -----

module {
// CHECK-LABEL: test_f32_regions
func.func @test_f32_regions(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<i1>) -> tensor<1xf32> {
  // COMMON: %[[IF_RESULT:.*]] = tosa.cond_if %arg2 : tensor<i1> -> tensor<1xf16>
  %0 = tosa.cond_if %arg2 : tensor<i1> -> tensor<1xf32> {
    // COMMON: %[[ID0:.*]] = tosa.identity %{{.*}} : (tensor<1xf16>) -> tensor<1xf16>
    %1 = tosa.identity %arg0 : (tensor<1xf32>) -> tensor<1xf32>
    // COMMON: tosa.yield %[[ID0]] : tensor<1xf16>
    tosa.yield %1 : tensor<1xf32>
  } else {
    // COMMON: %[[ID1:.*]] = tosa.identity %{{.*}} : (tensor<1xf16>) -> tensor<1xf16>
    %1 = tosa.identity %arg1 : (tensor<1xf32>) -> tensor<1xf32>
    // COMMON: tosa.yield %[[ID1]] : tensor<1xf16>
    tosa.yield %1 : tensor<1xf32>
  }
  // DEFAULT: %[[OUT:.*]] = tosa.cast %[[IF_RESULT]] : (tensor<1xf16>) -> tensor<1xf32>
  // DEFAULT: return %[[OUT]] : tensor<1xf32>
  // FUNCBOUND: return %[[IF_RESULT]] : tensor<1xf16>
  return %0 : tensor<1xf32>
}
}

// -----

module {
// CHECK-LABEL: test_f32_add_diagnostic
func.func @test_f32_add_diagnostic(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error @+1 {{failed to legalize operation 'tosa.add'}}
  %0 = tosa.add %arg0, %arg1 : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}
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
