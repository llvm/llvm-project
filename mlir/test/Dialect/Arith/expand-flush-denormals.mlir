// RUN: mlir-opt %s -arith-expand=include-flush-denormals=true -split-input-file | FileCheck %s

// Expansion for f32:
//   exp  mask        = 0x7f800000   (sign 0, exp all 1, mantissa 0)
//   clear-man mask   = 0xff800000   (sign 1, exp all 1, mantissa 0)
// When the exponent field is zero (±0 or denormal), the mantissa bits are
// cleared, yielding a sign-preserved zero. Otherwise the bits pass through.

// CHECK-LABEL: func @flush_denormals_f32
// CHECK-SAME:    (%[[ARG0:.+]]: f32) -> f32
// CHECK:         %[[BITS:.+]] = arith.bitcast %[[ARG0]] : f32 to i32
// CHECK:         %[[EXP_MASK:.+]] = arith.constant 2139095040 : i32
// CHECK:         %[[CLEAR_MAN_MASK:.+]] = arith.constant -8388608 : i32
// CHECK:         %[[ZERO:.+]] = arith.constant 0 : i32
// CHECK:         %[[EXP:.+]] = arith.andi %[[BITS]], %[[EXP_MASK]] : i32
// CHECK:         %[[EXP_ZERO:.+]] = arith.cmpi eq, %[[EXP]], %[[ZERO]] : i32
// CHECK:         %[[CLEARED:.+]] = arith.andi %[[BITS]], %[[CLEAR_MAN_MASK]] : i32
// CHECK:         %[[RES_BITS:.+]] = arith.select %[[EXP_ZERO]], %[[CLEARED]], %[[BITS]] : i32
// CHECK:         %[[RES:.+]] = arith.bitcast %[[RES_BITS]] : i32 to f32
// CHECK:         return %[[RES]] : f32
func.func @flush_denormals_f32(%arg0: f32) -> f32 {
  %0 = arith.flush_denormals %arg0 : f32
  return %0 : f32
}

// -----

// Expansion for bf16:
//   exp  mask      = 0x7f80
//   clear-man mask = 0xff80 (-128 as signed i16)

// CHECK-LABEL: func @flush_denormals_bf16
// CHECK:         arith.bitcast %{{.*}} : bf16 to i16
// CHECK:         %[[EXP_MASK:.+]] = arith.constant 32640 : i16
// CHECK:         %[[CLEAR_MAN_MASK:.+]] = arith.constant -128 : i16
// CHECK:         arith.bitcast %{{.*}} : i16 to bf16
func.func @flush_denormals_bf16(%arg0: bf16) -> bf16 {
  %0 = arith.flush_denormals %arg0 : bf16
  return %0 : bf16
}

// -----

// Expansion for f16:
//   exp  mask      = 0x7c00
//   clear-man mask = 0xfc00 (-1024 as signed i16)

// CHECK-LABEL: func @flush_denormals_f16
// CHECK:         arith.bitcast %{{.*}} : f16 to i16
// CHECK:         %[[EXP_MASK:.+]] = arith.constant 31744 : i16
// CHECK:         %[[CLEAR_MAN_MASK:.+]] = arith.constant -1024 : i16
// CHECK:         arith.bitcast %{{.*}} : i16 to f16
func.func @flush_denormals_f16(%arg0: f16) -> f16 {
  %0 = arith.flush_denormals %arg0 : f16
  return %0 : f16
}

// -----

// Expansion for f64 (verifies wide APInt masks work):
//   exp  mask      = 0x7ff0000000000000 =  9218868437227405312
//   clear-man mask = 0xfff0000000000000 = -4503599627370496 (signed i64)

// CHECK-LABEL: func @flush_denormals_f64
// CHECK:         arith.bitcast %{{.*}} : f64 to i64
// CHECK:         %[[EXP_MASK:.+]] = arith.constant 9218868437227405312 : i64
// CHECK:         %[[CLEAR_MAN_MASK:.+]] = arith.constant -4503599627370496 : i64
// CHECK:         arith.bitcast %{{.*}} : i64 to f64
func.func @flush_denormals_f64(%arg0: f64) -> f64 {
  %0 = arith.flush_denormals %arg0 : f64
  return %0 : f64
}

// -----

// CHECK-LABEL: func @flush_denormals_vector
// CHECK:         arith.bitcast %{{.*}} : vector<4xf32> to vector<4xi32>
// CHECK:         arith.andi %{{.*}} : vector<4xi32>
// CHECK:         arith.cmpi eq, %{{.*}} : vector<4xi32>
// CHECK:         arith.andi %{{.*}} : vector<4xi32>
// CHECK:         arith.select %{{.*}}, %{{.*}}, %{{.*}} : vector<4xi1>, vector<4xi32>
// CHECK:         arith.bitcast %{{.*}} : vector<4xi32> to vector<4xf32>
func.func @flush_denormals_vector(%arg0: vector<4xf32>) -> vector<4xf32> {
  %0 = arith.flush_denormals %arg0 : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: func @flush_denormals_tensor
// CHECK:         arith.bitcast %{{.*}} : tensor<8xf32> to tensor<8xi32>
// CHECK:         arith.bitcast %{{.*}} : tensor<8xi32> to tensor<8xf32>
func.func @flush_denormals_tensor(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = arith.flush_denormals %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}
