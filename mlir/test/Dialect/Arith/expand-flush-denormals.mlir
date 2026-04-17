// RUN: mlir-opt %s -arith-expand-flush-denormals -split-input-file | FileCheck %s

// Expansion for f32:
//   sign mask      = 0x80000000   (00000000011111111111111111111111)
//   exp  mask      = 0x7f800000   (01111111100000000000000000000000)
//   mantissa mask  = 0x007fffff   (00000000011111111111111111111111)
// Bit pattern for denormal: zero exponent and non-zero mantissa
// Bit pattern for zero: 0/1 sign bit, remaining bits zero

// CHECK-LABEL: func @flush_denormals_f32
// CHECK-SAME:    (%[[ARG0:.+]]: f32) -> f32
// CHECK:         %[[BITS:.+]] = arith.bitcast %[[ARG0]] : f32 to i32
// CHECK:         %[[MAN_MASK:.+]] = arith.constant 8388607 : i32
// CHECK:         %[[EXP_MASK:.+]] = arith.constant 2139095040 : i32
// CHECK:         %[[SIGN_MASK:.+]] = arith.constant -2147483648 : i32
// CHECK:         %[[ZERO:.+]] = arith.constant 0 : i32
// CHECK:         %[[EXP:.+]] = arith.andi %[[BITS]], %[[EXP_MASK]] : i32
// CHECK:         %[[EXP_ZERO:.+]] = arith.cmpi eq, %[[EXP]], %[[ZERO]] : i32
// CHECK:         %[[MAN:.+]] = arith.andi %[[BITS]], %[[MAN_MASK]] : i32
// CHECK:         %[[MAN_NONZERO:.+]] = arith.cmpi ne, %[[MAN]], %[[ZERO]] : i32
// CHECK:         %[[IS_DEN:.+]] = arith.andi %[[EXP_ZERO]], %[[MAN_NONZERO]] : i1
// CHECK:         %[[SIGN_ONLY:.+]] = arith.andi %[[BITS]], %[[SIGN_MASK]] : i32
// CHECK:         %[[RES_BITS:.+]] = arith.select %[[IS_DEN]], %[[SIGN_ONLY]], %[[BITS]] : i32
// CHECK:         %[[RES:.+]] = arith.bitcast %[[RES_BITS]] : i32 to f32
// CHECK:         return %[[RES]] : f32
func.func @flush_denormals_f32(%arg0: f32) -> f32 {
  %0 = arith.flush_denormals %arg0 : f32
  return %0 : f32
}

// -----

// Expansion for bf16:
//   sign mask      = 0x8000
//   exp  mask      = 0x7f80
//   mantissa mask  = 0x007f

// CHECK-LABEL: func @flush_denormals_bf16
// CHECK:         arith.bitcast %{{.*}} : bf16 to i16
// CHECK:         %[[MAN_MASK:.+]] = arith.constant 127 : i16
// CHECK:         %[[EXP_MASK:.+]] = arith.constant 32640 : i16
// CHECK:         %[[SIGN_MASK:.+]] = arith.constant -32768 : i16
// CHECK:         arith.bitcast %{{.*}} : i16 to bf16
func.func @flush_denormals_bf16(%arg0: bf16) -> bf16 {
  %0 = arith.flush_denormals %arg0 : bf16
  return %0 : bf16
}

// -----

// Expansion for f16:
//   sign mask      = 0x8000
//   exp  mask      = 0x7c00
//   mantissa mask  = 0x03ff

// CHECK-LABEL: func @flush_denormals_f16
// CHECK:         arith.bitcast %{{.*}} : f16 to i16
// CHECK:         %[[MAN_MASK:.+]] = arith.constant 1023 : i16
// CHECK:         %[[EXP_MASK:.+]] = arith.constant 31744 : i16
// CHECK:         %[[SIGN_MASK:.+]] = arith.constant -32768 : i16
// CHECK:         arith.bitcast %{{.*}} : i16 to f16
func.func @flush_denormals_f16(%arg0: f16) -> f16 {
  %0 = arith.flush_denormals %arg0 : f16
  return %0 : f16
}

// -----

// Expansion for f64 (verifies wide APInt masks work):
//   sign mask = 0x8000000000000000 = -9223372036854775808 (signed i64)
//   exp  mask = 0x7ff0000000000000 =  9218868437227405312
//   man  mask = 0x000fffffffffffff =     4503599627370495

// CHECK-LABEL: func @flush_denormals_f64
// CHECK:         arith.bitcast %{{.*}} : f64 to i64
// CHECK:         %[[MAN_MASK:.+]] = arith.constant 4503599627370495 : i64
// CHECK:         %[[EXP_MASK:.+]] = arith.constant 9218868437227405312 : i64
// CHECK:         %[[SIGN_MASK:.+]] = arith.constant -9223372036854775808 : i64
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
// CHECK:         arith.cmpi ne, %{{.*}} : vector<4xi32>
// CHECK:         arith.andi %{{.*}} : vector<4xi1>
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
