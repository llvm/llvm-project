// RUN: mlir-opt %s --arith-expand="include-f6e2m3fn=true include-f6e3m2fn=true \
// RUN:     include-f8e3m4=true include-f8e4m3=true include-f8e4m3b11fnuz=true \
// RUN:     include-f8e4m3fnuz=true include-f8e5m2fnuz=true" > %t
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=KEPT --implicit-check-not=arith.truncf \
// RUN:   --implicit-check-not=arith.extf --implicit-check-not=vector. < %t
// RUN: mlir-opt %s --arith-expand | \
// RUN:   FileCheck %s --check-prefix=OFF --implicit-check-not=arith.bitcast
// RUN: mlir-opt %s --arith-expand="include-f8e4m3b11fnuz=true" | \
// RUN:   FileCheck %s --check-prefix=B11

// Each expansion computes the result bits in an i32 carrier (i64 for f64 and
// wider) and ends in an arith.bitcast; checks pin the format constants.
// KEPT lists the float ops left, OFF enables no format, B11 one.

// extf, one function per source format. It is exact: the exponent is rebiased
// by (dstBias - srcBias) << dstMantBits and a NaN keeps its sign and payload
// under the destination's quiet NaN.

// 15360 = (127 - 7) << 7; Inf maps to 0x7F80 (32640), NaNs under 0x7FC0.
// CHECK-LABEL: func.func @ext_f8E4M3_bf16(
// CHECK:         arith.extui %{{.*}} : i8 to i32
// CHECK-DAG:     arith.constant 15360 : i32
// CHECK-DAG:     arith.constant 32640 : i32
// CHECK-DAG:     arith.constant 32704 : i32
// CHECK:         %[[HALF:.*]] = arith.trunci %{{.*}} : i32 to i16
// CHECK-NEXT:    arith.bitcast %[[HALF]] : i16 to bf16
// OFF:           arith.extf %arg0 : f8E4M3 to bf16
func.func @ext_f8E4M3_bf16(%arg0: f8E4M3) -> bf16 {
  %0 = arith.extf %arg0 : f8E4M3 to bf16
  return %0 : bf16
}

// An f64 destination uses the i64 carrier; the single NaN 0x80 becomes the
// positive quiet NaN 0x7FF8000000000000.
// CHECK-LABEL: func.func @ext_f8E4M3B11FNUZ_f64(
// CHECK:         arith.extui %{{.*}} : i8 to i64
// CHECK-NOT:     i32
// CHECK:         arith.constant 9221120237041090560 : i64
// CHECK-NOT:     i32
// CHECK:         arith.bitcast %{{.*}} : i64 to f64
// B11-LABEL:   func.func @ext_f8E4M3B11FNUZ_f64(
// B11:           arith.extui %{{.*}} : i8 to i64
// B11-NOT:       arith.extf
// B11:           arith.bitcast %{{.*}} : i64 to f64
func.func @ext_f8E4M3B11FNUZ_f64(%arg0: f8E4M3B11FNUZ) -> f64 {
  %0 = arith.extf %arg0 : f8E4M3B11FNUZ to f64
  return %0 : f64
}

// 14336 = (15 - 1) << 10; f6E2M3FN has no NaN to carry.
// CHECK-LABEL: func.func @ext_f6E2M3FN_f16(
// CHECK:         arith.extui %{{.*}} : i6 to i32
// CHECK-DAG:     arith.constant 14336 : i32
// CHECK:         %[[HALF:.*]] = arith.trunci %{{.*}} : i32 to i16
// CHECK-NEXT:    arith.bitcast %[[HALF]] : i16 to f16
func.func @ext_f6E2M3FN_f16(%arg0: f6E2M3FN) -> f16 {
  %0 = arith.extf %arg0 : f6E2M3FN to f16
  return %0 : f16
}

// 4593671619917905920 = (1023 - 3) << 52.
// CHECK-LABEL: func.func @ext_f6E3M2FN_f64(
// CHECK:         arith.extui %{{.*}} : i6 to i64
// CHECK-DAG:     arith.constant 4593671619917905920 : i64
// CHECK:         arith.bitcast %{{.*}} : i64 to f64
func.func @ext_f6E3M2FN_f64(%arg0: f6E3M2FN) -> f64 {
  %0 = arith.extf %arg0 : f6E3M2FN to f64
  return %0 : f64
}

// 998244352 = (127 - 8) << 23; the single NaN 0x80 becomes +0x7FC00000.
// CHECK-LABEL: func.func @ext_f8E4M3FNUZ_f32(
// CHECK:         arith.extui %{{.*}} : i8 to i32
// CHECK-DAG:     arith.constant 998244352 : i32
// CHECK-DAG:     arith.constant 2143289344 : i32
// CHECK:         arith.bitcast %{{.*}} : i32 to f32
func.func @ext_f8E4M3FNUZ_f32(%arg0: f8E4M3FNUZ) -> f32 {
  %0 = arith.extf %arg0 : f8E4M3FNUZ to f32
  return %0 : f32
}

// f16's bias is one below f8E5M2FNUZ's: a normal code adds (15 - 16) << 10.
// CHECK-LABEL: func.func @ext_f8E5M2FNUZ_f16(
// CHECK:         arith.extui %{{.*}} : i8 to i32
// CHECK-DAG:     arith.constant -1024 : i32
// CHECK-DAG:     arith.constant 32256 : i32
// CHECK:         %[[HALF:.*]] = arith.trunci %{{.*}} : i32 to i16
// CHECK-NEXT:    arith.bitcast %[[HALF]] : i16 to f16
func.func @ext_f8E5M2FNUZ_f16(%arg0: f8E5M2FNUZ) -> f16 {
  %0 = arith.extf %arg0 : f8E5M2FNUZ to f16
  return %0 : f16
}

// 1040187392 = (127 - 3) << 23; the Inf code maps to 0x7F800000.
// CHECK-LABEL: func.func @ext_f8E3M4_f32(
// CHECK:         arith.extui %{{.*}} : i8 to i32
// CHECK-DAG:     arith.constant 1040187392 : i32
// CHECK-DAG:     arith.constant 2139095040 : i32
// CHECK-DAG:     arith.constant 2143289344 : i32
// CHECK:         arith.bitcast %{{.*}} : i32 to f32
func.func @ext_f8E3M4_f32(%arg0: f8E3M4) -> f32 {
  %0 = arith.extf %arg0 : f8E3M4 to f32
  return %0 : f32
}

// An extf into a format without a zero (f8E8M0FNU) or without a sign
// (f8E5M3FNU) decodes to f32 and ends in one arith.truncf with the op's
// fastmath flags, which only include-f8e8m0 expands.
// CHECK-LABEL: func.func @ext_f6E2M3FN_no_zero_or_sign(
// CHECK:         %[[BITS:.*]] = arith.bitcast %arg0 : f6E2M3FN to i6
// CHECK-NEXT:    arith.extui %[[BITS]] : i6 to i32
// CHECK:         %[[W0:.*]] = arith.bitcast %{{.*}} : i32 to f32
// CHECK-NEXT:    %[[R0:.*]] = arith.truncf %[[W0]] fastmath<nnan,contract> : f32 to f8E8M0FNU
// CHECK:         %[[W1:.*]] = arith.bitcast %{{.*}} : i32 to f32
// CHECK-NEXT:    %[[R1:.*]] = arith.truncf %[[W1]] : f32 to f8E5M3FNU
// CHECK-NEXT:    return %[[R0]], %[[R1]]
// KEPT:          arith.truncf %{{.*}} fastmath<nnan,contract> : f32 to f8E8M0FNU
// KEPT:          arith.truncf %{{.*}} : f32 to f8E5M3FNU
func.func @ext_f6E2M3FN_no_zero_or_sign(%arg0: f6E2M3FN)
    -> (f8E8M0FNU, f8E5M3FNU) {
  %0 = arith.extf %arg0 fastmath<nnan,contract> : f6E2M3FN to f8E8M0FNU
  %1 = arith.extf %arg0 : f6E2M3FN to f8E5M3FNU
  return %0, %1 : f8E8M0FNU, f8E5M3FNU
}

// Shapes: one copy of the expansion serves any static or scalable shape, with
// splat constants (one trunci for the whole vector, one multiply in the
// f6E3M2FN decode).
// CHECK-LABEL: func.func @shapes(
// CHECK:         arith.bitcast %arg0 : vector<8xf8E4M3> to vector<8xi8>
// CHECK-NEXT:    arith.extui %{{.*}} : vector<8xi8> to vector<8xi32>
// CHECK:         arith.constant dense<1006632960> : vector<8xi32>
// CHECK:         arith.bitcast %{{.*}} : vector<8xi32> to vector<8xf32>
// CHECK:         arith.extui %{{.*}} : vector<2x3xi6> to vector<2x3xi32>
// CHECK:         arith.trunci %{{.*}} : vector<2x3xi32> to vector<2x3xi16>
// CHECK-NEXT:    arith.bitcast %{{.*}} : vector<2x3xi16> to vector<2x3xf16>
// CHECK:         arith.extui %{{.*}} : vector<i8> to vector<i32>
// CHECK:         arith.bitcast %{{.*}} : vector<i32> to vector<f32>
// CHECK:         arith.extui %{{.*}} : tensor<2x2xi8> to tensor<2x2xi32>
// CHECK:         arith.bitcast %{{.*}} : tensor<2x2xi32> to tensor<2x2xf32>
// CHECK:         arith.extui %{{.*}} : vector<[8]xi6> to vector<[8]xi32>
// CHECK-NOT:     arith.muli
// CHECK:         arith.muli %{{.*}}, %{{.*}} : vector<[8]xi32>
// CHECK-NOT:     arith.muli
// CHECK:         arith.bitcast %{{.*}} : vector<[8]xi32> to vector<[8]xf32>
func.func @shapes(%arg0: vector<8xf8E4M3>, %arg1: vector<2x3xf6E2M3FN>,
                  %arg2: vector<f8E3M4>, %arg3: tensor<2x2xf8E4M3FNUZ>,
                  %arg4: vector<[8]xf6E3M2FN>)
    -> (vector<8xf32>, vector<2x3xf16>, vector<f32>, tensor<2x2xf32>,
        vector<[8]xf32>) {
  %0 = arith.extf %arg0 : vector<8xf8E4M3> to vector<8xf32>
  %1 = arith.extf %arg1 : vector<2x3xf6E2M3FN> to vector<2x3xf16>
  %2 = arith.extf %arg2 : vector<f8E3M4> to vector<f32>
  %3 = arith.extf %arg3 : tensor<2x2xf8E4M3FNUZ> to tensor<2x2xf32>
  %4 = arith.extf %arg4 : vector<[8]xf6E3M2FN> to vector<[8]xf32>
  return %0, %1, %2, %3, %4 : vector<8xf32>, vector<2x3xf16>, vector<f32>,
      tensor<2x2xf32>, vector<[8]xf32>
}

// Partners outside f16 / bf16 / f32 / f64: an extf to f128 or f80 is the
// expansion to f64 followed by an exact arith.extf with the op's fastmath
// flags, and one to tf32 goes through f16 the same way.
// CHECK-LABEL: func.func @wide_partners(
// CHECK:         arith.extui %{{.*}} : i8 to i64
// CHECK:         %[[W0:.*]] = arith.bitcast %{{.*}} : i64 to f64
// CHECK-NEXT:    %[[R0:.*]] = arith.extf %[[W0]] fastmath<nnan,ninf> : f64 to f128
// CHECK:         arith.extui %{{.*}} : vector<4xi6> to vector<4xi64>
// CHECK:         %[[W1:.*]] = arith.bitcast %{{.*}} : vector<4xi64> to vector<4xf64>
// CHECK-NEXT:    %[[R1:.*]] = arith.extf %[[W1]] : vector<4xf64> to vector<4xf80>
// CHECK:         arith.trunci %{{.*}} : i32 to i16
// CHECK-NEXT:    %[[W2:.*]] = arith.bitcast %{{.*}} : i16 to f16
// CHECK-NEXT:    %[[R2:.*]] = arith.extf %[[W2]] : f16 to tf32
// CHECK-NEXT:    return %[[R0]], %[[R1]], %[[R2]]
// KEPT:          arith.extf %{{.*}} fastmath<nnan,ninf> : f64 to f128
// KEPT:          arith.extf %{{.*}} : vector<4xf64> to vector<4xf80>
// KEPT:          arith.extf %{{.*}} : f16 to tf32
func.func @wide_partners(%arg0: f8E4M3, %arg1: vector<4xf6E2M3FN>,
                         %arg2: f8E5M2FNUZ)
    -> (f128, vector<4xf80>, tf32) {
  %0 = arith.extf %arg0 fastmath<nnan,ninf> : f8E4M3 to f128
  %1 = arith.extf %arg1 : vector<4xf6E2M3FN> to vector<4xf80>
  %2 = arith.extf %arg2 : f8E5M2FNUZ to tf32
  return %0, %1, %2 : f128, vector<4xf80>, tf32
}

// Left alone: dynamic shapes (no splat constant), f8E8M0FNU (only
// include-f8e8m0 expands it, see expand-ops.mlir) and f8E5M3FNU (no option).
// KEPT:          arith.extf %arg0 : tensor<?x4xf8E4M3> to tensor<?x4xf32>
// KEPT:          arith.extf %arg1 : f8E8M0FNU to f32
// KEPT:          arith.extf %arg2 : f8E5M3FNU to f32
func.func @left_alone(%arg0: tensor<?x4xf8E4M3>, %arg1: f8E8M0FNU,
                      %arg2: f8E5M3FNU) -> (tensor<?x4xf32>, f32, f32) {
  %0 = arith.extf %arg0 : tensor<?x4xf8E4M3> to tensor<?x4xf32>
  %1 = arith.extf %arg1 : f8E8M0FNU to f32
  %2 = arith.extf %arg2 : f8E5M3FNU to f32
  return %0, %1, %2 : tensor<?x4xf32>, f32, f32
}
