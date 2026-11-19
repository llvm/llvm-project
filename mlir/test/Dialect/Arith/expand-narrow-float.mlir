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
// RUN: mlir-opt %s --arith-expand="include-f6e2m3fn=true" | \
// RUN:   FileCheck %s --check-prefix=E2M3
// RUN: mlir-opt %s --arith-expand="include-f6e3m2fn=true" | \
// RUN:   FileCheck %s --check-prefix=E3M2
// RUN: mlir-opt %s --arith-expand="include-f8e3m4=true" | \
// RUN:   FileCheck %s --check-prefix=E3M4
// RUN: mlir-opt %s --arith-expand="include-f8e4m3=true" | \
// RUN:   FileCheck %s --check-prefix=E4M3
// RUN: mlir-opt %s --arith-expand="include-f8e4m3fnuz=true" | \
// RUN:   FileCheck %s --check-prefix=FNUZ43
// RUN: mlir-opt %s --arith-expand="include-f8e5m2fnuz=true" | \
// RUN:   FileCheck %s --check-prefix=FNUZ52

// Each expansion computes the code in an i32 (i64 from f64 or wider) carrier
// and ends in arith.trunci + arith.bitcast; checks pin the format and rounding
// constants.
// KEPT lists the float ops left, OFF enables no format, the others one each.

// truncf, one function per destination format.

// to_nearest_away adds 2^8 (bf16 has 7 mantissa bits); f6E2M3FN has no Inf or
// NaN, so an overflow saturates at 31 (7.5).
// CHECK-LABEL: func.func @trunc_bf16_f6E2M3FN_to_nearest_away(
// CHECK:         %[[BITS:.*]] = arith.bitcast %arg0 : bf16 to i16
// CHECK-NEXT:    arith.extui %[[BITS]] : i16 to i32
// CHECK-DAG:     arith.constant 256 : i32
// CHECK-DAG:     arith.constant 31 : i32
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i32 to i6
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i6 to f6E2M3FN
func.func @trunc_bf16_f6E2M3FN_to_nearest_away(%arg0: bf16) -> f6E2M3FN {
  %0 = arith.truncf %arg0 to_nearest_away : bf16 to f6E2M3FN
  return %0 : f6E2M3FN
}

// An f64 source is computed in i64 with no i32 value; 31 is max-finite (28.0).
// CHECK-LABEL: func.func @trunc_f64_f6E3M2FN(
// CHECK:         arith.bitcast %arg0 : f64 to i64
// CHECK-NOT:     i32
// CHECK:         arith.constant 31 : i64
// CHECK-NOT:     i32
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i64 to i6
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i6 to f6E3M2FN
func.func @trunc_f64_f6E3M2FN(%arg0: f64) -> f6E3M2FN {
  %0 = arith.truncf %arg0 : f64 to f6E3M2FN
  return %0 : f6E3M2FN
}

// upward rounds the magnitude up for x >= 0 (a signed compare of the bits) by
// adding 2^25 - 1: a positive overflow is +Inf (120), a negative one stops at
// max-finite 119 (240). include-f8e4m3b11fnuz alone leaves it.
// CHECK-LABEL: func.func @trunc_f32_f8E4M3_upward(
// CHECK:         %[[BITS:.*]] = arith.bitcast %arg0 : f32 to i32
// CHECK-DAG:     arith.cmpi sge, %[[BITS]], %{{.*}} : i32
// CHECK-DAG:     arith.constant 33554431 : i32
// CHECK-DAG:     arith.constant 119 : i32
// CHECK-DAG:     arith.constant 120 : i32
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i32 to i8
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i8 to f8E4M3
// B11-LABEL:   func.func @trunc_f32_f8E4M3_upward(
// B11:           arith.truncf %arg0 upward : f32 to f8E4M3
func.func @trunc_f32_f8E4M3_upward(%arg0: f32) -> f8E4M3 {
  %0 = arith.truncf %arg0 upward : f32 to f8E4M3
  return %0 : f8E4M3
}

// f8E4M3B11FNUZ: max-finite 127 (30.0) and the single NaN 128 (0x80, the -0
// slot); 2047 = 2^11 - 1 is the f16 to_nearest_even addend.
// CHECK-LABEL: func.func @trunc_f16_f8E4M3B11FNUZ(
// CHECK:         %[[BITS:.*]] = arith.bitcast %arg0 : f16 to i16
// CHECK-NEXT:    arith.extui %[[BITS]] : i16 to i32
// CHECK-DAG:     arith.constant 2047 : i32
// CHECK-DAG:     arith.constant 127 : i32
// CHECK-DAG:     arith.constant 128 : i32
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i32 to i8
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i8 to f8E4M3B11FNUZ
// B11-LABEL:   func.func @trunc_f16_f8E4M3B11FNUZ(
// B11:           arith.bitcast %arg0 : f16 to i16
// B11-NOT:       arith.truncf
// B11:           arith.bitcast %{{.*}} : i8 to f8E4M3B11FNUZ
func.func @trunc_f16_f8E4M3B11FNUZ(%arg0: f16) -> f8E4M3B11FNUZ {
  %0 = arith.truncf %arg0 : f16 to f8E4M3B11FNUZ
  return %0 : f8E4M3B11FNUZ
}

// to_nearest_even spelled out. f8E4M3FNUZ has bias 8 (its normals start at the
// f32 exponent field 127 - 8 + 1 = 120), max-finite 127 and the single NaN 128.
// CHECK-LABEL: func.func @trunc_f32_f8E4M3FNUZ(
// CHECK:         arith.bitcast %arg0 : f32 to i32
// CHECK-DAG:     arith.constant 120 : i32
// CHECK-DAG:     arith.constant 127 : i32
// CHECK-DAG:     arith.constant 128 : i32
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i32 to i8
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i8 to f8E4M3FNUZ
func.func @trunc_f32_f8E4M3FNUZ(%arg0: f32) -> f8E4M3FNUZ {
  %0 = arith.truncf %arg0 to_nearest_even : f32 to f8E4M3FNUZ
  return %0 : f8E4M3FNUZ
}

// f8E5M2FNUZ's bias (16) is above f16's (15): the source is first extended
// exactly to f32 and rounded once from there (2^24 - 1; 112 = 127 - 16 + 1).
// CHECK-LABEL: func.func @trunc_f16_f8E5M2FNUZ(
// CHECK:         %[[WIDE:.*]] = arith.extf %arg0 : f16 to f32
// CHECK:         arith.bitcast %[[WIDE]] : f32 to i32
// CHECK-DAG:     arith.constant 16777215 : i32
// CHECK-DAG:     arith.constant 112 : i32
// CHECK-DAG:     arith.constant 128 : i32
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i32 to i8
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i8 to f8E5M2FNUZ
// KEPT:          arith.extf %arg0 : f16 to f32
func.func @trunc_f16_f8E5M2FNUZ(%arg0: f16) -> f8E5M2FNUZ {
  %0 = arith.truncf %arg0 : f16 to f8E5M2FNUZ
  return %0 : f8E5M2FNUZ
}

// f8E3M4 (bias 3) has an Inf (112 = 0x70) and NaNs (quiet NaN 120 = 0x78);
// 255 = 2^8 - 1 is the bf16 to_nearest_even addend.
// CHECK-LABEL: func.func @trunc_bf16_f8E3M4(
// CHECK:         %[[BITS:.*]] = arith.bitcast %arg0 : bf16 to i16
// CHECK-NEXT:    arith.extui %[[BITS]] : i16 to i32
// CHECK-DAG:     arith.constant 255 : i32
// CHECK-DAG:     arith.constant 112 : i32
// CHECK-DAG:     arith.constant 120 : i32
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i32 to i8
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i8 to f8E3M4
func.func @trunc_bf16_f8E3M4(%arg0: bf16) -> f8E3M4 {
  %0 = arith.truncf %arg0 : bf16 to f8E3M4
  return %0 : f8E3M4
}

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

// Narrow to narrow. The source is decoded exactly to f32 bits (956301312 =
// (127 - 13) << 23, 2^-13 being the smallest f8E4M3B11FNUZ subnormal) and
// rounded once, here downward (2^25 - 1); only the destination's option is
// needed, so include-f8e4m3b11fnuz alone leaves it.
// CHECK-LABEL: func.func @trunc_f8E4M3B11FNUZ_f6E3M2FN_downward(
// CHECK:         %[[BITS:.*]] = arith.bitcast %arg0 : f8E4M3B11FNUZ to i8
// CHECK-NEXT:    arith.extui %[[BITS]] : i8 to i32
// CHECK-DAG:     arith.constant 956301312 : i32
// CHECK-DAG:     arith.constant 33554431 : i32
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i32 to i6
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i6 to f6E3M2FN
// B11-LABEL:   func.func @trunc_f8E4M3B11FNUZ_f6E3M2FN_downward(
// B11:           arith.truncf %arg0 downward : f8E4M3B11FNUZ to f6E3M2FN
func.func @trunc_f8E4M3B11FNUZ_f6E3M2FN_downward(%arg0: f8E4M3B11FNUZ)
    -> f6E3M2FN {
  %0 = arith.truncf %arg0 downward : f8E4M3B11FNUZ to f6E3M2FN
  return %0 : f6E3M2FN
}

// A narrow-to-narrow extf decodes f6E2M3FN to f32 bits (0.125 = 1040187392)
// and encodes f8E4M3 (Inf 120), exactly; either option enables it.
// CHECK-LABEL: func.func @ext_f6E2M3FN_f8E4M3(
// CHECK:         %[[BITS:.*]] = arith.bitcast %arg0 : f6E2M3FN to i6
// CHECK-NEXT:    arith.extui %[[BITS]] : i6 to i32
// CHECK-DAG:     arith.constant 1040187392 : i32
// CHECK-DAG:     arith.constant 120 : i32
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i32 to i8
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i8 to f8E4M3
// E2M3-LABEL:  func.func @ext_f6E2M3FN_f8E4M3(
// E2M3:          arith.bitcast %arg0 : f6E2M3FN to i6
// E2M3-NOT:      arith.extf
// E2M3:          arith.bitcast %{{.*}} : i8 to f8E4M3
func.func @ext_f6E2M3FN_f8E4M3(%arg0: f6E2M3FN) -> f8E4M3 {
  %0 = arith.extf %arg0 : f6E2M3FN to f8E4M3
  return %0 : f8E4M3
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

// fastmath flags carry over to the float ops the expansion keeps, but those
// before the rounding keep only nnan (with contract, ExtFOp::fold would fold
// the widening of a contract truncf back to the truncf's operand), here the
// exact widening of an f16 source into f8E5M2FNUZ.
// CHECK-LABEL: func.func @fastmath(
// CHECK:         arith.extf %arg0 fastmath<nnan> : f16 to f32
// KEPT:          arith.extf %arg0 fastmath<nnan> : f16 to f32
func.func @fastmath(%arg0: f16) -> f8E5M2FNUZ {
  %0 = arith.truncf %arg0 fastmath<nnan,contract> : f16 to f8E5M2FNUZ
  return %0 : f8E5M2FNUZ
}

// Each option enables its own format and no other: the one expanded truncf is
// the only one whose result is a bitcast from an integer.
// E2M3-LABEL:   func.func @one_truncf_per_format(
// E2M3-NOT:       : i{{[0-9]+}} to f
// E2M3:           : i6 to f6E2M3FN{{$}}
// E2M3-NOT:       : i{{[0-9]+}} to f
// E2M3:           return
// E3M2-LABEL:   func.func @one_truncf_per_format(
// E3M2-NOT:       : i{{[0-9]+}} to f
// E3M2:           : i6 to f6E3M2FN{{$}}
// E3M2-NOT:       : i{{[0-9]+}} to f
// E3M2:           return
// E3M4-LABEL:   func.func @one_truncf_per_format(
// E3M4-NOT:       : i{{[0-9]+}} to f
// E3M4:           : i8 to f8E3M4{{$}}
// E3M4-NOT:       : i{{[0-9]+}} to f
// E3M4:           return
// E4M3-LABEL:   func.func @one_truncf_per_format(
// E4M3-NOT:       : i{{[0-9]+}} to f
// E4M3:           : i8 to f8E4M3{{$}}
// E4M3-NOT:       : i{{[0-9]+}} to f
// E4M3:           return
// FNUZ43-LABEL: func.func @one_truncf_per_format(
// FNUZ43-NOT:     : i{{[0-9]+}} to f
// FNUZ43:         : i8 to f8E4M3FNUZ{{$}}
// FNUZ43-NOT:     : i{{[0-9]+}} to f
// FNUZ43:         return
// B11-LABEL:    func.func @one_truncf_per_format(
// B11-NOT:        : i{{[0-9]+}} to f
// B11:            : i8 to f8E4M3B11FNUZ{{$}}
// B11-NOT:        : i{{[0-9]+}} to f
// B11:            return
// FNUZ52-LABEL: func.func @one_truncf_per_format(
// FNUZ52-NOT:     : i{{[0-9]+}} to f
// FNUZ52:         : i8 to f8E5M2FNUZ{{$}}
// FNUZ52-NOT:     : i{{[0-9]+}} to f
// FNUZ52:         return
func.func @one_truncf_per_format(%arg0: f32)
    -> (f6E2M3FN, f6E3M2FN, f8E3M4, f8E4M3, f8E4M3FNUZ, f8E4M3B11FNUZ,
        f8E5M2FNUZ) {
  %0 = arith.truncf %arg0 : f32 to f6E2M3FN
  %1 = arith.truncf %arg0 : f32 to f6E3M2FN
  %2 = arith.truncf %arg0 : f32 to f8E3M4
  %3 = arith.truncf %arg0 : f32 to f8E4M3
  %4 = arith.truncf %arg0 : f32 to f8E4M3FNUZ
  %5 = arith.truncf %arg0 : f32 to f8E4M3B11FNUZ
  %6 = arith.truncf %arg0 : f32 to f8E5M2FNUZ
  return %0, %1, %2, %3, %4, %5, %6
      : f6E2M3FN, f6E3M2FN, f8E3M4, f8E4M3, f8E4M3FNUZ, f8E4M3B11FNUZ,
        f8E5M2FNUZ
}

// Shapes: one copy of the expansion serves any static or scalable shape, with
// splat constants (one trunci for the whole vector, one multiply in the
// f6E3M2FN decode).
// CHECK-LABEL: func.func @shapes(
// CHECK:         arith.bitcast %arg0 : vector<4xf32> to vector<4xi32>
// CHECK-NOT:     arith.trunci
// CHECK:         arith.constant dense<120> : vector<4xi32>
// CHECK-NOT:     arith.trunci
// CHECK:         arith.trunci %{{.*}} : vector<4xi32> to vector<4xi8>
// CHECK-NEXT:    arith.bitcast %{{.*}} : vector<4xi8> to vector<4xf8E4M3>
// CHECK:         arith.extui %{{.*}} : vector<8xi8> to vector<8xi32>
// CHECK:         arith.bitcast %{{.*}} : vector<8xi32> to vector<8xf32>
// CHECK:         arith.bitcast %arg2 : vector<2x3xf32> to vector<2x3xi32>
// CHECK:         arith.trunci %{{.*}} : vector<2x3xi32> to vector<2x3xi6>
// CHECK:         arith.bitcast %arg3 : vector<f32> to vector<i32>
// CHECK:         arith.trunci %{{.*}} : vector<i32> to vector<i8>
// CHECK:         arith.bitcast %arg4 : tensor<4xf64> to tensor<4xi64>
// CHECK:         arith.constant dense<31> : tensor<4xi64>
// CHECK:         arith.trunci %{{.*}} : tensor<4xi64> to tensor<4xi6>
// CHECK:         arith.extui %{{.*}} : tensor<2x2xi8> to tensor<2x2xi32>
// CHECK:         arith.bitcast %{{.*}} : tensor<2x2xi32> to tensor<2x2xf32>
// CHECK:         arith.bitcast %arg6 : vector<[4]xf32> to vector<[4]xi32>
// CHECK:         arith.constant dense<112> : vector<[4]xi32>
// CHECK:         arith.trunci %{{.*}} : vector<[4]xi32> to vector<[4]xi8>
// CHECK:         arith.extui %{{.*}} : vector<[8]xi6> to vector<[8]xi32>
// CHECK-NOT:     arith.muli
// CHECK:         arith.muli %{{.*}}, %{{.*}} : vector<[8]xi32>
// CHECK-NOT:     arith.muli
// CHECK:         arith.bitcast %{{.*}} : vector<[8]xi32> to vector<[8]xf32>
func.func @shapes(%arg0: vector<4xf32>, %arg1: vector<8xf8E4M3>,
                  %arg2: vector<2x3xf32>, %arg3: vector<f32>,
                  %arg4: tensor<4xf64>, %arg5: tensor<2x2xf8E4M3FNUZ>,
                  %arg6: vector<[4]xf32>, %arg7: vector<[8]xf6E3M2FN>)
    -> (vector<4xf8E4M3>, vector<8xf32>, vector<2x3xf6E2M3FN>,
        vector<f8E5M2FNUZ>, tensor<4xf6E3M2FN>, tensor<2x2xf32>,
        vector<[4]xf8E3M4>, vector<[8]xf32>) {
  %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xf8E4M3>
  %1 = arith.extf %arg1 : vector<8xf8E4M3> to vector<8xf32>
  %2 = arith.truncf %arg2 : vector<2x3xf32> to vector<2x3xf6E2M3FN>
  %3 = arith.truncf %arg3 : vector<f32> to vector<f8E5M2FNUZ>
  %4 = arith.truncf %arg4 : tensor<4xf64> to tensor<4xf6E3M2FN>
  %5 = arith.extf %arg5 : tensor<2x2xf8E4M3FNUZ> to tensor<2x2xf32>
  %6 = arith.truncf %arg6 : vector<[4]xf32> to vector<[4]xf8E3M4>
  %7 = arith.extf %arg7 : vector<[8]xf6E3M2FN> to vector<[8]xf32>
  return %0, %1, %2, %3, %4, %5, %6, %7
      : vector<4xf8E4M3>, vector<8xf32>, vector<2x3xf6E2M3FN>,
        vector<f8E5M2FNUZ>, tensor<4xf6E3M2FN>, tensor<2x2xf32>,
        vector<[4]xf8E3M4>, vector<[8]xf32>
}

// Partners outside f16 / bf16 / f32 / f64: a tf32 source is extended exactly
// to f32 first; an extf to f128 or f80 is the expansion to f64 followed by an
// exact arith.extf with the op's fastmath flags, and one to tf32 goes through
// f16 the same way.
// CHECK-LABEL: func.func @wide_partners(
// CHECK:         %[[WIDE:.*]] = arith.extf %arg0 : tf32 to f32
// CHECK:         arith.bitcast %[[WIDE]] : f32 to i32
// CHECK:         arith.bitcast %{{.*}} : i8 to f8E4M3FNUZ
// CHECK:         arith.extui %{{.*}} : i8 to i64
// CHECK:         %[[W0:.*]] = arith.bitcast %{{.*}} : i64 to f64
// CHECK-NEXT:    %[[R0:.*]] = arith.extf %[[W0]] fastmath<nnan,ninf> : f64 to f128
// CHECK:         arith.extui %{{.*}} : vector<4xi6> to vector<4xi64>
// CHECK:         %[[W1:.*]] = arith.bitcast %{{.*}} : vector<4xi64> to vector<4xf64>
// CHECK-NEXT:    %[[R1:.*]] = arith.extf %[[W1]] : vector<4xf64> to vector<4xf80>
// CHECK:         arith.trunci %{{.*}} : i32 to i16
// CHECK-NEXT:    %[[W2:.*]] = arith.bitcast %{{.*}} : i16 to f16
// CHECK-NEXT:    %[[R2:.*]] = arith.extf %[[W2]] : f16 to tf32
// CHECK-NEXT:    return %{{.*}}, %[[R0]], %[[R1]], %[[R2]]
// KEPT:          arith.extf %arg0 : tf32 to f32
// KEPT:          arith.extf %{{.*}} fastmath<nnan,ninf> : f64 to f128
// KEPT:          arith.extf %{{.*}} : vector<4xf64> to vector<4xf80>
// KEPT:          arith.extf %{{.*}} : f16 to tf32
func.func @wide_partners(%arg0: tf32, %arg1: f8E4M3, %arg2: vector<4xf6E2M3FN>,
                         %arg3: f8E5M2FNUZ)
    -> (f8E4M3FNUZ, f128, vector<4xf80>, tf32) {
  %0 = arith.truncf %arg0 : tf32 to f8E4M3FNUZ
  %1 = arith.extf %arg1 fastmath<nnan,ninf> : f8E4M3 to f128
  %2 = arith.extf %arg2 : vector<4xf6E2M3FN> to vector<4xf80>
  %3 = arith.extf %arg3 : f8E5M2FNUZ to tf32
  return %0, %1, %2, %3 : f8E4M3FNUZ, f128, vector<4xf80>, tf32
}

// Left alone: dynamic shapes (no splat constant), f8E8M0FNU (only
// include-f8e8m0 expands it, see expand-ops.mlir) and f8E5M3FNU (no option).
// KEPT:          arith.truncf %arg0 : tensor<?xf32> to tensor<?xf6E2M3FN>
// KEPT:          arith.extf %arg1 : tensor<?x4xf8E4M3> to tensor<?x4xf32>
// KEPT:          arith.truncf %arg2 : f32 to f8E8M0FNU
// KEPT:          arith.extf %arg3 : f8E8M0FNU to f32
// KEPT:          arith.truncf %arg2 : f32 to f8E5M3FNU
func.func @left_alone(%arg0: tensor<?xf32>, %arg1: tensor<?x4xf8E4M3>,
                      %arg2: f32, %arg3: f8E8M0FNU)
    -> (tensor<?xf6E2M3FN>, tensor<?x4xf32>, f8E8M0FNU, f32, f8E5M3FNU) {
  %0 = arith.truncf %arg0 : tensor<?xf32> to tensor<?xf6E2M3FN>
  %1 = arith.extf %arg1 : tensor<?x4xf8E4M3> to tensor<?x4xf32>
  %2 = arith.truncf %arg2 : f32 to f8E8M0FNU
  %3 = arith.extf %arg3 : f8E8M0FNU to f32
  %4 = arith.truncf %arg2 : f32 to f8E5M3FNU
  return %0, %1, %2, %3, %4 : tensor<?xf6E2M3FN>, tensor<?x4xf32>, f8E8M0FNU,
      f32, f8E5M3FNU
}
