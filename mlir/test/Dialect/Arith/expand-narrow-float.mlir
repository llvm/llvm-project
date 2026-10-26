// RUN: mlir-opt %s --arith-expand="include-f4e2m1=true include-f6e2m3fn=true \
// RUN:     include-f6e3m2fn=true include-f8e3m4=true include-f8e4m3=true \
// RUN:     include-f8e4m3b11fnuz=true include-f8e4m3fn=true \
// RUN:     include-f8e4m3fnuz=true include-f8e5m2=true \
// RUN:     include-f8e5m2fnuz=true" > %t
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=KEPT --implicit-check-not=arith.truncf \
// RUN:   --implicit-check-not=arith.extf --implicit-check-not=vector. < %t
// RUN: mlir-opt %s --arith-expand | \
// RUN:   FileCheck %s --check-prefix=OFF --implicit-check-not=arith.bitcast
// RUN: mlir-opt %s --arith-expand="include-f4e2m1=true" | \
// RUN:   FileCheck %s --check-prefix=F4
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
// RUN: mlir-opt %s --arith-expand="include-f8e4m3fn=true" | \
// RUN:   FileCheck %s --check-prefix=FN43
// RUN: mlir-opt %s --arith-expand="include-f8e4m3fnuz=true" | \
// RUN:   FileCheck %s --check-prefix=FNUZ43
// RUN: mlir-opt %s --arith-expand="include-f8e5m2=true" | \
// RUN:   FileCheck %s --check-prefix=E5M2
// RUN: mlir-opt %s --arith-expand="include-f8e5m2fnuz=true" | \
// RUN:   FileCheck %s --check-prefix=FNUZ52

// Each expansion computes the code in an i32 (i64 from f64 or wider) carrier
// and ends in arith.trunci + arith.bitcast; checks pin the format and rounding
// constants.
// KEPT lists the float ops left, OFF enables no format, the others one each.

// truncf, one function per destination format.

// to_nearest_even is the default (2^24 - 1); 126 is the max-finite code (448)
// and 127 the NaN an overflow gives.
// CHECK-LABEL: func.func @trunc_f32_f8E4M3FN(
// CHECK:         arith.bitcast %arg0 : f32 to i32
// CHECK-DAG:     arith.constant 16777215 : i32
// CHECK-DAG:     arith.constant 126 : i32
// CHECK-DAG:     arith.constant 127 : i32
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i32 to i8
// CHECK-NEXT:    %[[RES:.*]] = arith.bitcast %[[CODE]] : i8 to f8E4M3FN
// CHECK-NEXT:    return %[[RES]] : f8E4M3FN
// OFF:           arith.truncf %arg0 : f32 to f8E4M3FN
// F4-LABEL:    func.func @trunc_f32_f8E4M3FN(
// F4:            arith.truncf %arg0 : f32 to f8E4M3FN
func.func @trunc_f32_f8E4M3FN(%arg0: f32) -> f8E4M3FN {
  %0 = arith.truncf %arg0 : f32 to f8E4M3FN
  return %0 : f8E4M3FN
}

// f16 and f8E5M2 have the same exponent layout (5 bits, bias 15), so
// toward_zero is a shift of the f16 magnitude by 8: no rounding addend and no
// clamp, as it never passes max-finite 123 (57344).
// CHECK-LABEL: func.func @trunc_f16_f8E5M2_toward_zero(
// CHECK:         %[[BITS:.*]] = arith.bitcast %arg0 : f16 to i16
// CHECK-NEXT:    %[[SRC:.*]] = arith.extui %[[BITS]] : i16 to i32
// CHECK-NEXT:    %[[MASK:.*]] = arith.constant 32767 : i32
// CHECK-NEXT:    %[[MAG:.*]] = arith.andi %[[SRC]], %[[MASK]] : i32
// CHECK-NEXT:    %[[C8:.*]] = arith.constant 8 : i32
// CHECK-NEXT:    %[[R:.*]] = arith.shrui %[[MAG]], %[[C8]] : i32
// CHECK-NOT:     arith.{{addi|minui|maxui|maxsi|subi}}
// CHECK:         arith.ori %[[R]], %{{.*}} : i32
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i32 to i8
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i8 to f8E5M2
func.func @trunc_f16_f8E5M2_toward_zero(%arg0: f16) -> f8E5M2 {
  %0 = arith.truncf %arg0 toward_zero : f16 to f8E5M2
  return %0 : f8E5M2
}

// The other modes: the magnitude plus the mode's addend (127 and the kept LSB,
// 128, or 255), shifted by 8; Inf code 124; NaN above the f16 Inf 31744.
// CHECK-LABEL: func.func @trunc_f16_f8E5M2_modes(
// CHECK:         %[[MAG0:.*]] = arith.andi
// CHECK-NOT:     arith.{{minui|maxui|maxsi|subi}}
// CHECK:         %[[C127:.*]] = arith.constant 127 : i32
// CHECK-NEXT:    %[[B0:.*]] = arith.addi %[[MAG0]], %[[C127]] : i32
// CHECK-NEXT:    %[[E0:.*]] = arith.addi %[[B0]], %{{.*}} : i32
// CHECK-NEXT:    %[[S0:.*]] = arith.constant 8 : i32
// CHECK-NEXT:    %[[R0:.*]] = arith.shrui %[[E0]], %[[S0]] : i32
// CHECK-NEXT:    %[[INF:.*]] = arith.constant 31744 : i32
// CHECK-NEXT:    arith.cmpi ugt, %[[MAG0]], %[[INF]] : i32
// CHECK-NOT:     arith.{{minui|maxui|maxsi|subi}}
// CHECK:         arith.ori %[[R0]], %{{.*}} : i32
// CHECK:         arith.bitcast %{{.*}} : i8 to f8E5M2
// CHECK:         %[[MAG1:.*]] = arith.andi
// CHECK-NEXT:    %[[C128:.*]] = arith.constant 128 : i32
// CHECK-NEXT:    arith.addi %[[MAG1]], %[[C128]] : i32
// CHECK-NOT:     arith.{{minui|maxui|maxsi|subi}}
// CHECK:         arith.bitcast %{{.*}} : i8 to f8E5M2
// CHECK:         arith.cmpi ult, %{{.*}}, %{{.*}} : i32
// CHECK:         %[[MAG2:.*]] = arith.andi
// CHECK-NEXT:    %[[C255:.*]] = arith.constant 255 : i32
// CHECK-NEXT:    %[[Z2:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[A2:.*]] = arith.select %{{.*}}, %[[C255]], %[[Z2]] : i32
// CHECK-NEXT:    arith.addi %[[MAG2]], %[[A2]] : i32
// CHECK-NOT:     arith.{{minui|maxui|maxsi|subi}}
// CHECK:         arith.bitcast %{{.*}} : i8 to f8E5M2
// CHECK:         arith.cmpi uge, %{{.*}}, %{{.*}} : i32
// CHECK:         %[[MAG3:.*]] = arith.andi
// CHECK-NEXT:    %[[C255B:.*]] = arith.constant 255 : i32
// CHECK-NEXT:    %[[Z3:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[A3:.*]] = arith.select %{{.*}}, %[[C255B]], %[[Z3]] : i32
// CHECK-NEXT:    arith.addi %[[MAG3]], %[[A3]] : i32
// CHECK-NOT:     arith.{{minui|maxui|maxsi|subi}}
// CHECK:         return
func.func @trunc_f16_f8E5M2_modes(%arg0: f16)
    -> (f8E5M2, f8E5M2, f8E5M2, f8E5M2) {
  %0 = arith.truncf %arg0 : f16 to f8E5M2
  %1 = arith.truncf %arg0 to_nearest_away : f16 to f8E5M2
  %2 = arith.truncf %arg0 upward : f16 to f8E5M2
  %3 = arith.truncf %arg0 downward : f16 to f8E5M2
  return %0, %1, %2, %3 : f8E5M2, f8E5M2, f8E5M2, f8E5M2
}

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

// downward rounds the magnitude up for x < 0 by adding 2^54 - 1 in the i64
// carrier; f4E2M1FN saturates at 7 (6.0).
// CHECK-LABEL: func.func @trunc_f64_f4E2M1FN_downward(
// CHECK:         %[[BITS:.*]] = arith.bitcast %arg0 : f64 to i64
// CHECK-DAG:     arith.cmpi slt, %[[BITS]], %{{.*}} : i64
// CHECK-DAG:     arith.constant 18014398509481983 : i64
// CHECK-DAG:     arith.constant 7 : i64
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i64 to i4
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i4 to f4E2M1FN
// F4-LABEL:    func.func @trunc_f64_f4E2M1FN_downward(
// F4:            arith.bitcast %arg0 : f64 to i64
// F4-NOT:        arith.truncf
// F4:            arith.bitcast %{{.*}} : i4 to f4E2M1FN
func.func @trunc_f64_f4E2M1FN_downward(%arg0: f64) -> f4E2M1FN {
  %0 = arith.truncf %arg0 downward : f64 to f4E2M1FN
  return %0 : f4E2M1FN
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

// 1006632960 = (127 - 7) << 23; 2143289344 = 0x7FC00000.
// CHECK-LABEL: func.func @ext_f8E4M3FN_f32(
// CHECK:         %[[BITS:.*]] = arith.bitcast %arg0 : f8E4M3FN to i8
// CHECK-NEXT:    arith.extui %[[BITS]] : i8 to i32
// CHECK-DAG:     arith.constant 1006632960 : i32
// CHECK-DAG:     arith.constant 2143289344 : i32
// CHECK:         %[[RES:.*]] = arith.bitcast %{{.*}} : i32 to f32
// CHECK-NEXT:    return %[[RES]] : f32
// F4-LABEL:    func.func @ext_f8E4M3FN_f32(
// F4:            arith.extf %arg0 : f8E4M3FN to f32
func.func @ext_f8E4M3FN_f32(%arg0: f8E4M3FN) -> f32 {
  %0 = arith.extf %arg0 : f8E4M3FN to f32
  return %0 : f32
}

// f8E5M2 is the high byte of an f16: the decode is the code, sign included,
// shifted by 8, and a NaN (magnitude above the Inf code 124) also gets the
// quiet bit from 0x7E00 (32256).
// CHECK-LABEL: func.func @ext_f8E5M2_f16(
// CHECK:         %[[SRC:.*]] = arith.extui %{{.*}} : i8 to i32
// CHECK-NEXT:    %[[C127:.*]] = arith.constant 127 : i32
// CHECK-NEXT:    %[[MAG:.*]] = arith.andi %[[SRC]], %[[C127]] : i32
// CHECK-NEXT:    %[[C255:.*]] = arith.constant 255 : i32
// CHECK-NEXT:    %[[CODE:.*]] = arith.andi %[[SRC]], %[[C255]] : i32
// CHECK-NEXT:    %[[C8:.*]] = arith.constant 8 : i32
// CHECK-NEXT:    %[[SHIFTED:.*]] = arith.shli %[[CODE]], %[[C8]] : i32
// CHECK-NEXT:    %[[INF:.*]] = arith.constant 124 : i32
// CHECK-NEXT:    %[[ISNAN:.*]] = arith.cmpi ugt, %[[MAG]], %[[INF]] : i32
// CHECK-NEXT:    %[[QNAN:.*]] = arith.constant 32256 : i32
// CHECK-NEXT:    %[[NAN:.*]] = arith.ori %[[SHIFTED]], %[[QNAN]] : i32
// CHECK-NEXT:    %[[RES:.*]] = arith.select %[[ISNAN]], %[[NAN]], %[[SHIFTED]] : i32
// CHECK-NEXT:    %[[HALF:.*]] = arith.trunci %[[RES]] : i32 to i16
// CHECK-NEXT:    arith.bitcast %[[HALF]] : i16 to f16
func.func @ext_f8E5M2_f16(%arg0: f8E5M2) -> f16 {
  %0 = arith.extf %arg0 : f8E5M2 to f16
  return %0 : f16
}

// bf16 is written directly too, with no float extension in between.
// CHECK-LABEL: func.func @ext_f8E5M2_bf16(
// CHECK-NOT:     arith.extf
// CHECK:         %[[HALF:.*]] = arith.trunci %{{.*}} : i32 to i16
// CHECK-NEXT:    %[[RES:.*]] = arith.bitcast %[[HALF]] : i16 to bf16
// CHECK-NEXT:    return %[[RES]] : bf16
func.func @ext_f8E5M2_bf16(%arg0: f8E5M2) -> bf16 {
  %0 = arith.extf %arg0 : f8E5M2 to bf16
  return %0 : bf16
}

// To f32, f64, f128 and f80 the code is shifted into an f16, the type with its
// exponent layout, in the i32 carrier, and one exact arith.extf with the op's
// fastmath flags takes it further.
// CHECK-LABEL: func.func @ext_f8E5M2_wide(
// CHECK:         arith.extui %{{.*}} : i8 to i32
// CHECK-NOT:     arith.select
// CHECK:         %[[S0:.*]] = arith.constant 8 : i32
// CHECK-NEXT:    arith.shli %{{.*}}, %[[S0]] : i32
// CHECK:         %[[H0:.*]] = arith.trunci %{{.*}} : i32 to i16
// CHECK-NEXT:    %[[F0:.*]] = arith.bitcast %[[H0]] : i16 to f16
// CHECK-NEXT:    %[[R0:.*]] = arith.extf %[[F0]] : f16 to f32
// CHECK:         arith.extui %{{.*}} : i8 to i32
// CHECK-NOT:     i64
// CHECK:         %[[H1:.*]] = arith.trunci %{{.*}} : i32 to i16
// CHECK-NEXT:    %[[F1:.*]] = arith.bitcast %[[H1]] : i16 to f16
// CHECK-NEXT:    %[[R1:.*]] = arith.extf %[[F1]] fastmath<nnan,ninf> : f16 to f64
// CHECK:         %[[H2:.*]] = arith.trunci %{{.*}} : i32 to i16
// CHECK-NEXT:    %[[F2:.*]] = arith.bitcast %[[H2]] : i16 to f16
// CHECK-NEXT:    %[[R2:.*]] = arith.extf %[[F2]] : f16 to f128
// CHECK:         %[[H3:.*]] = arith.trunci %{{.*}} : vector<4xi32> to vector<4xi16>
// CHECK-NEXT:    %[[F3:.*]] = arith.bitcast %[[H3]] : vector<4xi16> to vector<4xf16>
// CHECK-NEXT:    %[[R3:.*]] = arith.extf %[[F3]] : vector<4xf16> to vector<4xf80>
// CHECK-NEXT:    return %[[R0]], %[[R1]], %[[R2]], %[[R3]]
// KEPT:          arith.extf %{{.*}} : f16 to f32
// KEPT:          arith.extf %{{.*}} fastmath<nnan,ninf> : f16 to f64
// KEPT:          arith.extf %{{.*}} : f16 to f128
// KEPT:          arith.extf %{{.*}} : vector<4xf16> to vector<4xf80>
func.func @ext_f8E5M2_wide(%arg0: f8E5M2, %arg1: vector<4xf8E5M2>)
    -> (f32, f64, f128, vector<4xf80>) {
  %0 = arith.extf %arg0 : f8E5M2 to f32
  %1 = arith.extf %arg0 fastmath<nnan,ninf> : f8E5M2 to f64
  %2 = arith.extf %arg0 : f8E5M2 to f128
  %3 = arith.extf %arg1 : vector<4xf8E5M2> to vector<4xf80>
  return %0, %1, %2, %3 : f32, f64, f128, vector<4xf80>
}

// 15360 = (127 - 7) << 7; Inf maps to 0x7F80 (32640), NaNs under 0x7FC0.
// CHECK-LABEL: func.func @ext_f8E4M3_bf16(
// CHECK:         arith.extui %{{.*}} : i8 to i32
// CHECK-DAG:     arith.constant 15360 : i32
// CHECK-DAG:     arith.constant 32640 : i32
// CHECK-DAG:     arith.constant 32704 : i32
// CHECK:         %[[HALF:.*]] = arith.trunci %{{.*}} : i32 to i16
// CHECK-NEXT:    arith.bitcast %[[HALF]] : i16 to bf16
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

// 16128 = (127 - 1) << 7. include-f4e2m1 alone gives the same expansion; the
// codes 0-1 are a multiply by 0x3F00, the bf16 encoding of 0.5.
// CHECK-LABEL: func.func @ext_f4E2M1FN_bf16(
// CHECK:         arith.extui %{{.*}} : i4 to i32
// CHECK-DAG:     arith.constant 16128 : i32
// CHECK:         %[[HALF:.*]] = arith.trunci %{{.*}} : i32 to i16
// CHECK-NEXT:    arith.bitcast %[[HALF]] : i16 to bf16
// F4-LABEL:    func.func @ext_f4E2M1FN_bf16(
// F4:            arith.extui %{{.*}} : i4 to i32
// F4:            arith.muli
// F4-NOT:        arith.truncf
// F4:            arith.bitcast %{{.*}} : i16 to bf16
func.func @ext_f4E2M1FN_bf16(%arg0: f4E2M1FN) -> bf16 {
  %0 = arith.extf %arg0 : f4E2M1FN to bf16
  return %0 : bf16
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

// Narrow to narrow. The source is decoded exactly to f32 bits (939524096 =
// (127 - 15) << 23 for f8E5M2) and rounded once, here downward (2^25 - 1); only
// the destination's option is needed.
// CHECK-LABEL: func.func @trunc_f8E5M2_f4E2M1FN_downward(
// CHECK:         %[[BITS:.*]] = arith.bitcast %arg0 : f8E5M2 to i8
// CHECK-NEXT:    arith.extui %[[BITS]] : i8 to i32
// CHECK-DAG:     arith.constant 939524096 : i32
// CHECK-DAG:     arith.constant 33554431 : i32
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i32 to i4
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i4 to f4E2M1FN
// F4-LABEL:    func.func @trunc_f8E5M2_f4E2M1FN_downward(
// F4:            arith.bitcast %arg0 : f8E5M2 to i8
// F4-NOT:        arith.truncf
// F4:            arith.bitcast %{{.*}} : i4 to f4E2M1FN
func.func @trunc_f8E5M2_f4E2M1FN_downward(%arg0: f8E5M2) -> f4E2M1FN {
  %0 = arith.truncf %arg0 downward : f8E5M2 to f4E2M1FN
  return %0 : f4E2M1FN
}

// The source's option does not enable a truncf: include-f8e4m3b11fnuz alone
// leaves this one.
// CHECK-LABEL: func.func @trunc_f8E4M3B11FNUZ_f4E2M1FN(
// CHECK:         arith.bitcast %arg0 : f8E4M3B11FNUZ to i8
// CHECK:         arith.bitcast %{{.*}} : i4 to f4E2M1FN
// B11-LABEL:   func.func @trunc_f8E4M3B11FNUZ_f4E2M1FN(
// B11:           arith.truncf %arg0 : f8E4M3B11FNUZ to f4E2M1FN
func.func @trunc_f8E4M3B11FNUZ_f4E2M1FN(%arg0: f8E4M3B11FNUZ) -> f4E2M1FN {
  %0 = arith.truncf %arg0 : f8E4M3B11FNUZ to f4E2M1FN
  return %0 : f4E2M1FN
}

// truncf(extf(x)) folds to a narrow-to-narrow truncf before any pattern runs
// (every f8E4M3FNUZ value is exact in f32). Under F4 the extf stays, and the
// truncf is expanded from the f8E4M3FNUZ bits without reading an f32 value.
// CHECK-LABEL: func.func @trunc_ext_f8E4M3FNUZ_f4E2M1FN(
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i32 to i4
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i4 to f4E2M1FN
// F4-LABEL:    func.func @trunc_ext_f8E4M3FNUZ_f4E2M1FN(
// F4:            arith.extf %arg0 : f8E4M3FNUZ to f32
// F4:            arith.bitcast %arg0 : f8E4M3FNUZ to i8
// F4-NOT:        arith.bitcast {{.*}} : f32 to i32
// F4:            arith.bitcast %{{.*}} : i4 to f4E2M1FN
func.func @trunc_ext_f8E4M3FNUZ_f4E2M1FN(%arg0: f8E4M3FNUZ) -> f4E2M1FN {
  %0 = arith.extf %arg0 : f8E4M3FNUZ to f32
  %1 = arith.truncf %0 : f32 to f4E2M1FN
  return %1 : f4E2M1FN
}

// An f8E8M0FNU source (no zero, which the extension needs) is decoded by its
// exponent shift, code 0x00 being 2^-127 (the f32 subnormal 4194304).
// CHECK-LABEL: func.func @trunc_f8E8M0FNU_f4E2M1FN_toward_zero(
// CHECK:         %[[BITS:.*]] = arith.bitcast %arg0 : f8E8M0FNU to i8
// CHECK:         arith.extui %[[BITS]] : i8 to i32
// CHECK:         arith.constant 4194304 : i32
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i32 to i4
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i4 to f4E2M1FN
func.func @trunc_f8E8M0FNU_f4E2M1FN_toward_zero(%arg0: f8E8M0FNU) -> f4E2M1FN {
  %0 = arith.truncf %arg0 toward_zero : f8E8M0FNU to f4E2M1FN
  return %0 : f4E2M1FN
}

// A narrow-to-narrow extf decodes f4E2M1FN to f32 bits (0.5 = 1056964608) and
// encodes f8E5M2 (overflow clamp 124), exactly; either option enables it.
// CHECK-LABEL: func.func @ext_f4E2M1FN_f8E5M2(
// CHECK:         %[[BITS:.*]] = arith.bitcast %arg0 : f4E2M1FN to i4
// CHECK-NEXT:    arith.extui %[[BITS]] : i4 to i32
// CHECK-DAG:     arith.constant 1056964608 : i32
// CHECK-DAG:     arith.constant 124 : i32
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i32 to i8
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i8 to f8E5M2
// F4-LABEL:    func.func @ext_f4E2M1FN_f8E5M2(
// F4:            arith.bitcast %arg0 : f4E2M1FN to i4
// F4-NOT:        arith.extf
// F4:            arith.bitcast %{{.*}} : i8 to f8E5M2
func.func @ext_f4E2M1FN_f8E5M2(%arg0: f4E2M1FN) -> f8E5M2 {
  %0 = arith.extf %arg0 : f4E2M1FN to f8E5M2
  return %0 : f8E5M2
}

// An extf from f4E2M1FN into a format without a zero (f8E8M0FNU) or without a
// sign (f8E5M3FNU) decodes to f32 (0.5 = 1056964608) and ends in one
// arith.truncf, which only include-f8e8m0 expands.
// CHECK-LABEL: func.func @ext_f4E2M1FN_no_zero_or_sign(
// CHECK:         %[[BITS:.*]] = arith.bitcast %arg0 : f4E2M1FN to i4
// CHECK-NEXT:    arith.extui %[[BITS]] : i4 to i32
// CHECK-DAG:     arith.constant 1056964608 : i32
// CHECK:         %[[W0:.*]] = arith.bitcast %{{.*}} : i32 to f32
// CHECK-NEXT:    %[[R0:.*]] = arith.truncf %[[W0]] : f32 to f8E8M0FNU
// CHECK:         %[[W1:.*]] = arith.bitcast %{{.*}} : i32 to f32
// CHECK-NEXT:    %[[R1:.*]] = arith.truncf %[[W1]] : f32 to f8E5M3FNU
// CHECK-NEXT:    return %[[R0]], %[[R1]]
// KEPT:          arith.truncf %{{.*}} : f32 to f8E8M0FNU
// KEPT:          arith.truncf %{{.*}} : f32 to f8E5M3FNU
// F4-LABEL:    func.func @ext_f4E2M1FN_no_zero_or_sign(
// F4:            arith.bitcast %arg0 : f4E2M1FN to i4
// F4-NOT:        arith.extf
// F4:            arith.truncf %{{.*}} : f32 to f8E8M0FNU
// F4-NOT:        arith.extf
// F4:            arith.truncf %{{.*}} : f32 to f8E5M3FNU
func.func @ext_f4E2M1FN_no_zero_or_sign(%arg0: f4E2M1FN)
    -> (f8E8M0FNU, f8E5M3FNU) {
  %0 = arith.extf %arg0 : f4E2M1FN to f8E8M0FNU
  %1 = arith.extf %arg0 : f4E2M1FN to f8E5M3FNU
  return %0, %1 : f8E8M0FNU, f8E5M3FNU
}

// fastmath flags carry over to the float ops the expansion keeps, but those
// before the rounding keep only nnan (with contract, ExtFOp::fold would fold
// the widening of a contract truncf back to the truncf's operand): the exact
// widening of an f16 source into f8E5M2FNUZ, the extf of an f80, and the truncf
// and extf that round an f128 to odd, where nnan also drops the NaN rebuild.
// The final truncf of an extf into f8E8M0FNU keeps every flag.
// CHECK-LABEL: func.func @fastmath(
// CHECK:         arith.extf %arg0 fastmath<nnan> : f16 to f32
// CHECK:         arith.truncf %{{.*}} fastmath<nnan,contract> : f32 to f8E8M0FNU
// CHECK:         %[[R:.*]] = arith.truncf %arg2 fastmath<nnan> : f128 to f64
// CHECK-NEXT:    arith.extf %[[R]] fastmath<nnan> : f64 to f128
// CHECK-NOT:     arith.constant 60 : i128
// CHECK:         %[[W:.*]] = arith.extf %arg3 : f80 to f128
// CHECK-NEXT:    %[[R3:.*]] = arith.truncf %[[W]] : f128 to f64
// CHECK-NEXT:    arith.extf %[[R3]] : f64 to f128
// CHECK:         arith.constant 60 : i128
// KEPT:          arith.extf %arg0 fastmath<nnan> : f16 to f32
// KEPT:          arith.truncf %{{.*}} fastmath<nnan,contract> : f32 to f8E8M0FNU
// KEPT:          arith.truncf %arg2 fastmath<nnan> : f128 to f64
// KEPT:          arith.extf %{{.*}} fastmath<nnan> : f64 to f128
// KEPT:          arith.extf %arg3 : f80 to f128
// KEPT:          arith.truncf %{{.*}} : f128 to f64
// KEPT:          arith.extf %{{.*}} : f64 to f128
func.func @fastmath(%arg0: f16, %arg1: f4E2M1FN, %arg2: f128, %arg3: f80)
    -> (f8E5M2FNUZ, f8E8M0FNU, f8E4M3FN, f8E4M3FN) {
  %0 = arith.truncf %arg0 fastmath<nnan,contract> : f16 to f8E5M2FNUZ
  %1 = arith.extf %arg1 fastmath<nnan,contract> : f4E2M1FN to f8E8M0FNU
  %2 = arith.truncf %arg2 fastmath<nnan,contract> : f128 to f8E4M3FN
  %3 = arith.truncf %arg3 fastmath<contract> : f80 to f8E4M3FN
  return %0, %1, %2, %3 : f8E5M2FNUZ, f8E8M0FNU, f8E4M3FN, f8E4M3FN
}

// Each option enables its own format and no other: the one expanded truncf is
// the only one whose result is a bitcast from an integer.
// F4-LABEL:     func.func @one_truncf_per_format(
// F4-NOT:         : i{{[0-9]+}} to f
// F4:             : i4 to f4E2M1FN{{$}}
// F4-NOT:         : i{{[0-9]+}} to f
// F4:             return
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
// FN43-LABEL:   func.func @one_truncf_per_format(
// FN43-NOT:       : i{{[0-9]+}} to f
// FN43:           : i8 to f8E4M3FN{{$}}
// FN43-NOT:       : i{{[0-9]+}} to f
// FN43:           return
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
// E5M2-LABEL:   func.func @one_truncf_per_format(
// E5M2-NOT:       : i{{[0-9]+}} to f
// E5M2:           : i8 to f8E5M2{{$}}
// E5M2-NOT:       : i{{[0-9]+}} to f
// E5M2:           return
// FNUZ52-LABEL: func.func @one_truncf_per_format(
// FNUZ52-NOT:     : i{{[0-9]+}} to f
// FNUZ52:         : i8 to f8E5M2FNUZ{{$}}
// FNUZ52-NOT:     : i{{[0-9]+}} to f
// FNUZ52:         return
func.func @one_truncf_per_format(%arg0: f32)
    -> (f4E2M1FN, f6E2M3FN, f6E3M2FN, f8E3M4, f8E4M3, f8E4M3FN, f8E4M3FNUZ,
        f8E4M3B11FNUZ, f8E5M2, f8E5M2FNUZ) {
  %0 = arith.truncf %arg0 : f32 to f4E2M1FN
  %1 = arith.truncf %arg0 : f32 to f6E2M3FN
  %2 = arith.truncf %arg0 : f32 to f6E3M2FN
  %3 = arith.truncf %arg0 : f32 to f8E3M4
  %4 = arith.truncf %arg0 : f32 to f8E4M3
  %5 = arith.truncf %arg0 : f32 to f8E4M3FN
  %6 = arith.truncf %arg0 : f32 to f8E4M3FNUZ
  %7 = arith.truncf %arg0 : f32 to f8E4M3B11FNUZ
  %8 = arith.truncf %arg0 : f32 to f8E5M2
  %9 = arith.truncf %arg0 : f32 to f8E5M2FNUZ
  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9
      : f4E2M1FN, f6E2M3FN, f6E3M2FN, f8E3M4, f8E4M3, f8E4M3FN, f8E4M3FNUZ,
        f8E4M3B11FNUZ, f8E5M2, f8E5M2FNUZ
}

// Shapes: one copy of the expansion serves any static or scalable shape, with
// splat constants (one trunci for the whole vector, one multiply in the
// f6E3M2FN decode); the f8E5M2 extf goes through f16.
// CHECK-LABEL: func.func @shapes(
// CHECK:         arith.bitcast %arg0 : vector<4xf32> to vector<4xi32>
// CHECK-NOT:     arith.trunci
// CHECK:         arith.constant dense<126> : vector<4xi32>
// CHECK-NOT:     arith.trunci
// CHECK:         arith.trunci %{{.*}} : vector<4xi32> to vector<4xi8>
// CHECK-NEXT:    arith.bitcast %{{.*}} : vector<4xi8> to vector<4xf8E4M3FN>
// CHECK:         arith.extui %{{.*}} : vector<8xi8> to vector<8xi32>
// CHECK-NOT:     arith.muli
// CHECK:         arith.trunci %{{.*}} : vector<8xi32> to vector<8xi16>
// CHECK-NEXT:    %[[V8:.*]] = arith.bitcast %{{.*}} : vector<8xi16> to vector<8xf16>
// CHECK-NEXT:    arith.extf %[[V8]] : vector<8xf16> to vector<8xf32>
// CHECK:         arith.bitcast %arg2 : vector<2x3xf32> to vector<2x3xi32>
// CHECK:         arith.trunci %{{.*}} : vector<2x3xi32> to vector<2x3xi6>
// CHECK:         arith.bitcast %arg3 : vector<f32> to vector<i32>
// CHECK:         arith.trunci %{{.*}} : vector<i32> to vector<i8>
// CHECK:         arith.bitcast %arg4 : tensor<4xf64> to tensor<4xi64>
// CHECK:         arith.constant dense<7> : tensor<4xi64>
// CHECK:         arith.trunci %{{.*}} : tensor<4xi64> to tensor<4xi4>
// CHECK:         arith.extui %{{.*}} : tensor<2x2xi8> to tensor<2x2xi32>
// CHECK:         arith.bitcast %{{.*}} : tensor<2x2xi32> to tensor<2x2xf32>
// CHECK:         arith.bitcast %arg6 : vector<[4]xf32> to vector<[4]xi32>
// CHECK:         arith.constant dense<124> : vector<[4]xi32>
// CHECK:         arith.trunci %{{.*}} : vector<[4]xi32> to vector<[4]xi8>
// CHECK:         arith.extui %{{.*}} : vector<[8]xi6> to vector<[8]xi32>
// CHECK-NOT:     arith.muli
// CHECK:         arith.muli %{{.*}}, %{{.*}} : vector<[8]xi32>
// CHECK-NOT:     arith.muli
// CHECK:         arith.bitcast %{{.*}} : vector<[8]xi32> to vector<[8]xf32>
// KEPT:          arith.extf %{{.*}} : vector<8xf16> to vector<8xf32>
func.func @shapes(%arg0: vector<4xf32>, %arg1: vector<8xf8E5M2>,
                  %arg2: vector<2x3xf32>, %arg3: vector<f32>,
                  %arg4: tensor<4xf64>, %arg5: tensor<2x2xf8E4M3FN>,
                  %arg6: vector<[4]xf32>, %arg7: vector<[8]xf6E3M2FN>)
    -> (vector<4xf8E4M3FN>, vector<8xf32>, vector<2x3xf6E2M3FN>,
        vector<f8E5M2>, tensor<4xf4E2M1FN>, tensor<2x2xf32>,
        vector<[4]xf8E5M2>, vector<[8]xf32>) {
  %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xf8E4M3FN>
  %1 = arith.extf %arg1 : vector<8xf8E5M2> to vector<8xf32>
  %2 = arith.truncf %arg2 : vector<2x3xf32> to vector<2x3xf6E2M3FN>
  %3 = arith.truncf %arg3 : vector<f32> to vector<f8E5M2>
  %4 = arith.truncf %arg4 : tensor<4xf64> to tensor<4xf4E2M1FN>
  %5 = arith.extf %arg5 : tensor<2x2xf8E4M3FN> to tensor<2x2xf32>
  %6 = arith.truncf %arg6 : vector<[4]xf32> to vector<[4]xf8E5M2>
  %7 = arith.extf %arg7 : vector<[8]xf6E3M2FN> to vector<[8]xf32>
  return %0, %1, %2, %3, %4, %5, %6, %7
      : vector<4xf8E4M3FN>, vector<8xf32>, vector<2x3xf6E2M3FN>, vector<f8E5M2>,
        tensor<4xf4E2M1FN>, tensor<2x2xf32>, vector<[4]xf8E5M2>, vector<[8]xf32>
}

// Partners outside f16 / bf16 / f32 / f64: a tf32 source is extended exactly
// to f32 first; an extf to f128 or f80 is the expansion to f64 followed by an
// exact arith.extf, and one to tf32 goes through f16 the same way.
// CHECK-LABEL: func.func @wide_partners(
// CHECK:         %[[WIDE:.*]] = arith.extf %arg0 : tf32 to f32
// CHECK:         arith.bitcast %[[WIDE]] : f32 to i32
// CHECK:         arith.bitcast %{{.*}} : i8 to f8E4M3FN
// CHECK:         arith.extui %{{.*}} : i8 to i64
// CHECK:         %[[W0:.*]] = arith.bitcast %{{.*}} : i64 to f64
// CHECK-NEXT:    %[[R0:.*]] = arith.extf %[[W0]] : f64 to f128
// CHECK:         arith.extui %{{.*}} : vector<4xi4> to vector<4xi64>
// CHECK:         %[[W1:.*]] = arith.bitcast %{{.*}} : vector<4xi64> to vector<4xf64>
// CHECK-NEXT:    %[[R1:.*]] = arith.extf %[[W1]] : vector<4xf64> to vector<4xf80>
// CHECK:         arith.trunci %{{.*}} : i32 to i16
// CHECK-NEXT:    %[[W2:.*]] = arith.bitcast %{{.*}} : i16 to f16
// CHECK-NEXT:    %[[R2:.*]] = arith.extf %[[W2]] : f16 to tf32
// CHECK-NEXT:    return %{{.*}}, %[[R0]], %[[R1]], %[[R2]]
// KEPT:          arith.extf %arg0 : tf32 to f32
// KEPT:          arith.extf %{{.*}} : f64 to f128
// KEPT:          arith.extf %{{.*}} : vector<4xf64> to vector<4xf80>
// KEPT:          arith.extf %{{.*}} : f16 to tf32
func.func @wide_partners(%arg0: tf32, %arg1: f8E4M3FN, %arg2: vector<4xf4E2M1FN>,
                         %arg3: f8E5M2)
    -> (f8E4M3FN, f128, vector<4xf80>, tf32) {
  %0 = arith.truncf %arg0 : tf32 to f8E4M3FN
  %1 = arith.extf %arg1 : f8E4M3FN to f128
  %2 = arith.extf %arg2 : vector<4xf4E2M1FN> to vector<4xf80>
  %3 = arith.extf %arg3 : f8E5M2 to tf32
  return %0, %1, %2, %3 : f8E4M3FN, f128, vector<4xf80>, tf32
}

// A truncf from f128 is first rounded to odd into f64: the f64 truncf,
// extended back, shows whether it was inexact (other bits) and rounded up
// (larger bits, as the signs agree); a rounded-up result is stepped down and an
// inexact one gets an odd LSB. A NaN is rebuilt from the f128 bits (sign, top
// 52 payload bits, quiet NaN 0x7FF8000000000000); 2^127 - 1 masks the sign off
// and 0x7FFF << 112 is the f128 Inf. The f64 expansion then rounds once
// (9007199254740991 = 2^53 - 1, the f64 to_nearest_even addend).
// CHECK-LABEL: func.func @trunc_f128_f8E5M2(
// CHECK:         %[[R:.*]] = arith.truncf %arg0 : f128 to f64
// CHECK-NEXT:    %[[BACK:.*]] = arith.extf %[[R]] : f64 to f128
// CHECK-NEXT:    %[[X:.*]] = arith.bitcast %arg0 : f128 to i128
// CHECK-NEXT:    %[[B:.*]] = arith.bitcast %[[BACK]] : f128 to i128
// CHECK-NEXT:    %[[INEXACT:.*]] = arith.cmpi ne, %[[B]], %[[X]] : i128
// CHECK-NEXT:    %[[UP:.*]] = arith.cmpi ugt, %[[B]], %[[X]] : i128
// CHECK-NEXT:    %[[BITS:.*]] = arith.bitcast %[[R]] : f64 to i64
// CHECK-NEXT:    %[[UP64:.*]] = arith.extui %[[UP]] : i1 to i64
// CHECK-NEXT:    %[[TZ:.*]] = arith.subi %[[BITS]], %[[UP64]] : i64
// CHECK-NEXT:    %[[STICKY:.*]] = arith.extui %[[INEXACT]] : i1 to i64
// CHECK-NEXT:    %[[ODD:.*]] = arith.ori %[[TZ]], %[[STICKY]] : i64
// CHECK-NEXT:    %[[C60:.*]] = arith.constant 60 : i128
// CHECK-NEXT:    %[[SH:.*]] = arith.shrui %[[X]], %[[C60]] : i128
// CHECK-NEXT:    %[[TOP:.*]] = arith.trunci %[[SH]] : i128 to i64
// CHECK-NEXT:    %[[PMASK:.*]] = arith.constant 4503599627370495 : i64
// CHECK-NEXT:    %[[PAYLOAD:.*]] = arith.andi %[[TOP]], %[[PMASK]] : i64
// CHECK-NEXT:    %[[ZERO:.*]] = arith.constant 0 : i128
// CHECK-NEXT:    %[[NEG:.*]] = arith.cmpi slt, %[[X]], %[[ZERO]] : i128
// CHECK-NEXT:    %[[NQ:.*]] = arith.constant -2251799813685248 : i64
// CHECK-NEXT:    %[[PQ:.*]] = arith.constant 9221120237041090560 : i64
// CHECK-NEXT:    %[[Q:.*]] = arith.select %[[NEG]], %[[NQ]], %[[PQ]] : i64
// CHECK-NEXT:    %[[NAN:.*]] = arith.ori %[[PAYLOAD]], %[[Q]] : i64
// CHECK-NEXT:    %[[AMASK:.*]] = arith.constant 170141183460469231731687303715884105727 : i128
// CHECK-NEXT:    %[[MAG:.*]] = arith.andi %[[X]], %[[AMASK]] : i128
// CHECK-NEXT:    %[[INF:.*]] = arith.constant 170135991163610696904058773219554885632 : i128
// CHECK-NEXT:    %[[ISNAN:.*]] = arith.cmpi ugt, %[[MAG]], %[[INF]] : i128
// CHECK-NEXT:    %[[SRC:.*]] = arith.select %[[ISNAN]], %[[NAN]], %[[ODD]] : i64
// CHECK-NEXT:    %[[M:.*]] = arith.constant 9223372036854775807 : i64
// CHECK-NEXT:    arith.andi %[[SRC]], %[[M]] : i64
// CHECK:         arith.constant 9007199254740991 : i64
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i64 to i8
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i8 to f8E5M2
// KEPT:          arith.truncf %arg0 : f128 to f64
// KEPT:          arith.extf %{{.*}} : f64 to f128
// OFF:           arith.truncf %arg0 : f128 to f8E5M2
func.func @trunc_f128_f8E5M2(%arg0: f128) -> f8E5M2 {
  %0 = arith.truncf %arg0 : f128 to f8E5M2
  return %0 : f8E5M2
}

// An f80 source is extended exactly to f128 first; upward then rounds the f64
// magnitude up for x >= 0 (2^54 - 1). A vector takes the same ops.
// CHECK-LABEL: func.func @trunc_wide_sources(
// CHECK:         %[[W:.*]] = arith.extf %arg0 : f80 to f128
// CHECK-NEXT:    %[[R:.*]] = arith.truncf %[[W]] : f128 to f64
// CHECK-NEXT:    arith.extf %[[R]] : f64 to f128
// CHECK:         arith.cmpi ugt, %{{.*}}, %{{.*}} : i128
// CHECK:         arith.cmpi ugt, %{{.*}}, %{{.*}} : i128
// CHECK-NEXT:    %[[SRC:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : i64
// CHECK-NEXT:    %[[Z:.*]] = arith.constant 0 : i64
// CHECK-NEXT:    arith.cmpi sge, %[[SRC]], %[[Z]] : i64
// CHECK:         arith.constant 18014398509481983 : i64
// CHECK:         %[[CODE:.*]] = arith.trunci %{{.*}} : i64 to i8
// CHECK-NEXT:    arith.bitcast %[[CODE]] : i8 to f8E4M3FN
// CHECK:         arith.truncf %arg1 : vector<4xf128> to vector<4xf64>
// CHECK:         arith.constant dense<60> : vector<4xi128>
// CHECK:         arith.trunci %{{.*}} : vector<4xi64> to vector<4xi6>
// CHECK-NEXT:    arith.bitcast %{{.*}} : vector<4xi6> to vector<4xf6E3M2FN>
// KEPT:          arith.extf %arg0 : f80 to f128
// KEPT:          arith.truncf %{{.*}} : f128 to f64
// KEPT:          arith.extf %{{.*}} : f64 to f128
// KEPT:          arith.truncf %arg1 : vector<4xf128> to vector<4xf64>
// KEPT:          arith.extf %{{.*}} : vector<4xf64> to vector<4xf128>
func.func @trunc_wide_sources(%arg0: f80, %arg1: vector<4xf128>)
    -> (f8E4M3FN, vector<4xf6E3M2FN>) {
  %0 = arith.truncf %arg0 upward : f80 to f8E4M3FN
  %1 = arith.truncf %arg1 toward_zero : vector<4xf128> to vector<4xf6E3M2FN>
  return %0, %1 : f8E4M3FN, vector<4xf6E3M2FN>
}

// Left alone: dynamic shapes (no splat constant), f8E8M0FNU (only
// include-f8e8m0 expands it, see expand-ops.mlir) and f8E5M3FNU (no option).
// KEPT:          arith.truncf %arg0 : tensor<?xf32> to tensor<?xf4E2M1FN>
// KEPT:          arith.extf %arg1 : tensor<?x4xf8E4M3FN> to tensor<?x4xf32>
// KEPT:          arith.truncf %arg2 : f32 to f8E8M0FNU
// KEPT:          arith.extf %arg3 : f8E8M0FNU to f32
// KEPT:          arith.truncf %arg2 : f32 to f8E5M3FNU
func.func @left_alone(%arg0: tensor<?xf32>, %arg1: tensor<?x4xf8E4M3FN>,
                      %arg2: f32, %arg3: f8E8M0FNU)
    -> (tensor<?xf4E2M1FN>, tensor<?x4xf32>, f8E8M0FNU, f32, f8E5M3FNU) {
  %0 = arith.truncf %arg0 : tensor<?xf32> to tensor<?xf4E2M1FN>
  %1 = arith.extf %arg1 : tensor<?x4xf8E4M3FN> to tensor<?x4xf32>
  %2 = arith.truncf %arg2 : f32 to f8E8M0FNU
  %3 = arith.extf %arg3 : f8E8M0FNU to f32
  %4 = arith.truncf %arg2 : f32 to f8E5M3FNU
  return %0, %1, %2, %3, %4 : tensor<?xf4E2M1FN>, tensor<?x4xf32>, f8E8M0FNU,
      f32, f8E5M3FNU
}
