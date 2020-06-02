; RUN: llc < %s -asm-verbose=0 -mtriple=aarch64-none-eabi | FileCheck %s

define <4 x i16> @v4bf16_to_v4i16(float, <4 x bfloat> %a) nounwind {
; CHECK-LABEL: v4bf16_to_v4i16:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <4 x bfloat> %a to <4 x i16>
  ret <4 x i16> %1
}

define <2 x i32> @v4bf16_to_v2i32(float, <4 x bfloat> %a) nounwind {
; CHECK-LABEL: v4bf16_to_v2i32:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <4 x bfloat> %a to <2 x i32>
  ret <2 x i32> %1
}

define <1 x i64> @v4bf16_to_v1i64(float, <4 x bfloat> %a) nounwind {
; CHECK-LABEL: v4bf16_to_v1i64:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <4 x bfloat> %a to <1 x i64>
  ret <1 x i64> %1
}

define i64 @v4bf16_to_i64(float, <4 x bfloat> %a) nounwind {
; CHECK-LABEL: v4bf16_to_i64:
; CHECK-NEXT: fmov x0, d1
; CHECK-NEXT: ret
entry:
  %1 = bitcast <4 x bfloat> %a to i64
  ret i64 %1
}

define <2 x float> @v4bf16_to_v2float(float, <4 x bfloat> %a) nounwind {
; CHECK-LABEL: v4bf16_to_v2float:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <4 x bfloat> %a to <2 x float>
  ret <2 x float> %1
}

define <1 x double> @v4bf16_to_v1double(float, <4 x bfloat> %a) nounwind {
; CHECK-LABEL: v4bf16_to_v1double:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <4 x bfloat> %a to <1 x double>
  ret <1 x double> %1
}

define double @v4bf16_to_double(float, <4 x bfloat> %a) nounwind {
; CHECK-LABEL: v4bf16_to_double:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <4 x bfloat> %a to double
  ret double %1
}


define <4 x bfloat> @v4i16_to_v4bf16(float, <4 x i16> %a) nounwind {
; CHECK-LABEL: v4i16_to_v4bf16:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <4 x i16> %a to <4 x bfloat>
  ret <4 x bfloat> %1
}

define <4 x bfloat> @v2i32_to_v4bf16(float, <2 x i32> %a) nounwind {
; CHECK-LABEL: v2i32_to_v4bf16:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <2 x i32> %a to <4 x bfloat>
  ret <4 x bfloat> %1
}

define <4 x bfloat> @v1i64_to_v4bf16(float, <1 x i64> %a) nounwind {
; CHECK-LABEL: v1i64_to_v4bf16:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <1 x i64> %a to <4 x bfloat>
  ret <4 x bfloat> %1
}

define <4 x bfloat> @i64_to_v4bf16(float, i64 %a) nounwind {
; CHECK-LABEL: i64_to_v4bf16:
; CHECK-NEXT: fmov d0, x0
; CHECK-NEXT: ret
entry:
  %1 = bitcast i64 %a to <4 x bfloat>
  ret <4 x bfloat> %1
}

define <4 x bfloat> @v2float_to_v4bf16(float, <2 x float> %a) nounwind {
; CHECK-LABEL: v2float_to_v4bf16:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <2 x float> %a to <4 x bfloat>
  ret <4 x bfloat> %1
}

define <4 x bfloat> @v1double_to_v4bf16(float, <1 x double> %a) nounwind {
; CHECK-LABEL: v1double_to_v4bf16:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <1 x double> %a to <4 x bfloat>
  ret <4 x bfloat> %1
}

define <4 x bfloat> @double_to_v4bf16(float, double %a) nounwind {
; CHECK-LABEL: double_to_v4bf16:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast double %a to <4 x bfloat>
  ret <4 x bfloat> %1
}

define <8 x i16> @v8bf16_to_v8i16(float, <8 x bfloat> %a) nounwind {
; CHECK-LABEL: v8bf16_to_v8i16:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <8 x bfloat> %a to <8 x i16>
  ret <8 x i16> %1
}

define <4 x i32> @v8bf16_to_v4i32(float, <8 x bfloat> %a) nounwind {
; CHECK-LABEL: v8bf16_to_v4i32:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <8 x bfloat> %a to <4 x i32>
  ret <4 x i32> %1
}

define <2 x i64> @v8bf16_to_v2i64(float, <8 x bfloat> %a) nounwind {
; CHECK-LABEL: v8bf16_to_v2i64:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <8 x bfloat> %a to <2 x i64>
  ret <2 x i64> %1
}

define <4 x float> @v8bf16_to_v4float(float, <8 x bfloat> %a) nounwind {
; CHECK-LABEL: v8bf16_to_v4float:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <8 x bfloat> %a to <4 x float>
  ret <4 x float> %1
}

define <2 x double> @v8bf16_to_v2double(float, <8 x bfloat> %a) nounwind {
; CHECK-LABEL: v8bf16_to_v2double:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <8 x bfloat> %a to <2 x double>
  ret <2 x double> %1
}

define <8 x bfloat> @v8i16_to_v8bf16(float, <8 x i16> %a) nounwind {
; CHECK-LABEL: v8i16_to_v8bf16:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <8 x i16> %a to <8 x bfloat>
  ret <8 x bfloat> %1
}

define <8 x bfloat> @v4i32_to_v8bf16(float, <4 x i32> %a) nounwind {
; CHECK-LABEL: v4i32_to_v8bf16:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <4 x i32> %a to <8 x bfloat>
  ret <8 x bfloat> %1
}

define <8 x bfloat> @v2i64_to_v8bf16(float, <2 x i64> %a) nounwind {
; CHECK-LABEL: v2i64_to_v8bf16:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <2 x i64> %a to <8 x bfloat>
  ret <8 x bfloat> %1
}

define <8 x bfloat> @v4float_to_v8bf16(float, <4 x float> %a) nounwind {
; CHECK-LABEL: v4float_to_v8bf16:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <4 x float> %a to <8 x bfloat>
  ret <8 x bfloat> %1
}

define <8 x bfloat> @v2double_to_v8bf16(float, <2 x double> %a) nounwind {
; CHECK-LABEL: v2double_to_v8bf16:
; CHECK-NEXT: mov v0.16b, v1.16b
; CHECK-NEXT: ret
entry:
  %1 = bitcast <2 x double> %a to <8 x bfloat>
  ret <8 x bfloat> %1
}
