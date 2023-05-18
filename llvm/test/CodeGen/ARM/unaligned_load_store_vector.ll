;RUN: llc -mtriple=arm-eabi -mattr=+v7 -mattr=+neon %s -o - | FileCheck %s

;ALIGN = 1
;SIZE  = 64
;TYPE  = <8 x i8>
define void @v64_v8i8_1(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v64_v8i8_1:
entry:
;CHECK: vld1.8
  %v1 = load  <8 x i8>,  ptr %in, align 1
;CHECK: vst1.8
  store <8 x i8> %v1, ptr %out, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 64
;TYPE  = <4 x i16>
define void @v64_v4i16_1(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v64_v4i16_1:
entry:
;CHECK: vld1.8
  %v1 = load  <4 x i16>,  ptr %in, align 1
;CHECK: vst1.8
  store <4 x i16> %v1, ptr %out, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 64
;TYPE  = <2 x i32>
define void @v64_v2i32_1(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v64_v2i32_1:
entry:
;CHECK: vld1.8
  %v1 = load  <2 x i32>,  ptr %in, align 1
;CHECK: vst1.8
  store <2 x i32> %v1, ptr %out, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 64
;TYPE  = <2 x float>
define void @v64_v2f32_1(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v64_v2f32_1:
entry:
;CHECK: vld1.8
  %v1 = load  <2 x float>,  ptr %in, align 1
;CHECK: vst1.8
  store <2 x float> %v1, ptr %out, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 128
;TYPE  = <16 x i8>
define void @v128_v16i8_1(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v16i8_1:
entry:
;CHECK: vld1.8
  %v1 = load  <16 x i8>,  ptr %in, align 1
;CHECK: vst1.8
  store <16 x i8> %v1, ptr %out, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 128
;TYPE  = <8 x i16>
define void @v128_v8i16_1(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v8i16_1:
entry:
;CHECK: vld1.8
  %v1 = load  <8 x i16>,  ptr %in, align 1
;CHECK: vst1.8
  store <8 x i16> %v1, ptr %out, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 128
;TYPE  = <4 x i32>
define void @v128_v4i32_1(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v4i32_1:
entry:
;CHECK: vld1.8
  %v1 = load  <4 x i32>,  ptr %in, align 1
;CHECK: vst1.8
  store <4 x i32> %v1, ptr %out, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 128
;TYPE  = <2 x i64>
define void @v128_v2i64_1(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v2i64_1:
entry:
;CHECK: vld1.8
  %v1 = load  <2 x i64>,  ptr %in, align 1
;CHECK: vst1.8
  store <2 x i64> %v1, ptr %out, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 128
;TYPE  = <4 x float>
define void @v128_v4f32_1(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v4f32_1:
entry:
;CHECK: vld1.8
  %v1 = load  <4 x float>,  ptr %in, align 1
;CHECK: vst1.8
  store <4 x float> %v1, ptr %out, align 1
  ret void
}


;ALIGN = 2
;SIZE  = 64
;TYPE  = <8 x i8>
define void @v64_v8i8_2(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v64_v8i8_2:
entry:
;CHECK: vld1.16
  %v1 = load  <8 x i8>,  ptr %in, align 2
;CHECK: vst1.16
  store <8 x i8> %v1, ptr %out, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 64
;TYPE  = <4 x i16>
define void @v64_v4i16_2(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v64_v4i16_2:
entry:
;CHECK: vld1.16
  %v1 = load  <4 x i16>,  ptr %in, align 2
;CHECK: vst1.16
  store <4 x i16> %v1, ptr %out, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 64
;TYPE  = <2 x i32>
define void @v64_v2i32_2(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v64_v2i32_2:
entry:
;CHECK: vld1.16
  %v1 = load  <2 x i32>,  ptr %in, align 2
;CHECK: vst1.16
  store <2 x i32> %v1, ptr %out, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 64
;TYPE  = <2 x float>
define void @v64_v2f32_2(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v64_v2f32_2:
entry:
;CHECK: vld1.16
  %v1 = load  <2 x float>,  ptr %in, align 2
;CHECK: vst1.16
  store <2 x float> %v1, ptr %out, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 128
;TYPE  = <16 x i8>
define void @v128_v16i8_2(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v16i8_2:
entry:
;CHECK: vld1.16
  %v1 = load  <16 x i8>,  ptr %in, align 2
;CHECK: vst1.16
  store <16 x i8> %v1, ptr %out, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 128
;TYPE  = <8 x i16>
define void @v128_v8i16_2(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v8i16_2:
entry:
;CHECK: vld1.16
  %v1 = load  <8 x i16>,  ptr %in, align 2
;CHECK: vst1.16
  store <8 x i16> %v1, ptr %out, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 128
;TYPE  = <4 x i32>
define void @v128_v4i32_2(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v4i32_2:
entry:
;CHECK: vld1.16
  %v1 = load  <4 x i32>,  ptr %in, align 2
;CHECK: vst1.16
  store <4 x i32> %v1, ptr %out, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 128
;TYPE  = <2 x i64>
define void @v128_v2i64_2(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v2i64_2:
entry:
;CHECK: vld1.16
  %v1 = load  <2 x i64>,  ptr %in, align 2
;CHECK: vst1.16
  store <2 x i64> %v1, ptr %out, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 128
;TYPE  = <4 x float>
define void @v128_v4f32_2(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v4f32_2:
entry:
;CHECK: vld1.16
  %v1 = load  <4 x float>,  ptr %in, align 2
;CHECK: vst1.16
  store <4 x float> %v1, ptr %out, align 2
  ret void
}


;ALIGN = 4
;SIZE  = 64
;TYPE  = <8 x i8>
define void @v64_v8i8_4(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v64_v8i8_4:
entry:
;CHECK: vldr
  %v1 = load  <8 x i8>,  ptr %in, align 4
;CHECK: vstr
  store <8 x i8> %v1, ptr %out, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 64
;TYPE  = <4 x i16>
define void @v64_v4i16_4(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v64_v4i16_4:
entry:
;CHECK: vldr
  %v1 = load  <4 x i16>,  ptr %in, align 4
;CHECK: vstr
  store <4 x i16> %v1, ptr %out, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 64
;TYPE  = <2 x i32>
define void @v64_v2i32_4(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v64_v2i32_4:
entry:
;CHECK: vldr
  %v1 = load  <2 x i32>,  ptr %in, align 4
;CHECK: vstr
  store <2 x i32> %v1, ptr %out, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 64
;TYPE  = <2 x float>
define void @v64_v2f32_4(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v64_v2f32_4:
entry:
;CHECK: vldr
  %v1 = load  <2 x float>,  ptr %in, align 4
;CHECK: vstr
  store <2 x float> %v1, ptr %out, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 128
;TYPE  = <16 x i8>
define void @v128_v16i8_4(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v16i8_4:
entry:
;CHECK: vld1.32
  %v1 = load  <16 x i8>,  ptr %in, align 4
;CHECK: vst1.32
  store <16 x i8> %v1, ptr %out, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 128
;TYPE  = <8 x i16>
define void @v128_v8i16_4(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v8i16_4:
entry:
;CHECK: vld1.32
  %v1 = load  <8 x i16>,  ptr %in, align 4
;CHECK: vst1.32
  store <8 x i16> %v1, ptr %out, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 128
;TYPE  = <4 x i32>
define void @v128_v4i32_4(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v4i32_4:
entry:
;CHECK: vld1.32
  %v1 = load  <4 x i32>,  ptr %in, align 4
;CHECK: vst1.32
  store <4 x i32> %v1, ptr %out, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 128
;TYPE  = <2 x i64>
define void @v128_v2i64_4(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v2i64_4:
entry:
;CHECK: vld1.32
  %v1 = load  <2 x i64>,  ptr %in, align 4
;CHECK: vst1.32
  store <2 x i64> %v1, ptr %out, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 128
;TYPE  = <4 x float>
define void @v128_v4f32_4(ptr noalias nocapture %out, ptr noalias nocapture %in) nounwind {
;CHECK-LABEL: v128_v4f32_4:
entry:
;CHECK: vld1.32
  %v1 = load  <4 x float>,  ptr %in, align 4
;CHECK: vst1.32
  store <4 x float> %v1, ptr %out, align 4
  ret void
}

define void @test_weird_type(<3 x double> %in, ptr %ptr) {
; CHECK-LABEL: test_weird_type:
; CHECK: vst1

  %vec.int = bitcast <3 x double> %in to <3 x i64>
  store <3 x i64> %vec.int, ptr %ptr, align 8
  ret void
}
