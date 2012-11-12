;RUN: llc < %s -march=arm -mattr=+v7 -mattr=+neon | FileCheck %s

;ALIGN = 1
;SIZE  = 64
;TYPE  = <8 x i8>
define void @v64_v8i8_1(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v64_v8i8_1:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <8 x i8>*
  %vo  = bitcast i8* %po to <8 x i8>*
;CHECK: vld1.8
  %v1 = load  <8 x i8>* %vi, align 1
;CHECK: vst1.8
  store <8 x i8> %v1, <8 x i8>* %vo, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 64
;TYPE  = <4 x i16>
define void @v64_v4i16_1(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v64_v4i16_1:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <4 x i16>*
  %vo  = bitcast i8* %po to <4 x i16>*
;CHECK: vld1.8
  %v1 = load  <4 x i16>* %vi, align 1
;CHECK: vst1.8
  store <4 x i16> %v1, <4 x i16>* %vo, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 64
;TYPE  = <2 x i32>
define void @v64_v2i32_1(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v64_v2i32_1:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <2 x i32>*
  %vo  = bitcast i8* %po to <2 x i32>*
;CHECK: vld1.8
  %v1 = load  <2 x i32>* %vi, align 1
;CHECK: vst1.8
  store <2 x i32> %v1, <2 x i32>* %vo, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 64
;TYPE  = <2 x float>
define void @v64_v2f32_1(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v64_v2f32_1:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <2 x float>*
  %vo  = bitcast i8* %po to <2 x float>*
;CHECK: vld1.8
  %v1 = load  <2 x float>* %vi, align 1
;CHECK: vst1.8
  store <2 x float> %v1, <2 x float>* %vo, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 128
;TYPE  = <16 x i8>
define void @v128_v16i8_1(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v16i8_1:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <16 x i8>*
  %vo  = bitcast i8* %po to <16 x i8>*
;CHECK: vld1.8
  %v1 = load  <16 x i8>* %vi, align 1
;CHECK: vst1.8
  store <16 x i8> %v1, <16 x i8>* %vo, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 128
;TYPE  = <8 x i16>
define void @v128_v8i16_1(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v8i16_1:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <8 x i16>*
  %vo  = bitcast i8* %po to <8 x i16>*
;CHECK: vld1.8
  %v1 = load  <8 x i16>* %vi, align 1
;CHECK: vst1.8
  store <8 x i16> %v1, <8 x i16>* %vo, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 128
;TYPE  = <4 x i32>
define void @v128_v4i32_1(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v4i32_1:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <4 x i32>*
  %vo  = bitcast i8* %po to <4 x i32>*
;CHECK: vld1.8
  %v1 = load  <4 x i32>* %vi, align 1
;CHECK: vst1.8
  store <4 x i32> %v1, <4 x i32>* %vo, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 128
;TYPE  = <2 x i64>
define void @v128_v2i64_1(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v2i64_1:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <2 x i64>*
  %vo  = bitcast i8* %po to <2 x i64>*
;CHECK: vld1.8
  %v1 = load  <2 x i64>* %vi, align 1
;CHECK: vst1.8
  store <2 x i64> %v1, <2 x i64>* %vo, align 1
  ret void
}


;ALIGN = 1
;SIZE  = 128
;TYPE  = <4 x float>
define void @v128_v4f32_1(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v4f32_1:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <4 x float>*
  %vo  = bitcast i8* %po to <4 x float>*
;CHECK: vld1.8
  %v1 = load  <4 x float>* %vi, align 1
;CHECK: vst1.8
  store <4 x float> %v1, <4 x float>* %vo, align 1
  ret void
}


;ALIGN = 2
;SIZE  = 64
;TYPE  = <8 x i8>
define void @v64_v8i8_2(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v64_v8i8_2:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <8 x i8>*
  %vo  = bitcast i8* %po to <8 x i8>*
;CHECK: vld1.16
  %v1 = load  <8 x i8>* %vi, align 2
;CHECK: vst1.16
  store <8 x i8> %v1, <8 x i8>* %vo, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 64
;TYPE  = <4 x i16>
define void @v64_v4i16_2(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v64_v4i16_2:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <4 x i16>*
  %vo  = bitcast i8* %po to <4 x i16>*
;CHECK: vld1.16
  %v1 = load  <4 x i16>* %vi, align 2
;CHECK: vst1.16
  store <4 x i16> %v1, <4 x i16>* %vo, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 64
;TYPE  = <2 x i32>
define void @v64_v2i32_2(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v64_v2i32_2:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <2 x i32>*
  %vo  = bitcast i8* %po to <2 x i32>*
;CHECK: vld1.16
  %v1 = load  <2 x i32>* %vi, align 2
;CHECK: vst1.16
  store <2 x i32> %v1, <2 x i32>* %vo, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 64
;TYPE  = <2 x float>
define void @v64_v2f32_2(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v64_v2f32_2:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <2 x float>*
  %vo  = bitcast i8* %po to <2 x float>*
;CHECK: vld1.16
  %v1 = load  <2 x float>* %vi, align 2
;CHECK: vst1.16
  store <2 x float> %v1, <2 x float>* %vo, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 128
;TYPE  = <16 x i8>
define void @v128_v16i8_2(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v16i8_2:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <16 x i8>*
  %vo  = bitcast i8* %po to <16 x i8>*
;CHECK: vld1.16
  %v1 = load  <16 x i8>* %vi, align 2
;CHECK: vst1.16
  store <16 x i8> %v1, <16 x i8>* %vo, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 128
;TYPE  = <8 x i16>
define void @v128_v8i16_2(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v8i16_2:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <8 x i16>*
  %vo  = bitcast i8* %po to <8 x i16>*
;CHECK: vld1.16
  %v1 = load  <8 x i16>* %vi, align 2
;CHECK: vst1.16
  store <8 x i16> %v1, <8 x i16>* %vo, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 128
;TYPE  = <4 x i32>
define void @v128_v4i32_2(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v4i32_2:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <4 x i32>*
  %vo  = bitcast i8* %po to <4 x i32>*
;CHECK: vld1.16
  %v1 = load  <4 x i32>* %vi, align 2
;CHECK: vst1.16
  store <4 x i32> %v1, <4 x i32>* %vo, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 128
;TYPE  = <2 x i64>
define void @v128_v2i64_2(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v2i64_2:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <2 x i64>*
  %vo  = bitcast i8* %po to <2 x i64>*
;CHECK: vld1.16
  %v1 = load  <2 x i64>* %vi, align 2
;CHECK: vst1.16
  store <2 x i64> %v1, <2 x i64>* %vo, align 2
  ret void
}


;ALIGN = 2
;SIZE  = 128
;TYPE  = <4 x float>
define void @v128_v4f32_2(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v4f32_2:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <4 x float>*
  %vo  = bitcast i8* %po to <4 x float>*
;CHECK: vld1.16
  %v1 = load  <4 x float>* %vi, align 2
;CHECK: vst1.16
  store <4 x float> %v1, <4 x float>* %vo, align 2
  ret void
}


;ALIGN = 4
;SIZE  = 64
;TYPE  = <8 x i8>
define void @v64_v8i8_4(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v64_v8i8_4:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <8 x i8>*
  %vo  = bitcast i8* %po to <8 x i8>*
;CHECK: vldr
  %v1 = load  <8 x i8>* %vi, align 4
;CHECK: vstr
  store <8 x i8> %v1, <8 x i8>* %vo, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 64
;TYPE  = <4 x i16>
define void @v64_v4i16_4(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v64_v4i16_4:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <4 x i16>*
  %vo  = bitcast i8* %po to <4 x i16>*
;CHECK: vldr
  %v1 = load  <4 x i16>* %vi, align 4
;CHECK: vstr
  store <4 x i16> %v1, <4 x i16>* %vo, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 64
;TYPE  = <2 x i32>
define void @v64_v2i32_4(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v64_v2i32_4:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <2 x i32>*
  %vo  = bitcast i8* %po to <2 x i32>*
;CHECK: vldr
  %v1 = load  <2 x i32>* %vi, align 4
;CHECK: vstr
  store <2 x i32> %v1, <2 x i32>* %vo, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 64
;TYPE  = <2 x float>
define void @v64_v2f32_4(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v64_v2f32_4:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <2 x float>*
  %vo  = bitcast i8* %po to <2 x float>*
;CHECK: vldr
  %v1 = load  <2 x float>* %vi, align 4
;CHECK: vstr
  store <2 x float> %v1, <2 x float>* %vo, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 128
;TYPE  = <16 x i8>
define void @v128_v16i8_4(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v16i8_4:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <16 x i8>*
  %vo  = bitcast i8* %po to <16 x i8>*
;CHECK: vld1.32
  %v1 = load  <16 x i8>* %vi, align 4
;CHECK: vst1.32
  store <16 x i8> %v1, <16 x i8>* %vo, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 128
;TYPE  = <8 x i16>
define void @v128_v8i16_4(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v8i16_4:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <8 x i16>*
  %vo  = bitcast i8* %po to <8 x i16>*
;CHECK: vld1.32
  %v1 = load  <8 x i16>* %vi, align 4
;CHECK: vst1.32
  store <8 x i16> %v1, <8 x i16>* %vo, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 128
;TYPE  = <4 x i32>
define void @v128_v4i32_4(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v4i32_4:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <4 x i32>*
  %vo  = bitcast i8* %po to <4 x i32>*
;CHECK: vld1.32
  %v1 = load  <4 x i32>* %vi, align 4
;CHECK: vst1.32
  store <4 x i32> %v1, <4 x i32>* %vo, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 128
;TYPE  = <2 x i64>
define void @v128_v2i64_4(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v2i64_4:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <2 x i64>*
  %vo  = bitcast i8* %po to <2 x i64>*
;CHECK: vld1.32
  %v1 = load  <2 x i64>* %vi, align 4
;CHECK: vst1.32
  store <2 x i64> %v1, <2 x i64>* %vo, align 4
  ret void
}


;ALIGN = 4
;SIZE  = 128
;TYPE  = <4 x float>
define void @v128_v4f32_4(i8* noalias nocapture %out, i8* noalias nocapture %in) nounwind {
;CHECK: v128_v4f32_4:
entry:
  %po = getelementptr i8* %out, i32 0
  %pi = getelementptr i8* %in,  i32 0
  %vi  = bitcast i8* %pi to <4 x float>*
  %vo  = bitcast i8* %po to <4 x float>*
;CHECK: vld1.32
  %v1 = load  <4 x float>* %vi, align 4
;CHECK: vst1.32
  store <4 x float> %v1, <4 x float>* %vo, align 4
  ret void
}

