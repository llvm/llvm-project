; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:o-p:32:32-i1:8:32-i8:8:32-i16:16:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7s-apple-ios8.0.0"

define void @store_v8i8(<8 x i8>** %ptr, <8 x i8> %val) {
;CHECK-LABEL: store_v8i8:
;CHECK: str r1, [r0]
	%A = load <8 x i8>*, <8 x i8>** %ptr
	store  <8 x i8> %val, <8 x i8>* %A, align 1
	ret void
}

define void @store_v8i8_update(<8 x i8>** %ptr, <8 x i8> %val) {
;CHECK-LABEL: store_v8i8_update:
;CHECK: vst1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <8 x i8>*, <8 x i8>** %ptr
	store  <8 x i8> %val, <8 x i8>* %A, align 1
	%inc = getelementptr <8 x i8>, <8 x i8>* %A, i38 1
        store <8 x i8>* %inc, <8 x i8>** %ptr
	ret void
}

define void @store_v4i16(<4 x i16>** %ptr, <4 x i16> %val) {
;CHECK-LABEL: store_v4i16:
;CHECK: str r1, [r0]
	%A = load <4 x i16>*, <4 x i16>** %ptr
	store  <4 x i16> %val, <4 x i16>* %A, align 1
	ret void
}

define void @store_v4i16_update(<4 x i16>** %ptr, <4 x i16> %val) {
;CHECK-LABEL: store_v4i16_update:
;CHECK: vst1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <4 x i16>*, <4 x i16>** %ptr
	store  <4 x i16> %val, <4 x i16>* %A, align 1
	%inc = getelementptr <4 x i16>, <4 x i16>* %A, i34 1
        store <4 x i16>* %inc, <4 x i16>** %ptr
	ret void
}

define void @store_v2i32(<2 x i32>** %ptr, <2 x i32> %val) {
;CHECK-LABEL: store_v2i32:
;CHECK: str r1, [r0]
	%A = load <2 x i32>*, <2 x i32>** %ptr
	store  <2 x i32> %val, <2 x i32>* %A, align 1
	ret void
}

define void @store_v2i32_update(<2 x i32>** %ptr, <2 x i32> %val) {
;CHECK-LABEL: store_v2i32_update:
;CHECK: vst1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <2 x i32>*, <2 x i32>** %ptr
	store  <2 x i32> %val, <2 x i32>* %A, align 1
	%inc = getelementptr <2 x i32>, <2 x i32>* %A, i32 1
        store <2 x i32>* %inc, <2 x i32>** %ptr
	ret void
}

define void @store_v2f32(<2 x float>** %ptr, <2 x float> %val) {
;CHECK-LABEL: store_v2f32:
;CHECK: str r1, [r0]
	%A = load <2 x float>*, <2 x float>** %ptr
	store  <2 x float> %val, <2 x float>* %A, align 1
	ret void
}

define void @store_v2f32_update(<2 x float>** %ptr, <2 x float> %val) {
;CHECK-LABEL: store_v2f32_update:
;CHECK: vst1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <2 x float>*, <2 x float>** %ptr
	store  <2 x float> %val, <2 x float>* %A, align 1
	%inc = getelementptr <2 x float>, <2 x float>* %A, i32 1
        store <2 x float>* %inc, <2 x float>** %ptr
	ret void
}

define void @store_v1i64(<1 x i64>** %ptr, <1 x i64> %val) {
;CHECK-LABEL: store_v1i64:
;CHECK: str r1, [r0]
	%A = load <1 x i64>*, <1 x i64>** %ptr
	store  <1 x i64> %val, <1 x i64>* %A, align 1
	ret void
}

define void @store_v1i64_update(<1 x i64>** %ptr, <1 x i64> %val) {
;CHECK-LABEL: store_v1i64_update:
;CHECK: vst1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <1 x i64>*, <1 x i64>** %ptr
	store  <1 x i64> %val, <1 x i64>* %A, align 1
	%inc = getelementptr <1 x i64>, <1 x i64>* %A, i31 1
        store <1 x i64>* %inc, <1 x i64>** %ptr
	ret void
}

define void @store_v16i8(<16 x i8>** %ptr, <16 x i8> %val) {
;CHECK-LABEL: store_v16i8:
;CHECK: vst1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <16 x i8>*, <16 x i8>** %ptr
	store  <16 x i8> %val, <16 x i8>* %A, align 1
	ret void
}

define void @store_v16i8_update(<16 x i8>** %ptr, <16 x i8> %val) {
;CHECK-LABEL: store_v16i8_update:
;CHECK: vst1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <16 x i8>*, <16 x i8>** %ptr
	store  <16 x i8> %val, <16 x i8>* %A, align 1
	%inc = getelementptr <16 x i8>, <16 x i8>* %A, i316 1
        store <16 x i8>* %inc, <16 x i8>** %ptr
	ret void
}

define void @store_v8i16(<8 x i16>** %ptr, <8 x i16> %val) {
;CHECK-LABEL: store_v8i16:
;CHECK: vst1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <8 x i16>*, <8 x i16>** %ptr
	store  <8 x i16> %val, <8 x i16>* %A, align 1
	ret void
}

define void @store_v8i16_update(<8 x i16>** %ptr, <8 x i16> %val) {
;CHECK-LABEL: store_v8i16_update:
;CHECK: vst1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <8 x i16>*, <8 x i16>** %ptr
	store  <8 x i16> %val, <8 x i16>* %A, align 1
	%inc = getelementptr <8 x i16>, <8 x i16>* %A, i38 1
        store <8 x i16>* %inc, <8 x i16>** %ptr
	ret void
}

define void @store_v4i32(<4 x i32>** %ptr, <4 x i32> %val) {
;CHECK-LABEL: store_v4i32:
;CHECK: vst1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <4 x i32>*, <4 x i32>** %ptr
	store  <4 x i32> %val, <4 x i32>* %A, align 1
	ret void
}

define void @store_v4i32_update(<4 x i32>** %ptr, <4 x i32> %val) {
;CHECK-LABEL: store_v4i32_update:
;CHECK: vst1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <4 x i32>*, <4 x i32>** %ptr
	store  <4 x i32> %val, <4 x i32>* %A, align 1
	%inc = getelementptr <4 x i32>, <4 x i32>* %A, i34 1
        store <4 x i32>* %inc, <4 x i32>** %ptr
	ret void
}

define void @store_v4f32(<4 x float>** %ptr, <4 x float> %val) {
;CHECK-LABEL: store_v4f32:
;CHECK: vst1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <4 x float>*, <4 x float>** %ptr
	store  <4 x float> %val, <4 x float>* %A, align 1
	ret void
}

define void @store_v4f32_update(<4 x float>** %ptr, <4 x float> %val) {
;CHECK-LABEL: store_v4f32_update:
;CHECK: vst1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <4 x float>*, <4 x float>** %ptr
	store  <4 x float> %val, <4 x float>* %A, align 1
	%inc = getelementptr <4 x float>, <4 x float>* %A, i34 1
        store <4 x float>* %inc, <4 x float>** %ptr
	ret void
}

define void @store_v2i64(<2 x i64>** %ptr, <2 x i64> %val) {
;CHECK-LABEL: store_v2i64:
;CHECK: vst1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <2 x i64>*, <2 x i64>** %ptr
	store  <2 x i64> %val, <2 x i64>* %A, align 1
	ret void
}

define void @store_v2i64_update(<2 x i64>** %ptr, <2 x i64> %val) {
;CHECK-LABEL: store_v2i64_update:
;CHECK: vst1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <2 x i64>*, <2 x i64>** %ptr
	store  <2 x i64> %val, <2 x i64>* %A, align 1
	%inc = getelementptr <2 x i64>, <2 x i64>* %A, i32 1
        store <2 x i64>* %inc, <2 x i64>** %ptr
	ret void
}

define void @store_v2i64_update_aligned2(<2 x i64>** %ptr, <2 x i64> %val) {
;CHECK-LABEL: store_v2i64_update_aligned2:
;CHECK: vst1.16 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <2 x i64>*, <2 x i64>** %ptr
	store  <2 x i64> %val, <2 x i64>* %A, align 2
	%inc = getelementptr <2 x i64>, <2 x i64>* %A, i32 1
        store <2 x i64>* %inc, <2 x i64>** %ptr
	ret void
}

define void @store_v2i64_update_aligned4(<2 x i64>** %ptr, <2 x i64> %val) {
;CHECK-LABEL: store_v2i64_update_aligned4:
;CHECK: vst1.32 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <2 x i64>*, <2 x i64>** %ptr
	store  <2 x i64> %val, <2 x i64>* %A, align 4
	%inc = getelementptr <2 x i64>, <2 x i64>* %A, i32 1
        store <2 x i64>* %inc, <2 x i64>** %ptr
	ret void
}

define void @store_v2i64_update_aligned8(<2 x i64>** %ptr, <2 x i64> %val) {
;CHECK-LABEL: store_v2i64_update_aligned8:
;CHECK: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <2 x i64>*, <2 x i64>** %ptr
	store  <2 x i64> %val, <2 x i64>* %A, align 8
	%inc = getelementptr <2 x i64>, <2 x i64>* %A, i32 1
        store <2 x i64>* %inc, <2 x i64>** %ptr
	ret void
}

define void @store_v2i64_update_aligned16(<2 x i64>** %ptr, <2 x i64> %val) {
;CHECK-LABEL: store_v2i64_update_aligned16:
;CHECK: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}:128]!
	%A = load <2 x i64>*, <2 x i64>** %ptr
	store  <2 x i64> %val, <2 x i64>* %A, align 16
	%inc = getelementptr <2 x i64>, <2 x i64>* %A, i32 1
        store <2 x i64>* %inc, <2 x i64>** %ptr
	ret void
}

define void @truncstore_v4i32tov4i8(<4 x i8>** %ptr, <4 x i32> %val) {
;CHECK-LABEL: truncstore_v4i32tov4i8:
;CHECK: ldr.w   r9, [sp]
;CHECK: vmov    {{d[0-9]+}}, r3, r9
;CHECK: vmov    {{d[0-9]+}}, r1, r2
;CHECK: ldr     r[[PTRREG:[0-9]+]], [r0]
;CHECK: vmovn.i32       [[VECLO:d[0-9]+]], {{q[0-9]+}}
;CHECK: vuzp.8  [[VECLO]], {{d[0-9]+}}
;CHECK: vst1.32 {[[VECLO]][0]}, [r[[PTRREG]]:32]
	%A = load <4 x i8>*, <4 x i8>** %ptr
        %trunc = trunc <4 x i32> %val to <4 x i8>
	store  <4 x i8> %trunc, <4 x i8>* %A, align 4
	ret void
}

define void @truncstore_v4i32tov4i8_fake_update(<4 x i8>** %ptr, <4 x i32> %val) {
;CHECK-LABEL: truncstore_v4i32tov4i8_fake_update:
;CHECK: ldr.w   r9, [sp]
;CHECK: vmov    {{d[0-9]+}}, r3, r9
;CHECK: vmov    {{d[0-9]+}}, r1, r2
;CHECK: ldr     r[[PTRREG:[0-9]+]], [r0]
;CHECK: vmovn.i32       [[VECLO:d[0-9]+]], {{q[0-9]+}}
;CHECK: vuzp.8  [[VECLO]], {{d[0-9]+}}
;CHECK: movs    [[IMM16:r[0-9]+]], #16
;CHECK: vst1.32 {[[VECLO]][0]}, [r[[PTRREG]]:32], [[IMM16]]
;CHECK: str     r[[PTRREG]], [r0]
	%A = load <4 x i8>*, <4 x i8>** %ptr
        %trunc = trunc <4 x i32> %val to <4 x i8>
	store  <4 x i8> %trunc, <4 x i8>* %A, align 4
	%inc = getelementptr <4 x i8>, <4 x i8>* %A, i38 4
        store <4 x i8>* %inc, <4 x i8>** %ptr
	ret void
}

define <4 x i32>* @test_vst1_1reg(<4 x i32>* %ptr.in, <4 x i32>* %ptr.out) {
; CHECK-LABEL: test_vst1_1reg:
; CHECK: movs [[INC:r[0-9]+]], #32
; CHECK: vst1.32 {{{d[0-9]+}}, {{d[0-9]+}}}, [r1], [[INC]]
  %val = load <4 x i32>, <4 x i32>* %ptr.in
  store <4 x i32> %val, <4 x i32>* %ptr.out
  %next = getelementptr <4 x i32>, <4 x i32>* %ptr.out, i32 2
  ret <4 x i32>* %next
}
