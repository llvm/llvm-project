; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:o-p:32:32-i1:8:32-i8:8:32-i16:16:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7s-apple-ios8.0.0"

define <8 x i8> @load_v8i8(<8 x i8>** %ptr) {
;CHECK-LABEL: load_v8i8:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <8 x i8>*, <8 x i8>** %ptr
	%lA = load <8 x i8>, <8 x i8>* %A, align 1
	ret <8 x i8> %lA
}

define <8 x i8> @load_v8i8_update(<8 x i8>** %ptr) {
;CHECK-LABEL: load_v8i8_update:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <8 x i8>*, <8 x i8>** %ptr
	%lA = load <8 x i8>, <8 x i8>* %A, align 1
	%inc = getelementptr <8 x i8>, <8 x i8>* %A, i38 1
        store <8 x i8>* %inc, <8 x i8>** %ptr
	ret <8 x i8> %lA
}

define <4 x i16> @load_v4i16(<4 x i16>** %ptr) {
;CHECK-LABEL: load_v4i16:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <4 x i16>*, <4 x i16>** %ptr
	%lA = load <4 x i16>, <4 x i16>* %A, align 1
	ret <4 x i16> %lA
}

define <4 x i16> @load_v4i16_update(<4 x i16>** %ptr) {
;CHECK-LABEL: load_v4i16_update:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <4 x i16>*, <4 x i16>** %ptr
	%lA = load <4 x i16>, <4 x i16>* %A, align 1
	%inc = getelementptr <4 x i16>, <4 x i16>* %A, i34 1
        store <4 x i16>* %inc, <4 x i16>** %ptr
	ret <4 x i16> %lA
}

define <2 x i32> @load_v2i32(<2 x i32>** %ptr) {
;CHECK-LABEL: load_v2i32:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <2 x i32>*, <2 x i32>** %ptr
	%lA = load <2 x i32>, <2 x i32>* %A, align 1
	ret <2 x i32> %lA
}

define <2 x i32> @load_v2i32_update(<2 x i32>** %ptr) {
;CHECK-LABEL: load_v2i32_update:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <2 x i32>*, <2 x i32>** %ptr
	%lA = load <2 x i32>, <2 x i32>* %A, align 1
	%inc = getelementptr <2 x i32>, <2 x i32>* %A, i32 1
        store <2 x i32>* %inc, <2 x i32>** %ptr
	ret <2 x i32> %lA
}

define <2 x float> @load_v2f32(<2 x float>** %ptr) {
;CHECK-LABEL: load_v2f32:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <2 x float>*, <2 x float>** %ptr
	%lA = load <2 x float>, <2 x float>* %A, align 1
	ret <2 x float> %lA
}

define <2 x float> @load_v2f32_update(<2 x float>** %ptr) {
;CHECK-LABEL: load_v2f32_update:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <2 x float>*, <2 x float>** %ptr
	%lA = load <2 x float>, <2 x float>* %A, align 1
	%inc = getelementptr <2 x float>, <2 x float>* %A, i32 1
        store <2 x float>* %inc, <2 x float>** %ptr
	ret <2 x float> %lA
}

define <1 x i64> @load_v1i64(<1 x i64>** %ptr) {
;CHECK-LABEL: load_v1i64:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <1 x i64>*, <1 x i64>** %ptr
	%lA = load <1 x i64>, <1 x i64>* %A, align 1
	ret <1 x i64> %lA
}

define <1 x i64> @load_v1i64_update(<1 x i64>** %ptr) {
;CHECK-LABEL: load_v1i64_update:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <1 x i64>*, <1 x i64>** %ptr
	%lA = load <1 x i64>, <1 x i64>* %A, align 1
	%inc = getelementptr <1 x i64>, <1 x i64>* %A, i31 1
        store <1 x i64>* %inc, <1 x i64>** %ptr
	ret <1 x i64> %lA
}

define <16 x i8> @load_v16i8(<16 x i8>** %ptr) {
;CHECK-LABEL: load_v16i8:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <16 x i8>*, <16 x i8>** %ptr
	%lA = load <16 x i8>, <16 x i8>* %A, align 1
	ret <16 x i8> %lA
}

define <16 x i8> @load_v16i8_update(<16 x i8>** %ptr) {
;CHECK-LABEL: load_v16i8_update:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <16 x i8>*, <16 x i8>** %ptr
	%lA = load <16 x i8>, <16 x i8>* %A, align 1
	%inc = getelementptr <16 x i8>, <16 x i8>* %A, i316 1
        store <16 x i8>* %inc, <16 x i8>** %ptr
	ret <16 x i8> %lA
}

define <8 x i16> @load_v8i16(<8 x i16>** %ptr) {
;CHECK-LABEL: load_v8i16:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <8 x i16>*, <8 x i16>** %ptr
	%lA = load <8 x i16>, <8 x i16>* %A, align 1
	ret <8 x i16> %lA
}

define <8 x i16> @load_v8i16_update(<8 x i16>** %ptr) {
;CHECK-LABEL: load_v8i16_update:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <8 x i16>*, <8 x i16>** %ptr
	%lA = load <8 x i16>, <8 x i16>* %A, align 1
	%inc = getelementptr <8 x i16>, <8 x i16>* %A, i38 1
        store <8 x i16>* %inc, <8 x i16>** %ptr
	ret <8 x i16> %lA
}

define <4 x i32> @load_v4i32(<4 x i32>** %ptr) {
;CHECK-LABEL: load_v4i32:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <4 x i32>*, <4 x i32>** %ptr
	%lA = load <4 x i32>, <4 x i32>* %A, align 1
	ret <4 x i32> %lA
}

define <4 x i32> @load_v4i32_update(<4 x i32>** %ptr) {
;CHECK-LABEL: load_v4i32_update:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <4 x i32>*, <4 x i32>** %ptr
	%lA = load <4 x i32>, <4 x i32>* %A, align 1
	%inc = getelementptr <4 x i32>, <4 x i32>* %A, i34 1
        store <4 x i32>* %inc, <4 x i32>** %ptr
	ret <4 x i32> %lA
}

define <4 x float> @load_v4f32(<4 x float>** %ptr) {
;CHECK-LABEL: load_v4f32:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <4 x float>*, <4 x float>** %ptr
	%lA = load <4 x float>, <4 x float>* %A, align 1
	ret <4 x float> %lA
}

define <4 x float> @load_v4f32_update(<4 x float>** %ptr) {
;CHECK-LABEL: load_v4f32_update:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <4 x float>*, <4 x float>** %ptr
	%lA = load <4 x float>, <4 x float>* %A, align 1
	%inc = getelementptr <4 x float>, <4 x float>* %A, i34 1
        store <4 x float>* %inc, <4 x float>** %ptr
	ret <4 x float> %lA
}

define <2 x i64> @load_v2i64(<2 x i64>** %ptr) {
;CHECK-LABEL: load_v2i64:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load <2 x i64>*, <2 x i64>** %ptr
	%lA = load <2 x i64>, <2 x i64>* %A, align 1
	ret <2 x i64> %lA
}

define <2 x i64> @load_v2i64_update(<2 x i64>** %ptr) {
;CHECK-LABEL: load_v2i64_update:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <2 x i64>*, <2 x i64>** %ptr
	%lA = load <2 x i64>, <2 x i64>* %A, align 1
	%inc = getelementptr <2 x i64>, <2 x i64>* %A, i32 1
        store <2 x i64>* %inc, <2 x i64>** %ptr
	ret <2 x i64> %lA
}

; Make sure we change the type to match alignment if necessary.
define <2 x i64> @load_v2i64_update_aligned2(<2 x i64>** %ptr) {
;CHECK-LABEL: load_v2i64_update_aligned2:
;CHECK: vld1.16 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <2 x i64>*, <2 x i64>** %ptr
	%lA = load <2 x i64>, <2 x i64>* %A, align 2
	%inc = getelementptr <2 x i64>, <2 x i64>* %A, i32 1
        store <2 x i64>* %inc, <2 x i64>** %ptr
	ret <2 x i64> %lA
}

define <2 x i64> @load_v2i64_update_aligned4(<2 x i64>** %ptr) {
;CHECK-LABEL: load_v2i64_update_aligned4:
;CHECK: vld1.32 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <2 x i64>*, <2 x i64>** %ptr
	%lA = load <2 x i64>, <2 x i64>* %A, align 4
	%inc = getelementptr <2 x i64>, <2 x i64>* %A, i32 1
        store <2 x i64>* %inc, <2 x i64>** %ptr
	ret <2 x i64> %lA
}

define <2 x i64> @load_v2i64_update_aligned8(<2 x i64>** %ptr) {
;CHECK-LABEL: load_v2i64_update_aligned8:
;CHECK: vld1.64 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load <2 x i64>*, <2 x i64>** %ptr
	%lA = load <2 x i64>, <2 x i64>* %A, align 8
	%inc = getelementptr <2 x i64>, <2 x i64>* %A, i32 1
        store <2 x i64>* %inc, <2 x i64>** %ptr
	ret <2 x i64> %lA
}

define <2 x i64> @load_v2i64_update_aligned16(<2 x i64>** %ptr) {
;CHECK-LABEL: load_v2i64_update_aligned16:
;CHECK: vld1.64 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}:128]!
	%A = load <2 x i64>*, <2 x i64>** %ptr
	%lA = load <2 x i64>, <2 x i64>* %A, align 16
	%inc = getelementptr <2 x i64>, <2 x i64>* %A, i32 1
        store <2 x i64>* %inc, <2 x i64>** %ptr
	ret <2 x i64> %lA
}

; Make sure we don't break smaller-than-dreg extloads.
define <4 x i32> @zextload_v8i8tov8i32(<4 x i8>** %ptr) {
;CHECK-LABEL: zextload_v8i8tov8i32:
;CHECK: vld1.32 {{{d[0-9]+}}[0]}, [{{r[0-9]+}}:32]
;CHECK: vmovl.u8        {{q[0-9]+}}, {{d[0-9]+}}
;CHECK: vmovl.u16       {{q[0-9]+}}, {{d[0-9]+}}
	%A = load <4 x i8>*, <4 x i8>** %ptr
	%lA = load <4 x i8>, <4 x i8>* %A, align 4
        %zlA = zext <4 x i8> %lA to <4 x i32>
	ret <4 x i32> %zlA
}

define <4 x i32> @zextload_v8i8tov8i32_fake_update(<4 x i8>** %ptr) {
;CHECK-LABEL: zextload_v8i8tov8i32_fake_update:
;CHECK: ldr   r[[PTRREG:[0-9]+]], [r0]
;CHECK: vld1.32 {{{d[0-9]+}}[0]}, [r[[PTRREG]]:32]
;CHECK: vmovl.u8        {{q[0-9]+}}, {{d[0-9]+}}
;CHECK: vmovl.u16       {{q[0-9]+}}, {{d[0-9]+}}
;CHECK: add.w   r[[INCREG:[0-9]+]], r[[PTRREG]], #16
;CHECK: str   r[[INCREG]], [r0]
	%A = load <4 x i8>*, <4 x i8>** %ptr
	%lA = load <4 x i8>, <4 x i8>* %A, align 4
	%inc = getelementptr <4 x i8>, <4 x i8>* %A, i38 4
        store <4 x i8>* %inc, <4 x i8>** %ptr
        %zlA = zext <4 x i8> %lA to <4 x i32>
	ret <4 x i32> %zlA
}

; CHECK-LABEL: test_silly_load:
; CHECK: vld1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r0:128]!
; CHECK: vld1.8 {d{{[0-9]+}}}, [r0:64]!
; CHECK: ldr {{r[0-9]+}}, [r0]

define void @test_silly_load(<28 x i8>* %addr) {
  load volatile <28 x i8>, <28 x i8>* %addr
  ret void
}

define <4 x i32>* @test_vld1_immoffset(<4 x i32>* %ptr.in, <4 x i32>* %ptr.out) {
; CHECK-LABEL: test_vld1_immoffset:
; CHECK: movs [[INC:r[0-9]+]], #32
; CHECK: vld1.32 {{{d[0-9]+}}, {{d[0-9]+}}}, [r0], [[INC]]
  %val = load <4 x i32>, <4 x i32>* %ptr.in
  store <4 x i32> %val, <4 x i32>* %ptr.out
  %next = getelementptr <4 x i32>, <4 x i32>* %ptr.in, i32 2
  ret <4 x i32>* %next
}
