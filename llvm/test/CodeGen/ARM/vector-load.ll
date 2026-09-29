; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:o-p:32:32-i1:8:32-i8:8:32-i16:16:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7s-apple-ios8.0.0"

define <8 x i8> @load_v8i8(ptr %ptr) {
;CHECK-LABEL: load_v8i8:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load ptr, ptr %ptr
	%lA = load <8 x i8>, ptr %A, align 1
	ret <8 x i8> %lA
}

define <8 x i8> @load_v8i8_update(ptr %ptr) {
;CHECK-LABEL: load_v8i8_update:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%lA = load <8 x i8>, ptr %A, align 1
	%inc = getelementptr <8 x i8>, ptr %A, i38 1
        store ptr %inc, ptr %ptr
	ret <8 x i8> %lA
}

define <4 x i16> @load_v4i16(ptr %ptr) {
;CHECK-LABEL: load_v4i16:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load ptr, ptr %ptr
	%lA = load <4 x i16>, ptr %A, align 1
	ret <4 x i16> %lA
}

define <4 x i16> @load_v4i16_update(ptr %ptr) {
;CHECK-LABEL: load_v4i16_update:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%lA = load <4 x i16>, ptr %A, align 1
	%inc = getelementptr <4 x i16>, ptr %A, i34 1
        store ptr %inc, ptr %ptr
	ret <4 x i16> %lA
}

define <2 x i32> @load_v2i32(ptr %ptr) {
;CHECK-LABEL: load_v2i32:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load ptr, ptr %ptr
	%lA = load <2 x i32>, ptr %A, align 1
	ret <2 x i32> %lA
}

define <2 x i32> @load_v2i32_update(ptr %ptr) {
;CHECK-LABEL: load_v2i32_update:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%lA = load <2 x i32>, ptr %A, align 1
	%inc = getelementptr <2 x i32>, ptr %A, i32 1
        store ptr %inc, ptr %ptr
	ret <2 x i32> %lA
}

define <2 x float> @load_v2f32(ptr %ptr) {
;CHECK-LABEL: load_v2f32:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load ptr, ptr %ptr
	%lA = load <2 x float>, ptr %A, align 1
	ret <2 x float> %lA
}

define <2 x float> @load_v2f32_update(ptr %ptr) {
;CHECK-LABEL: load_v2f32_update:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%lA = load <2 x float>, ptr %A, align 1
	%inc = getelementptr <2 x float>, ptr %A, i32 1
        store ptr %inc, ptr %ptr
	ret <2 x float> %lA
}

define <1 x i64> @load_v1i64(ptr %ptr) {
;CHECK-LABEL: load_v1i64:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load ptr, ptr %ptr
	%lA = load <1 x i64>, ptr %A, align 1
	ret <1 x i64> %lA
}

define <1 x i64> @load_v1i64_update(ptr %ptr) {
;CHECK-LABEL: load_v1i64_update:
;CHECK: vld1.8 {{{d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%lA = load <1 x i64>, ptr %A, align 1
	%inc = getelementptr <1 x i64>, ptr %A, i31 1
        store ptr %inc, ptr %ptr
	ret <1 x i64> %lA
}

define <16 x i8> @load_v16i8(ptr %ptr) {
;CHECK-LABEL: load_v16i8:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load ptr, ptr %ptr
	%lA = load <16 x i8>, ptr %A, align 1
	ret <16 x i8> %lA
}

define <16 x i8> @load_v16i8_update(ptr %ptr) {
;CHECK-LABEL: load_v16i8_update:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%lA = load <16 x i8>, ptr %A, align 1
	%inc = getelementptr <16 x i8>, ptr %A, i316 1
        store ptr %inc, ptr %ptr
	ret <16 x i8> %lA
}

define <8 x i16> @load_v8i16(ptr %ptr) {
;CHECK-LABEL: load_v8i16:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load ptr, ptr %ptr
	%lA = load <8 x i16>, ptr %A, align 1
	ret <8 x i16> %lA
}

define <8 x i16> @load_v8i16_update(ptr %ptr) {
;CHECK-LABEL: load_v8i16_update:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%lA = load <8 x i16>, ptr %A, align 1
	%inc = getelementptr <8 x i16>, ptr %A, i38 1
        store ptr %inc, ptr %ptr
	ret <8 x i16> %lA
}

define <4 x i32> @load_v4i32(ptr %ptr) {
;CHECK-LABEL: load_v4i32:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load ptr, ptr %ptr
	%lA = load <4 x i32>, ptr %A, align 1
	ret <4 x i32> %lA
}

define <4 x i32> @load_v4i32_update(ptr %ptr) {
;CHECK-LABEL: load_v4i32_update:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%lA = load <4 x i32>, ptr %A, align 1
	%inc = getelementptr <4 x i32>, ptr %A, i34 1
        store ptr %inc, ptr %ptr
	ret <4 x i32> %lA
}

define <4 x float> @load_v4f32(ptr %ptr) {
;CHECK-LABEL: load_v4f32:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load ptr, ptr %ptr
	%lA = load <4 x float>, ptr %A, align 1
	ret <4 x float> %lA
}

define <4 x float> @load_v4f32_update(ptr %ptr) {
;CHECK-LABEL: load_v4f32_update:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%lA = load <4 x float>, ptr %A, align 1
	%inc = getelementptr <4 x float>, ptr %A, i34 1
        store ptr %inc, ptr %ptr
	ret <4 x float> %lA
}

define <2 x i64> @load_v2i64(ptr %ptr) {
;CHECK-LABEL: load_v2i64:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]
	%A = load ptr, ptr %ptr
	%lA = load <2 x i64>, ptr %A, align 1
	ret <2 x i64> %lA
}

define <2 x i64> @load_v2i64_update(ptr %ptr) {
;CHECK-LABEL: load_v2i64_update:
;CHECK: vld1.8 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%lA = load <2 x i64>, ptr %A, align 1
	%inc = getelementptr <2 x i64>, ptr %A, i32 1
        store ptr %inc, ptr %ptr
	ret <2 x i64> %lA
}

; Make sure we change the type to match alignment if necessary.
define <2 x i64> @load_v2i64_update_aligned2(ptr %ptr) {
;CHECK-LABEL: load_v2i64_update_aligned2:
;CHECK: vld1.16 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%lA = load <2 x i64>, ptr %A, align 2
	%inc = getelementptr <2 x i64>, ptr %A, i32 1
        store ptr %inc, ptr %ptr
	ret <2 x i64> %lA
}

define <2 x i64> @load_v2i64_update_aligned4(ptr %ptr) {
;CHECK-LABEL: load_v2i64_update_aligned4:
;CHECK: vld1.32 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%lA = load <2 x i64>, ptr %A, align 4
	%inc = getelementptr <2 x i64>, ptr %A, i32 1
        store ptr %inc, ptr %ptr
	ret <2 x i64> %lA
}

define <2 x i64> @load_v2i64_update_aligned8(ptr %ptr) {
;CHECK-LABEL: load_v2i64_update_aligned8:
;CHECK: vld1.64 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}]!
	%A = load ptr, ptr %ptr
	%lA = load <2 x i64>, ptr %A, align 8
	%inc = getelementptr <2 x i64>, ptr %A, i32 1
        store ptr %inc, ptr %ptr
	ret <2 x i64> %lA
}

define <2 x i64> @load_v2i64_update_aligned16(ptr %ptr) {
;CHECK-LABEL: load_v2i64_update_aligned16:
;CHECK: vld1.64 {{{d[0-9]+, d[0-9]+}}}, [{{r[0-9]+}}:128]!
	%A = load ptr, ptr %ptr
	%lA = load <2 x i64>, ptr %A, align 16
	%inc = getelementptr <2 x i64>, ptr %A, i32 1
        store ptr %inc, ptr %ptr
	ret <2 x i64> %lA
}

; Make sure we don't break smaller-than-dreg extloads.
define <4 x i32> @zextload_v8i8tov8i32(ptr %ptr) {
;CHECK-LABEL: zextload_v8i8tov8i32:
;CHECK: vld1.32 {{{d[0-9]+}}[0]}, [{{r[0-9]+}}:32]
;CHECK: vmovl.u8        {{q[0-9]+}}, {{d[0-9]+}}
;CHECK: vmovl.u16       {{q[0-9]+}}, {{d[0-9]+}}
	%A = load ptr, ptr %ptr
	%lA = load <4 x i8>, ptr %A, align 4
        %zlA = zext <4 x i8> %lA to <4 x i32>
	ret <4 x i32> %zlA
}

define <4 x i32> @zextload_v8i8tov8i32_fake_update(ptr %ptr) {
;CHECK-LABEL: zextload_v8i8tov8i32_fake_update:
;CHECK: ldr   r[[PTRREG:[0-9]+]], [r0]
;CHECK: vld1.32 {{{d[0-9]+}}[0]}, [r[[PTRREG]]:32]
;CHECK: vmovl.u8        {{q[0-9]+}}, {{d[0-9]+}}
;CHECK: vmovl.u16       {{q[0-9]+}}, {{d[0-9]+}}
;CHECK: add.w   r[[INCREG:[0-9]+]], r[[PTRREG]], #16
;CHECK: str   r[[INCREG]], [r0]
	%A = load ptr, ptr %ptr
	%lA = load <4 x i8>, ptr %A, align 4
	%inc = getelementptr <4 x i8>, ptr %A, i38 4
        store ptr %inc, ptr %ptr
        %zlA = zext <4 x i8> %lA to <4 x i32>
	ret <4 x i32> %zlA
}

; CHECK-LABEL: test_silly_load:
; CHECK: vld1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r0:128]!
; CHECK: vld1.8 {d{{[0-9]+}}}, [r0:64]!
; CHECK: ldr {{r[0-9]+}}, [r0]

define void @test_silly_load(ptr %addr) {
  load volatile <28 x i8>, ptr %addr
  ret void
}

define ptr @test_vld1_immoffset(ptr %ptr.in, ptr %ptr.out) {
; CHECK-LABEL: test_vld1_immoffset:
; CHECK: movs [[INC:r[0-9]+]], #32
; CHECK: vld1.32 {{{d[0-9]+}}, {{d[0-9]+}}}, [r0], [[INC]]
  %val = load <4 x i32>, ptr %ptr.in
  store <4 x i32> %val, ptr %ptr.out
  %next = getelementptr <4 x i32>, ptr %ptr.in, i32 2
  ret ptr %next
}
