; RUN: llc < %s -mtriple aarch64-apple-darwin -asm-verbose=false -disable-post-ra | FileCheck %s

define void @test_stnp_v4i64(<4 x i64>* %p, <4 x i64> %v) #0 {
; CHECK-LABEL: test_stnp_v4i64:
; CHECK-NEXT:  stnp q0, q1, [x0]
; CHECK-NEXT:  ret
  store <4 x i64> %v, <4 x i64>* %p, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v4i32(<4 x i32>* %p, <4 x i32> %v) #0 {
; CHECK-LABEL: test_stnp_v4i32:
; CHECK-NEXT:  mov d[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp d0, d[[HI]], [x0]
; CHECK-NEXT:  ret
  store <4 x i32> %v, <4 x i32>* %p, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v8i16(<8 x i16>* %p, <8 x i16> %v) #0 {
; CHECK-LABEL: test_stnp_v8i16:
; CHECK-NEXT:  mov d[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp d0, d[[HI]], [x0]
; CHECK-NEXT:  ret
  store <8 x i16> %v, <8 x i16>* %p, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v16i8(<16 x i8>* %p, <16 x i8> %v) #0 {
; CHECK-LABEL: test_stnp_v16i8:
; CHECK-NEXT:  mov d[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp d0, d[[HI]], [x0]
; CHECK-NEXT:  ret
  store <16 x i8> %v, <16 x i8>* %p, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v2i32(<2 x i32>* %p, <2 x i32> %v) #0 {
; CHECK-LABEL: test_stnp_v2i32:
; CHECK-NEXT:  mov s[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp s0, s[[HI]], [x0]
; CHECK-NEXT:  ret
  store <2 x i32> %v, <2 x i32>* %p, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v4i16(<4 x i16>* %p, <4 x i16> %v) #0 {
; CHECK-LABEL: test_stnp_v4i16:
; CHECK-NEXT:  mov s[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp s0, s[[HI]], [x0]
; CHECK-NEXT:  ret
  store <4 x i16> %v, <4 x i16>* %p, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v8i8(<8 x i8>* %p, <8 x i8> %v) #0 {
; CHECK-LABEL: test_stnp_v8i8:
; CHECK-NEXT:  mov s[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp s0, s[[HI]], [x0]
; CHECK-NEXT:  ret
  store <8 x i8> %v, <8 x i8>* %p, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v2f64(<2 x double>* %p, <2 x double> %v) #0 {
; CHECK-LABEL: test_stnp_v2f64:
; CHECK-NEXT:  mov d[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp d0, d[[HI]], [x0]
; CHECK-NEXT:  ret
  store <2 x double> %v, <2 x double>* %p, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v4f32(<4 x float>* %p, <4 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v4f32:
; CHECK-NEXT:  mov d[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp d0, d[[HI]], [x0]
; CHECK-NEXT:  ret
  store <4 x float> %v, <4 x float>* %p, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v2f32(<2 x float>* %p, <2 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v2f32:
; CHECK-NEXT:  mov s[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp s0, s[[HI]], [x0]
; CHECK-NEXT:  ret
  store <2 x float> %v, <2 x float>* %p, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v1f64(<1 x double>* %p, <1 x double> %v) #0 {
; CHECK-LABEL: test_stnp_v1f64:
; CHECK-NEXT:  mov s[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp s0, s[[HI]], [x0]
; CHECK-NEXT:  ret
  store <1 x double> %v, <1 x double>* %p, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v1i64(<1 x i64>* %p, <1 x i64> %v) #0 {
; CHECK-LABEL: test_stnp_v1i64:
; CHECK-NEXT:  mov s[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp s0, s[[HI]], [x0]
; CHECK-NEXT:  ret
  store <1 x i64> %v, <1 x i64>* %p, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_i64(i64* %p, i64 %v) #0 {
; CHECK-LABEL: test_stnp_i64:
; CHECK-NEXT:  lsr x[[HI:[0-9]+]], x1, #32
; CHECK-NEXT:  stnp w1, w[[HI]], [x0]
; CHECK-NEXT:  ret
  store i64 %v, i64* %p, align 1, !nontemporal !0
  ret void
}


define void @test_stnp_v2f64_offset(<2 x double>* %p, <2 x double> %v) #0 {
; CHECK-LABEL: test_stnp_v2f64_offset:
; CHECK-NEXT:  mov d[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp d0, d[[HI]], [x0, #16]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr <2 x double>, <2 x double>* %p, i32 1
  store <2 x double> %v, <2 x double>* %tmp0, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v2f64_offset_neg(<2 x double>* %p, <2 x double> %v) #0 {
; CHECK-LABEL: test_stnp_v2f64_offset_neg:
; CHECK-NEXT:  mov d[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp d0, d[[HI]], [x0, #-16]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr <2 x double>, <2 x double>* %p, i32 -1
  store <2 x double> %v, <2 x double>* %tmp0, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v2f32_offset(<2 x float>* %p, <2 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v2f32_offset:
; CHECK-NEXT:  mov s[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp s0, s[[HI]], [x0, #8]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr <2 x float>, <2 x float>* %p, i32 1
  store <2 x float> %v, <2 x float>* %tmp0, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v2f32_offset_neg(<2 x float>* %p, <2 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v2f32_offset_neg:
; CHECK-NEXT:  mov s[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp s0, s[[HI]], [x0, #-8]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr <2 x float>, <2 x float>* %p, i32 -1
  store <2 x float> %v, <2 x float>* %tmp0, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_i64_offset(i64* %p, i64 %v) #0 {
; CHECK-LABEL: test_stnp_i64_offset:
; CHECK-NEXT:  lsr x[[HI:[0-9]+]], x1, #32
; CHECK-NEXT:  stnp w1, w[[HI]], [x0, #8]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr i64, i64* %p, i32 1
  store i64 %v, i64* %tmp0, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_i64_offset_neg(i64* %p, i64 %v) #0 {
; CHECK-LABEL: test_stnp_i64_offset_neg:
; CHECK-NEXT:  lsr x[[HI:[0-9]+]], x1, #32
; CHECK-NEXT:  stnp w1, w[[HI]], [x0, #-8]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr i64, i64* %p, i32 -1
  store i64 %v, i64* %tmp0, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v4f32_invalid_offset_4(i8* %p, <4 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v4f32_invalid_offset_4:
; CHECK-NEXT:  add x[[PTR:[0-9]+]], x0, #4
; CHECK-NEXT:  mov d[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp d0, d[[HI]], [x[[PTR]]]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr i8, i8* %p, i32 4
  %tmp1 = bitcast i8* %tmp0 to <4 x float>*
  store <4 x float> %v, <4 x float>* %tmp1, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v4f32_invalid_offset_neg_4(i8* %p, <4 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v4f32_invalid_offset_neg_4:
; CHECK-NEXT:  sub x[[PTR:[0-9]+]], x0, #4
; CHECK-NEXT:  mov d[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp d0, d[[HI]], [x[[PTR]]]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr i8, i8* %p, i32 -4
  %tmp1 = bitcast i8* %tmp0 to <4 x float>*
  store <4 x float> %v, <4 x float>* %tmp1, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v4f32_invalid_offset_512(i8* %p, <4 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v4f32_invalid_offset_512:
; CHECK-NEXT:  add x[[PTR:[0-9]+]], x0, #512
; CHECK-NEXT:  mov d[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp d0, d[[HI]], [x[[PTR]]]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr i8, i8* %p, i32 512
  %tmp1 = bitcast i8* %tmp0 to <4 x float>*
  store <4 x float> %v, <4 x float>* %tmp1, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v4f32_offset_504(i8* %p, <4 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v4f32_offset_504:
; CHECK-NEXT:  mov d[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp d0, d[[HI]], [x0, #504]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr i8, i8* %p, i32 504
  %tmp1 = bitcast i8* %tmp0 to <4 x float>*
  store <4 x float> %v, <4 x float>* %tmp1, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v4f32_invalid_offset_508(i8* %p, <4 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v4f32_invalid_offset_508:
; CHECK-NEXT:  add x[[PTR:[0-9]+]], x0, #508
; CHECK-NEXT:  mov d[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp d0, d[[HI]], [x[[PTR]]]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr i8, i8* %p, i32 508
  %tmp1 = bitcast i8* %tmp0 to <4 x float>*
  store <4 x float> %v, <4 x float>* %tmp1, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v4f32_invalid_offset_neg_520(i8* %p, <4 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v4f32_invalid_offset_neg_520:
; CHECK-NEXT:  sub x[[PTR:[0-9]+]], x0, #520
; CHECK-NEXT:  mov d[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp d0, d[[HI]], [x[[PTR]]]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr i8, i8* %p, i32 -520
  %tmp1 = bitcast i8* %tmp0 to <4 x float>*
  store <4 x float> %v, <4 x float>* %tmp1, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v4f32_offset_neg_512(i8* %p, <4 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v4f32_offset_neg_512:
; CHECK-NEXT:  mov d[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp d0, d[[HI]], [x0, #-512]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr i8, i8* %p, i32 -512
  %tmp1 = bitcast i8* %tmp0 to <4 x float>*
  store <4 x float> %v, <4 x float>* %tmp1, align 1, !nontemporal !0
  ret void
}


define void @test_stnp_v2f32_invalid_offset_256(i8* %p, <2 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v2f32_invalid_offset_256:
; CHECK-NEXT:  add x[[PTR:[0-9]+]], x0, #256
; CHECK-NEXT:  mov s[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp s0, s[[HI]], [x[[PTR]]]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr i8, i8* %p, i32 256
  %tmp1 = bitcast i8* %tmp0 to <2 x float>*
  store <2 x float> %v, <2 x float>* %tmp1, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v2f32_offset_252(i8* %p, <2 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v2f32_offset_252:
; CHECK-NEXT:  mov s[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp s0, s[[HI]], [x0, #252]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr i8, i8* %p, i32 252
  %tmp1 = bitcast i8* %tmp0 to <2 x float>*
  store <2 x float> %v, <2 x float>* %tmp1, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v2f32_invalid_offset_neg_260(i8* %p, <2 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v2f32_invalid_offset_neg_260:
; CHECK-NEXT:  sub x[[PTR:[0-9]+]], x0, #260
; CHECK-NEXT:  mov s[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp s0, s[[HI]], [x[[PTR]]]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr i8, i8* %p, i32 -260
  %tmp1 = bitcast i8* %tmp0 to <2 x float>*
  store <2 x float> %v, <2 x float>* %tmp1, align 1, !nontemporal !0
  ret void
}

define void @test_stnp_v2f32_offset_neg_256(i8* %p, <2 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v2f32_offset_neg_256:
; CHECK-NEXT:  mov s[[HI:[0-9]+]], v0[1]
; CHECK-NEXT:  stnp s0, s[[HI]], [x0, #-256]
; CHECK-NEXT:  ret
  %tmp0 = getelementptr i8, i8* %p, i32 -256
  %tmp1 = bitcast i8* %tmp0 to <2 x float>*
  store <2 x float> %v, <2 x float>* %tmp1, align 1, !nontemporal !0
  ret void
}

declare void @dummy(<4 x float>*)

define void @test_stnp_v4f32_offset_alloca(<4 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v4f32_offset_alloca:
; CHECK:       stnp d0, d{{.*}}, [sp]
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  bl _dummy
  %tmp0 = alloca <4 x float>
  store <4 x float> %v, <4 x float>* %tmp0, align 1, !nontemporal !0
  call void @dummy(<4 x float>* %tmp0)
  ret void
}

define void @test_stnp_v4f32_offset_alloca_2(<4 x float> %v) #0 {
; CHECK-LABEL: test_stnp_v4f32_offset_alloca_2:
; CHECK:       stnp d0, d{{.*}}, [sp, #16]
; CHECK-NEXT:  mov x0, sp
; CHECK-NEXT:  bl _dummy
  %tmp0 = alloca <4 x float>, i32 2
  %tmp1 = getelementptr <4 x float>, <4 x float>* %tmp0, i32 1
  store <4 x float> %v, <4 x float>* %tmp1, align 1, !nontemporal !0
  call void @dummy(<4 x float>* %tmp0)
  ret void
}

define void @test_stnp_v32i8(<32 x i8> %v, <32 x i8>* %ptr) {
; CHECK-LABEL: _test_stnp_v32i8:
; CHECK-NEXT:    .cfi_startproc
; CHECK-NEXT:    stnp    q0, q1, [x0]
; CHECK-NEXT:    ret

entry:
  store <32 x i8> %v, <32 x i8>* %ptr, align 4, !nontemporal !0
  ret void
}

define void @test_stnp_v32i16(<32 x i16> %v, <32 x i16>* %ptr) {
; CHECK-LABEL: _test_stnp_v32i16:
; CHECK-NEXT:    .cfi_startproc
; CHECK-NEXT:    stnp    q2, q3, [x0, #32]
; CHECK-NEXT:    stnp    q0, q1, [x0]
; CHECK-NEXT:    ret

entry:
  store <32 x i16> %v, <32 x i16>* %ptr, align 4, !nontemporal !0
  ret void
}

define void @test_stnp_v32f16(<32 x half> %v, <32 x half>* %ptr) {
; CHECK-LABEL: _test_stnp_v32f16:
; CHECK-NEXT:    .cfi_startproc
; CHECK-NEXT:    stnp    q2, q3, [x0, #32]
; CHECK-NEXT:    stnp    q0, q1, [x0]
; CHECK-NEXT:    ret

entry:
  store <32 x half> %v, <32 x half>* %ptr, align 4, !nontemporal !0
  ret void
}

define void @test_stnp_v16i32(<16 x i32> %v, <16 x i32>* %ptr) {
; CHECK-LABEL: _test_stnp_v16i32:
; CHECK-NEXT:    .cfi_startproc
; CHECK-NEXT:    stnp    q2, q3, [x0, #32]
; CHECK-NEXT:    stnp    q0, q1, [x0]
; CHECK-NEXT:    ret

entry:
  store <16 x i32> %v, <16 x i32>* %ptr, align 4, !nontemporal !0
  ret void
}

define void @test_stnp_v16f32(<16 x float> %v, <16 x float>* %ptr) {
; CHECK-LABEL: _test_stnp_v16f32:
; CHECK-NEXT:    .cfi_startproc
; CHECK-NEXT:    stnp    q2, q3, [x0, #32]
; CHECK-NEXT:    stnp    q0, q1, [x0]
; CHECK-NEXT:    ret

entry:
  store <16 x float> %v, <16 x float>* %ptr, align 4, !nontemporal !0
  ret void
}

define void @test_stnp_v17f32(<17 x float> %v, <17 x float>* %ptr) {
; CHECK-LABEL: _test_stnp_v17f32:
; CHECK-NEXT:	.cfi_startproc
; CHECK-NEXT:	ldr	s16, [sp, #16]
; CHECK-NEXT:	mov.s	v0[1], v1[0]
; CHECK-NEXT:	mov.s	v4[1], v5[0]
; CHECK-NEXT:	ldr	s1, [sp]
; CHECK-NEXT:	add	x8, sp, #20
; CHECK-NEXT:	ld1.s	{ v16 }[1], [x8]
; CHECK-NEXT:	add	x8, sp, #4
; CHECK-NEXT:	ld1.s	{ v1 }[1], [x8]
; CHECK-NEXT:	add	x8, sp, #24
; CHECK-NEXT:	ld1.s	{ v16 }[2], [x8]
; CHECK-NEXT:	add	x8, sp, #8
; CHECK-NEXT:	ld1.s	{ v1 }[2], [x8]
; CHECK-NEXT:	add	x8, sp, #28
; CHECK-NEXT:	ld1.s	{ v16 }[3], [x8]
; CHECK-NEXT:	add	x8, sp, #12
; CHECK-NEXT:	mov.s	v0[2], v2[0]
; CHECK-NEXT:	ldr	s2, [sp, #32]
; CHECK-NEXT:	mov.s	v4[2], v6[0]
; CHECK-NEXT:	mov.s	v0[3], v3[0]
; CHECK-NEXT:	mov.s	v4[3], v7[0]
; CHECK-NEXT:	mov	d3, v4[1]
; CHECK-NEXT:	mov	d5, v0[1]
; CHECK-NEXT:	ld1.s	{ v1 }[3], [x8]
; CHECK-NEXT:	stnp	d4, d3, [x0, #16]
; CHECK-NEXT:	stnp	d0, d5, [x0]
; CHECK-NEXT:	mov	d0, v16[1]
; CHECK-NEXT:	mov	d3, v1[1]
; CHECK-NEXT:	stnp	d16, d0, [x0, #48]
; CHECK-NEXT:	stnp	d1, d3, [x0, #32]
; CHECK-NEXT:	str	s2, [x0, #64]
; CHECK-NEXT:	ret

entry:
  store <17 x float> %v, <17 x float>* %ptr, align 4, !nontemporal !0
  ret void
}
define void @test_stnp_v16i32_invalid_offset(<16 x i32> %v, <16 x i32>* %ptr) {
; CHECK-LABEL: _test_stnp_v16i32_invalid_offset:
; CHECK-NEXT:    .cfi_startproc
; CHECK-NEXT:    mov w8, #32000
; CHECK-NEXT:    mov w9, #32032
; CHECK-NEXT:    add x8, x0, x8
; CHECK-NEXT:    add x9, x0, x9
; CHECK-NEXT:    stnp    q2, q3, [x9]
; CHECK-NEXT:    stnp    q0, q1, [x8]
; CHECK-NEXT:    ret

entry:
  %gep = getelementptr <16 x i32>, <16 x i32>* %ptr, i32 500
  store <16 x i32> %v, <16 x i32>* %gep, align 4, !nontemporal !0
  ret void
}

define void @test_stnp_v16f64(<16 x double> %v, <16 x double>* %ptr) {
; CHECK-LABEL: _test_stnp_v16f64:
; CHECK-NEXT:    .cfi_startproc
; CHECK-NEXT:    stnp    q6, q7, [x0, #96]
; CHECK-NEXT:    stnp    q4, q5, [x0, #64]
; CHECK-NEXT:    stnp    q2, q3, [x0, #32]
; CHECK-NEXT:    stnp    q0, q1, [x0]
; CHECK-NEXT:    ret

entry:
  store <16 x double> %v, <16 x double>* %ptr, align 4, !nontemporal !0
  ret void
}

define void @test_stnp_v16i64(<16 x i64> %v, <16 x i64>* %ptr) {
; CHECK-LABEL: _test_stnp_v16i64:
; CHECK-NEXT:    .cfi_startproc
; CHECK-NEXT:    stnp    q6, q7, [x0, #96]
; CHECK-NEXT:    stnp    q4, q5, [x0, #64]
; CHECK-NEXT:    stnp    q2, q3, [x0, #32]
; CHECK-NEXT:    stnp    q0, q1, [x0]
; CHECK-NEXT:    ret

entry:
  store <16 x i64> %v, <16 x i64>* %ptr, align 4, !nontemporal !0
  ret void
}

!0 = !{ i32 1 }

attributes #0 = { nounwind }
