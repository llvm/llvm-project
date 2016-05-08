; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s -check-prefix=CHECK -check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx | FileCheck %s -check-prefix=CHECK -check-prefix=AVX
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s -check-prefix=CHECK -check-prefix=AVX2

; Make sure that we generate non-temporal stores for the test cases below.
; We use xorps for zeroing, so domain information isn't available anymore.

define void @test_zero_v4f32(<4 x float>* %dst) {
; CHECK-LABEL: test_zero_v4f32:
; SSE: movntps
; AVX: vmovntps
; AVX2: vmovntps
  store <4 x float> zeroinitializer, <4 x float>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_zero_v4i32(<4 x i32>* %dst) {
; CHECK-LABEL: test_zero_v4i32:
; SSE: movntps
; AVX: vmovntps
; AVX2: vmovntps
  store <4 x i32> zeroinitializer, <4 x i32>* %dst, align 16, !nontemporal !1
  store <4 x i32> zeroinitializer, <4 x i32>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_zero_v2f64(<2 x double>* %dst) {
; CHECK-LABEL: test_zero_v2f64:
; SSE: movntps
; AVX: vmovntps
; AVX2: vmovntps
  store <2 x double> zeroinitializer, <2 x double>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_zero_v2i64(<2 x i64>* %dst) {
; CHECK-LABEL: test_zero_v2i64:
; SSE: movntps
; AVX: vmovntps
; AVX2: vmovntps
  store <2 x i64> zeroinitializer, <2 x i64>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_zero_v8i16(<8 x i16>* %dst) {
; CHECK-LABEL: test_zero_v8i16:
; SSE: movntps
; AVX: vmovntps
; AVX2: vmovntps
  store <8 x i16> zeroinitializer, <8 x i16>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_zero_v16i8(<16 x i8>* %dst) {
; CHECK-LABEL: test_zero_v16i8:
; SSE: movntps
; AVX: vmovntps
; AVX2: vmovntps
  store <16 x i8> zeroinitializer, <16 x i8>* %dst, align 16, !nontemporal !1
  ret void
}

; And now YMM versions.

define void @test_zero_v8f32(<8 x float>* %dst) {
; CHECK-LABEL: test_zero_v8f32:
; AVX: vmovntps %ymm
; AVX2: vmovntps %ymm
  store <8 x float> zeroinitializer, <8 x float>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_zero_v8i32(<8 x i32>* %dst) {
; CHECK-LABEL: test_zero_v8i32:
; AVX: vmovntps %ymm
; AVX2: vmovntps %ymm
  store <8 x i32> zeroinitializer, <8 x i32>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_zero_v4f64(<4 x double>* %dst) {
; CHECK-LABEL: test_zero_v4f64:
; AVX: vmovntps %ymm
; AVX2: vmovntps %ymm
  store <4 x double> zeroinitializer, <4 x double>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_zero_v4i64(<4 x i64>* %dst) {
; CHECK-LABEL: test_zero_v4i64:
; AVX: vmovntps %ymm
; AVX2: vmovntps %ymm
  store <4 x i64> zeroinitializer, <4 x i64>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_zero_v16i16(<16 x i16>* %dst) {
; CHECK-LABEL: test_zero_v16i16:
; AVX: vmovntps %ymm
; AVX2: vmovntps %ymm
  store <16 x i16> zeroinitializer, <16 x i16>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_zero_v32i8(<32 x i8>* %dst) {
; CHECK-LABEL: test_zero_v32i8:
; AVX: vmovntps %ymm
; AVX2: vmovntps %ymm
  store <32 x i8> zeroinitializer, <32 x i8>* %dst, align 32, !nontemporal !1
  ret void
}


; Check that we also handle arguments.  Here the type survives longer.

define void @test_arg_v4f32(<4 x float> %arg, <4 x float>* %dst) {
; CHECK-LABEL: test_arg_v4f32:
; SSE: movntps
; AVX: vmovntps
; AVX2: vmovntps
  store <4 x float> %arg, <4 x float>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_arg_v4i32(<4 x i32> %arg, <4 x i32>* %dst) {
; CHECK-LABEL: test_arg_v4i32:
; SSE: movntps
; AVX: vmovntps
; AVX2: vmovntps
  store <4 x i32> %arg, <4 x i32>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_arg_v2f64(<2 x double> %arg, <2 x double>* %dst) {
; CHECK-LABEL: test_arg_v2f64:
; SSE: movntps
; AVX: vmovntps
; AVX2: vmovntps
  store <2 x double> %arg, <2 x double>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_arg_v2i64(<2 x i64> %arg, <2 x i64>* %dst) {
; CHECK-LABEL: test_arg_v2i64:
; SSE: movntps
; AVX: vmovntps
; AVX2: vmovntps
  store <2 x i64> %arg, <2 x i64>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_arg_v8i16(<8 x i16> %arg, <8 x i16>* %dst) {
; CHECK-LABEL: test_arg_v8i16:
; SSE: movntps
; AVX: vmovntps
; AVX2: vmovntps
  store <8 x i16> %arg, <8 x i16>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_arg_v16i8(<16 x i8> %arg, <16 x i8>* %dst) {
; CHECK-LABEL: test_arg_v16i8:
; SSE: movntps
; AVX: vmovntps
; AVX2: vmovntps
  store <16 x i8> %arg, <16 x i8>* %dst, align 16, !nontemporal !1
  ret void
}

; And now YMM versions.

define void @test_arg_v8f32(<8 x float> %arg, <8 x float>* %dst) {
; CHECK-LABEL: test_arg_v8f32:
; AVX: vmovntps %ymm
; AVX2: vmovntps %ymm
  store <8 x float> %arg, <8 x float>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_arg_v8i32(<8 x i32> %arg, <8 x i32>* %dst) {
; CHECK-LABEL: test_arg_v8i32:
; AVX: vmovntps %ymm
; AVX2: vmovntps %ymm
  store <8 x i32> %arg, <8 x i32>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_arg_v4f64(<4 x double> %arg, <4 x double>* %dst) {
; CHECK-LABEL: test_arg_v4f64:
; AVX: vmovntps %ymm
; AVX2: vmovntps %ymm
  store <4 x double> %arg, <4 x double>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_arg_v4i64(<4 x i64> %arg, <4 x i64>* %dst) {
; CHECK-LABEL: test_arg_v4i64:
; AVX: vmovntps %ymm
; AVX2: vmovntps %ymm
  store <4 x i64> %arg, <4 x i64>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_arg_v16i16(<16 x i16> %arg, <16 x i16>* %dst) {
; CHECK-LABEL: test_arg_v16i16:
; AVX: vmovntps %ymm
; AVX2: vmovntps %ymm
  store <16 x i16> %arg, <16 x i16>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_arg_v32i8(<32 x i8> %arg, <32 x i8>* %dst) {
; CHECK-LABEL: test_arg_v32i8:
; AVX: vmovntps %ymm
; AVX2: vmovntps %ymm
  store <32 x i8> %arg, <32 x i8>* %dst, align 32, !nontemporal !1
  ret void
}


; Now check that if the execution domain is trivially visible, we use it.
; We use an add to make the type survive all the way to the MOVNT.

define void @test_op_v4f32(<4 x float> %a, <4 x float> %b, <4 x float>* %dst) {
; CHECK-LABEL: test_op_v4f32:
; SSE: movntps
; AVX: vmovntps
; AVX2: vmovntps
  %r = fadd <4 x float> %a, %b
  store <4 x float> %r, <4 x float>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_op_v4i32(<4 x i32> %a, <4 x i32> %b, <4 x i32>* %dst) {
; CHECK-LABEL: test_op_v4i32:
; SSE: movntdq
; AVX: vmovntdq
; AVX2: vmovntdq
  %r = add <4 x i32> %a, %b
  store <4 x i32> %r, <4 x i32>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_op_v2f64(<2 x double> %a, <2 x double> %b, <2 x double>* %dst) {
; CHECK-LABEL: test_op_v2f64:
; SSE: movntpd
; AVX: vmovntpd
; AVX2: vmovntpd
  %r = fadd <2 x double> %a, %b
  store <2 x double> %r, <2 x double>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_op_v2i64(<2 x i64> %a, <2 x i64> %b, <2 x i64>* %dst) {
; CHECK-LABEL: test_op_v2i64:
; SSE: movntdq
; AVX: vmovntdq
; AVX2: vmovntdq
  %r = add <2 x i64> %a, %b
  store <2 x i64> %r, <2 x i64>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_op_v8i16(<8 x i16> %a, <8 x i16> %b, <8 x i16>* %dst) {
; CHECK-LABEL: test_op_v8i16:
; SSE: movntdq
; AVX: vmovntdq
; AVX2: vmovntdq
  %r = add <8 x i16> %a, %b
  store <8 x i16> %r, <8 x i16>* %dst, align 16, !nontemporal !1
  ret void
}

define void @test_op_v16i8(<16 x i8> %a, <16 x i8> %b, <16 x i8>* %dst) {
; CHECK-LABEL: test_op_v16i8:
; SSE: movntdq
; AVX: vmovntdq
; AVX2: vmovntdq
  %r = add <16 x i8> %a, %b
  store <16 x i8> %r, <16 x i8>* %dst, align 16, !nontemporal !1
  ret void
}

; And now YMM versions.

define void @test_op_v8f32(<8 x float> %a, <8 x float> %b, <8 x float>* %dst) {
; CHECK-LABEL: test_op_v8f32:
; AVX: vmovntps %ymm
; AVX2: vmovntps %ymm
  %r = fadd <8 x float> %a, %b
  store <8 x float> %r, <8 x float>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_op_v8i32(<8 x i32> %a, <8 x i32> %b, <8 x i32>* %dst) {
; CHECK-LABEL: test_op_v8i32:
; AVX: vmovntps %ymm
; AVX2: vmovntdq %ymm
  %r = add <8 x i32> %a, %b
  store <8 x i32> %r, <8 x i32>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_op_v4f64(<4 x double> %a, <4 x double> %b, <4 x double>* %dst) {
; CHECK-LABEL: test_op_v4f64:
; AVX: vmovntpd %ymm
; AVX2: vmovntpd %ymm
  %r = fadd <4 x double> %a, %b
  store <4 x double> %r, <4 x double>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_op_v4i64(<4 x i64> %a, <4 x i64> %b, <4 x i64>* %dst) {
; CHECK-LABEL: test_op_v4i64:
; AVX: vmovntps %ymm
; AVX2: vmovntdq %ymm
  %r = add <4 x i64> %a, %b
  store <4 x i64> %r, <4 x i64>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_op_v16i16(<16 x i16> %a, <16 x i16> %b, <16 x i16>* %dst) {
; CHECK-LABEL: test_op_v16i16:
; AVX: vmovntps %ymm
; AVX2: vmovntdq %ymm
  %r = add <16 x i16> %a, %b
  store <16 x i16> %r, <16 x i16>* %dst, align 32, !nontemporal !1
  ret void
}

define void @test_op_v32i8(<32 x i8> %a, <32 x i8> %b, <32 x i8>* %dst) {
; CHECK-LABEL: test_op_v32i8:
; AVX: vmovntps %ymm
; AVX2: vmovntdq %ymm
  %r = add <32 x i8> %a, %b
  store <32 x i8> %r, <32 x i8>* %dst, align 32, !nontemporal !1
  ret void
}

; 256-bit NT stores require 256-bit alignment.
; FIXME: For AVX, we could lower this to 2x movntps %xmm. Taken further, we
; could even scalarize to movnti when we have 1-alignment: nontemporal is
; probably always worth even some 20 instruction scalarization.
define void @test_unaligned_v8f32(<8 x float> %a, <8 x float> %b, <8 x float>* %dst) {
; CHECK-LABEL: test_unaligned_v8f32:
; SSE: movntps %xmm
; SSE: movntps %xmm
; AVX-NOT: movnt
; AVX: vmovups %ymm
; AVX2-NOT: movnt
; AVX2: vmovups %ymm
  %r = fadd <8 x float> %a, %b
  store <8 x float> %r, <8 x float>* %dst, align 16, !nontemporal !1
  ret void
}

!1 = !{i32 1}
