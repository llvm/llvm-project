; RUN: opt < %s -msan -msan-check-access-address=0 -S | FileCheck %s

; Test instrumentation of vector shift instructions.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare x86_mmx @llvm.x86.mmx.psll.d(x86_mmx, x86_mmx)
declare <8 x i32> @llvm.x86.avx2.psllv.d.256(<8 x i32>, <8 x i32>)
declare <4 x i32> @llvm.x86.avx2.psllv.d(<4 x i32>, <4 x i32>)
declare <8 x i16> @llvm.x86.sse2.psrl.w(<8 x i16>, <8 x i16>)
declare <8 x i16> @llvm.x86.sse2.pslli.w(<8 x i16>, i32)

define i64 @test_mmx(i64 %x.coerce, i64 %y.coerce) sanitize_memory {
entry:
  %0 = bitcast i64 %x.coerce to <2 x i32>
  %1 = bitcast <2 x i32> %0 to x86_mmx
  %2 = bitcast i64 %y.coerce to x86_mmx
  %3 = tail call x86_mmx @llvm.x86.mmx.psll.d(x86_mmx %1, x86_mmx %2)
  %4 = bitcast x86_mmx %3 to <2 x i32>
  %5 = bitcast <2 x i32> %4 to <1 x i64>
  %6 = extractelement <1 x i64> %5, i32 0
  ret i64 %6
}

; CHECK-LABEL: @test_mmx
; CHECK: = icmp ne i64 {{.*}}, 0
; CHECK: [[C:%.*]] = sext i1 {{.*}} to i64
; CHECK: [[A:%.*]] = call x86_mmx @llvm.x86.mmx.psll.d(
; CHECK: [[B:%.*]] = bitcast x86_mmx {{.*}}[[A]] to i64
; CHECK: = or i64 {{.*}}[[B]], {{.*}}[[C]]
; CHECK: call x86_mmx @llvm.x86.mmx.psll.d(
; CHECK: ret i64


define <8 x i16> @test_sse2_scalar(<8 x i16> %x, i32 %y) sanitize_memory {
entry:
  %0 = tail call <8 x i16> @llvm.x86.sse2.pslli.w(<8 x i16> %x, i32 %y)
  ret <8 x i16> %0
}

; CHECK-LABEL: @test_sse2_scalar
; CHECK: = icmp ne i32 {{.*}}, 0
; CHECK: = sext i1 {{.*}} to i128
; CHECK: = bitcast i128 {{.*}} to <8 x i16>
; CHECK: = call <8 x i16> @llvm.x86.sse2.pslli.w(
; CHECK: = or <8 x i16>
; CHECK: call <8 x i16> @llvm.x86.sse2.pslli.w(
; CHECK: ret <8 x i16>


define <8 x i16> @test_sse2(<8 x i16> %x, <8 x i16> %y) sanitize_memory {
entry:
  %0 = tail call <8 x i16> @llvm.x86.sse2.psrl.w(<8 x i16> %x, <8 x i16> %y)
  ret <8 x i16> %0
}

; CHECK-LABEL: @test_sse2
; CHECK: = bitcast <8 x i16> {{.*}} to i128
; CHECK: = trunc i128 {{.*}} to i64
; CHECK: = icmp ne i64 {{.*}}, 0
; CHECK: = sext i1 {{.*}} to i128
; CHECK: = bitcast i128 {{.*}} to <8 x i16>
; CHECK: = call <8 x i16> @llvm.x86.sse2.psrl.w(
; CHECK: = or <8 x i16>
; CHECK: call <8 x i16> @llvm.x86.sse2.psrl.w(
; CHECK: ret <8 x i16>


; Test variable shift (i.e. vector by vector).

define <4 x i32> @test_avx2(<4 x i32> %x, <4 x i32> %y) sanitize_memory {
entry:
  %0 = tail call <4 x i32> @llvm.x86.avx2.psllv.d(<4 x i32> %x, <4 x i32> %y)
  ret <4 x i32> %0
}

; CHECK-LABEL: @test_avx2
; CHECK: = icmp ne <4 x i32> {{.*}}, zeroinitializer
; CHECK: = sext <4 x i1> {{.*}} to <4 x i32>
; CHECK: = call <4 x i32> @llvm.x86.avx2.psllv.d(
; CHECK: = or <4 x i32>
; CHECK: = tail call <4 x i32> @llvm.x86.avx2.psllv.d(
; CHECK: ret <4 x i32>

define <8 x i32> @test_avx2_256(<8 x i32> %x, <8 x i32> %y) sanitize_memory {
entry:
  %0 = tail call <8 x i32> @llvm.x86.avx2.psllv.d.256(<8 x i32> %x, <8 x i32> %y)
  ret <8 x i32> %0
}

; CHECK-LABEL: @test_avx2_256
; CHECK: = icmp ne <8 x i32> {{.*}}, zeroinitializer
; CHECK: = sext <8 x i1> {{.*}} to <8 x i32>
; CHECK: = call <8 x i32> @llvm.x86.avx2.psllv.d.256(
; CHECK: = or <8 x i32>
; CHECK: = tail call <8 x i32> @llvm.x86.avx2.psllv.d.256(
; CHECK: ret <8 x i32>
