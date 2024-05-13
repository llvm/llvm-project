; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck  \
; RUN: %s
; REQUIRES: x86-registered-target

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare <8 x i16> @llvm.x86.sse2.packssdw.128(<4 x i32>, <4 x i32>) nounwind readnone
declare <32 x i8> @llvm.x86.avx2.packuswb(<16 x i16> %a, <16 x i16> %b) nounwind readnone
declare x86_mmx @llvm.x86.mmx.packuswb(x86_mmx, x86_mmx) nounwind readnone

define <8 x i16> @Test_packssdw_128(<4 x i32> %a, <4 x i32> %b) sanitize_memory {
entry:
  %c = tail call <8 x i16> @llvm.x86.sse2.packssdw.128(<4 x i32> %a, <4 x i32> %b) nounwind
  ret <8 x i16> %c
}

; CHECK-LABEL: @Test_packssdw_128(
; CHECK-DAG: icmp ne <4 x i32> {{.*}}, zeroinitializer
; CHECK-DAG: sext <4 x i1> {{.*}} to <4 x i32>
; CHECK-DAG: icmp ne <4 x i32> {{.*}}, zeroinitializer
; CHECK-DAG: sext <4 x i1> {{.*}} to <4 x i32>
; CHECK-DAG: call <8 x i16> @llvm.x86.sse2.packssdw.128(
; CHECK-DAG: call <8 x i16> @llvm.x86.sse2.packssdw.128(
; CHECK: ret <8 x i16>


define <32 x i8> @Test_avx_packuswb(<16 x i16> %a, <16 x i16> %b) sanitize_memory {
entry:
  %c = tail call <32 x i8> @llvm.x86.avx2.packuswb(<16 x i16> %a, <16 x i16> %b) nounwind
  ret <32 x i8> %c
}

; CHECK-LABEL: @Test_avx_packuswb(
; CHECK-DAG: icmp ne <16 x i16> {{.*}}, zeroinitializer
; CHECK-DAG: sext <16 x i1> {{.*}} to <16 x i16>
; CHECK-DAG: icmp ne <16 x i16> {{.*}}, zeroinitializer
; CHECK-DAG: sext <16 x i1> {{.*}} to <16 x i16>
; CHECK-DAG: call <32 x i8> @llvm.x86.avx2.packsswb(
; CHECK-DAG: call <32 x i8> @llvm.x86.avx2.packuswb(
; CHECK: ret <32 x i8>


define x86_mmx @Test_mmx_packuswb(x86_mmx %a, x86_mmx %b) sanitize_memory {
entry:
  %c = tail call x86_mmx @llvm.x86.mmx.packuswb(x86_mmx %a, x86_mmx %b) nounwind
  ret x86_mmx %c
}

; CHECK-LABEL: @Test_mmx_packuswb(
; CHECK-DAG: bitcast i64 {{.*}} to <4 x i16>
; CHECK-DAG: bitcast i64 {{.*}} to <4 x i16>
; CHECK-DAG: icmp ne <4 x i16> {{.*}}, zeroinitializer
; CHECK-DAG: sext <4 x i1> {{.*}} to <4 x i16>
; CHECK-DAG: icmp ne <4 x i16> {{.*}}, zeroinitializer
; CHECK-DAG: sext <4 x i1> {{.*}} to <4 x i16>
; CHECK-DAG: bitcast <4 x i16> {{.*}} to x86_mmx
; CHECK-DAG: bitcast <4 x i16> {{.*}} to x86_mmx
; CHECK-DAG: call x86_mmx @llvm.x86.mmx.packsswb({{.*}}
; CHECK-DAG: bitcast x86_mmx {{.*}} to i64
; CHECK-DAG: call x86_mmx @llvm.x86.mmx.packuswb({{.*}}
; CHECK: ret x86_mmx
