; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck  \
; RUN: %s
; REQUIRES: x86-registered-target

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16>, <8 x i16>) nounwind readnone
declare <1 x i64> @llvm.x86.ssse3.pmadd.ub.sw(<1 x i64>, <1 x i64>) nounwind readnone
declare <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8>, <16 x i8>) nounwind readnone
declare <1 x i64> @llvm.x86.mmx.psad.bw(<1 x i64>, <1 x i64>) nounwind readnone

define <4 x i32> @Test_sse2_pmadd_wd(<8 x i16> %a, <8 x i16> %b) sanitize_memory {
entry:
  %c = tail call <4 x i32> @llvm.x86.sse2.pmadd.wd(<8 x i16> %a, <8 x i16> %b) nounwind
  ret <4 x i32> %c
}

; CHECK-LABEL: @Test_sse2_pmadd_wd(
; CHECK: or <8 x i16>
; CHECK: bitcast <8 x i16> {{.*}} to <4 x i32>
; CHECK: icmp ne <4 x i32> {{.*}}, zeroinitializer
; CHECK: sext <4 x i1> {{.*}} to <4 x i32>
; CHECK: ret <4 x i32>


define <1 x i64> @Test_ssse3_pmadd_ub_sw(<1 x i64> %a, <1 x i64> %b) sanitize_memory {
entry:
  %c = tail call <1 x i64> @llvm.x86.ssse3.pmadd.ub.sw(<1 x i64> %a, <1 x i64> %b) nounwind
  ret <1 x i64> %c
}

; CHECK-LABEL: @Test_ssse3_pmadd_ub_sw(
; CHECK: or <1 x i64>
; CHECK: bitcast <1 x i64> {{.*}} to <4 x i16>
; CHECK: icmp ne <4 x i16> {{.*}}, zeroinitializer
; CHECK: sext <4 x i1> {{.*}} to <4 x i16>
; CHECK: bitcast <4 x i16> {{.*}} to <1 x i64>
; CHECK: ret <1 x i64>


define <2 x i64> @Test_x86_sse2_psad_bw(<16 x i8> %a, <16 x i8> %b) sanitize_memory {
  %c = tail call <2 x i64> @llvm.x86.sse2.psad.bw(<16 x i8> %a, <16 x i8> %b)
  ret <2 x i64> %c
}

; CHECK-LABEL: @Test_x86_sse2_psad_bw(
; CHECK: or <16 x i8> {{.*}}, {{.*}}
; CHECK: bitcast <16 x i8> {{.*}} to <2 x i64>
; CHECK: icmp ne <2 x i64> {{.*}}, zeroinitializer
; CHECK: sext <2 x i1> {{.*}} to <2 x i64>
; CHECK: lshr <2 x i64> {{.*}}, <i64 48, i64 48>
; CHECK: ret <2 x i64>


define <1 x i64> @Test_x86_mmx_psad_bw(<1 x i64> %a, <1 x i64> %b) sanitize_memory {
entry:
  %c = tail call <1 x i64> @llvm.x86.mmx.psad.bw(<1 x i64> %a, <1 x i64> %b) nounwind
  ret <1 x i64> %c
}

; CHECK-LABEL: @Test_x86_mmx_psad_bw(
; CHECK: or <1 x i64>
; CHECK: icmp ne i64
; CHECK: sext i1 {{.*}} to i64
; CHECK: lshr i64 {{.*}}, 48
; CHECK: ret <1 x i64>
