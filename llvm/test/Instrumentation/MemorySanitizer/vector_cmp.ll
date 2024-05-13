; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck  \
; RUN: %s
; REQUIRES: x86-registered-target

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare <4 x float> @llvm.x86.sse.cmp.ss(<4 x float>, <4 x float>, i8) nounwind readnone
declare <4 x float> @llvm.x86.sse.cmp.ps(<4 x float>, <4 x float>, i8) nounwind readnone
declare <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double>, <2 x double>, i8) nounwind readnone
declare i32 @llvm.x86.sse.comineq.ss(<4 x float>, <4 x float>) nounwind readnone
declare i32 @llvm.x86.sse2.ucomilt.sd(<2 x double>, <2 x double>) nounwind readnone


define <4 x float> @test_sse_cmp_ss(<4 x float> %a, <4 x float> %b) sanitize_memory {
entry:
  %0 = tail call <4 x float> @llvm.x86.sse.cmp.ss(<4 x float> %a, <4 x float> %b, i8 4)
  ret <4 x float> %0
}

; CHECK-LABEL: @test_sse_cmp_ss
; CHECK: %[[A:.*]] = or <4 x i32>
; CHECK: %[[B:.*]] = extractelement <4 x i32> %[[A]], i64 0
; CHECK: %[[C:.*]] = icmp ne i32 %[[B]], 0
; CHECK: %[[D:.*]] = sext i1 %[[C]] to i128
; CHECK: %[[E:.*]] = bitcast i128 %[[D]] to <4 x i32>
; CHECK: store <4 x i32> %[[E]]


define <4 x float> @test_sse_cmp_ps(<4 x float> %a, <4 x float> %b) sanitize_memory {
entry:
  %0 = tail call <4 x float> @llvm.x86.sse.cmp.ps(<4 x float> %a, <4 x float> %b, i8 4)
  ret <4 x float> %0
}

; CHECK-LABEL: @test_sse_cmp_ps
; CHECK: %[[A:.*]] = or <4 x i32>
; CHECK: %[[B:.*]] = icmp ne <4 x i32> %[[A]], zeroinitializer
; CHECK: %[[C:.*]] = sext <4 x i1> %[[B]] to <4 x i32>
; CHECK: store <4 x i32> %[[C]]


define <2 x double> @test_sse2_cmp_sd(<2 x double> %a, <2 x double> %b) sanitize_memory {
entry:
  %0 = tail call <2 x double> @llvm.x86.sse2.cmp.sd(<2 x double> %a, <2 x double> %b, i8 4)
  ret <2 x double> %0
}

; CHECK-LABEL: @test_sse2_cmp_sd
; CHECK: %[[A:.*]] = or <2 x i64>
; CHECK: %[[B:.*]] = extractelement <2 x i64> %[[A]], i64 0
; CHECK: %[[C:.*]] = icmp ne i64 %[[B]], 0
; CHECK: %[[D:.*]] = sext i1 %[[C]] to i128
; CHECK: %[[E:.*]] = bitcast i128 %[[D]] to <2 x i64>
; CHECK: store <2 x i64> %[[E]]


define i32 @test_sse_comineq_ss(<4 x float> %a, <4 x float> %b) sanitize_memory {
entry:
  %0 = tail call i32 @llvm.x86.sse.comineq.ss(<4 x float> %a, <4 x float> %b)
  ret i32 %0
}

; CHECK-LABEL: @test_sse_comineq_ss
; CHECK: %[[A:.*]] = or <4 x i32>
; CHECK: %[[B:.*]] = extractelement <4 x i32> %[[A]], i64 0
; CHECK: %[[C:.*]] = icmp ne i32 %[[B]], 0
; CHECK: %[[D:.*]] = sext i1 %[[C]] to i32
; CHECK: store i32 %[[D]]


define i32 @test_sse2_ucomilt_sd(<2 x double> %a, <2 x double> %b) sanitize_memory {
entry:
  %0 = tail call i32 @llvm.x86.sse2.ucomilt.sd(<2 x double> %a, <2 x double> %b)
  ret i32 %0
}

; CHECK-LABEL: @test_sse2_ucomilt_sd
; CHECK: %[[A:.*]] = or <2 x i64>
; CHECK: %[[B:.*]] = extractelement <2 x i64> %[[A]], i64 0
; CHECK: %[[C:.*]] = icmp ne i64 %[[B]], 0
; CHECK: %[[D:.*]] = sext i1 %[[C]] to i32
; CHECK: store i32 %[[D]]
