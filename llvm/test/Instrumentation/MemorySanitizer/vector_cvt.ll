; RUN: opt < %s -msan -msan-check-access-address=0 -S | FileCheck %s
; REQUIRES: x86-registered-target

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @llvm.x86.sse2.cvtsd2si(<2 x double>) nounwind readnone
declare <2 x double> @llvm.x86.sse2.cvtsi2sd(<2 x double>, i32) nounwind readnone
declare x86_mmx @llvm.x86.sse.cvtps2pi(<4 x float>) nounwind readnone

; Single argument vector conversion.

define i32 @test_cvtsd2si(<2 x double> %value) sanitize_memory {
entry:
  %0 = tail call i32 @llvm.x86.sse2.cvtsd2si(<2 x double> %value)
  ret i32 %0
}

; CHECK-LABEL: @test_cvtsd2si
; CHECK: [[S:%[_01-9a-z]+]] = extractelement <2 x i64> {{.*}}, i32 0
; CHECK: icmp ne {{.*}}[[S]], 0
; CHECK: br
; CHECK: call void @__msan_warning_noreturn
; CHECK: call i32 @llvm.x86.sse2.cvtsd2si
; CHECK: store i32 0, {{.*}} @__msan_retval_tls
; CHECK: ret i32

; x86_mmx packed vector conversion.

define x86_mmx @test_cvtps2pi(<4 x float> %value) sanitize_memory {
entry:
  %0 = tail call x86_mmx @llvm.x86.sse.cvtps2pi(<4 x float> %value)
  ret x86_mmx %0
}

; CHECK-LABEL: @test_cvtps2pi
; CHECK: extractelement <4 x i32> {{.*}}, i32 0
; CHECK: extractelement <4 x i32> {{.*}}, i32 1
; CHECK: [[S:%[_01-9a-z]+]] = or i32
; CHECK: icmp ne {{.*}}[[S]], 0
; CHECK: br
; CHECK: call void @__msan_warning_noreturn
; CHECK: call x86_mmx @llvm.x86.sse.cvtps2pi
; CHECK: store i64 0, {{.*}} @__msan_retval_tls
; CHECK: ret x86_mmx
