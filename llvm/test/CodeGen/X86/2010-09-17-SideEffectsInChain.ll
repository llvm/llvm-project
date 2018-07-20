; RUN: llc < %s -mcpu=core2 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.4"
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

define fastcc i32 @cli_magic_scandesc(i8* %in) nounwind ssp {
entry:
  %a = alloca [64 x i8]
  %b = getelementptr inbounds [64 x i8], [64 x i8]* %a, i64 0, i32 0
  %c = getelementptr inbounds [64 x i8], [64 x i8]* %a, i64 0, i32 30
  %d = load i8, i8* %b, align 8
  %e = load i8, i8* %c, align 8
  %f = bitcast [64 x i8]* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %f, i8* align 8 %in, i64 64, i1 false) nounwind
  store i8 %d, i8* %b, align 8
  store i8 %e, i8* %c, align 8
  ret i32 0
}

; CHECK: movq	___stack_chk_guard@GOTPCREL(%rip)
; CHECK: movb   (%rsp), [[R1:%.+]]
; CHECK: movb   30(%rsp), [[R0:%.+]]
; CHECK: movb   [[R1]], (%rsp)
; CHECK: movb   [[R0]], 30(%rsp)
; CHECK: callq	___stack_chk_fail
