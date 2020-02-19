; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s

%structA = type { i128 }
@stubA = internal unnamed_addr constant %structA zeroinitializer, align 8

; Make sure we don't hit llvm_unreachable.

define void @test1() {
; CHECK-LABEL: @test1
; CHECK: ret
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 undef, i8* align 8 bitcast (%structA* @stubA to i8*), i64 48, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1)
