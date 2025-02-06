; RUN: llc < %s -mtriple=aarch64-linux-gnu -mattr=-neon | FileCheck %s

%structA = type { i128 }
@stubA = internal unnamed_addr constant %structA zeroinitializer, align 8

; Make sure we don't hit llvm_unreachable.

define void @test1() {
; CHECK-LABEL: @test1
; CHECK: ret
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 undef, ptr align 8 @stubA, i64 48, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1)
