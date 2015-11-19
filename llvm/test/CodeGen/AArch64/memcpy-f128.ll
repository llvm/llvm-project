; RUN: llc < %s -march=aarch64 -mtriple=aarch64-linux-gnu | FileCheck %s

%structA = type { i128 }
@stubA = internal unnamed_addr constant %structA zeroinitializer, align 8

; Make sure we don't hit llvm_unreachable.

define void @test1() {
; CHECK-LABEL: @test1
; CHECK: adrp
; CHECK: ldr q0
; CHECK: str q0
; CHECK: ret
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* undef, i8* bitcast (%structA* @stubA to i8*), i64 48, i32 8, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1)
