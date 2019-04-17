; RUN: opt -constmerge -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"


; Test that with a TD we do merge and mark the alignment as 4
@T1A = internal unnamed_addr constant i32 1
@T1B = internal unnamed_addr constant i32 1, align 2
; CHECK: @T1B = internal unnamed_addr constant i32 1, align 4

define void @test1(i32** %P1, i32** %P2) {
  store i32* @T1A, i32** %P1
  store i32* @T1B, i32** %P2
  ret void
}


; Test that even with a TD we set the alignment to the maximum if both constants
; have explicit alignments.
@T2A = internal unnamed_addr constant i32 2, align 1
@T2B = internal unnamed_addr constant i32 2, align 2
; CHECK: @T2B = internal unnamed_addr constant i32 2, align 2

define void @test2(i32** %P1, i32** %P2) {
  store i32* @T2A, i32** %P1
  store i32* @T2B, i32** %P2
  ret void
}
