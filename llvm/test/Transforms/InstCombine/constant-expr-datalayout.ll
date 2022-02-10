; RUN: opt -passes=instcombine %s -S -o - | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%test1.struct = type { i32, i32 }
@test1.aligned_glbl = global %test1.struct zeroinitializer, align 4
define void @test1(i64 *%ptr) {
  store i64 and (i64 ptrtoint (i32* getelementptr (%test1.struct, %test1.struct* @test1.aligned_glbl, i32 0, i32 1) to i64), i64 3), i64* %ptr
; CHECK: store i64 0, i64* %ptr
  ret void
}
