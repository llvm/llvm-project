; RUN: opt -S -loop-reduce %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

; Make sure we don't crash.
define void @test() {
; CHECK-LABEL: @test(
bb:
  %tmp = load atomic i64, ptr addrspace(1) undef unordered, align 8
  %tmp1 = sub i64 4294967294, undef
  br label %bb5

bb2:                                              ; No predecessors!
  %tmp3 = add i32 undef, %tmp24
  unreachable

bb5:                                              ; preds = %bb5, %bb
  %tmp6 = phi i64 [ %tmp18, %bb5 ], [ %tmp, %bb ]
  %tmp7 = phi i32 [ %tmp19, %bb5 ], [ undef, %bb ]
  %tmp8 = phi i32 [ %tmp24, %bb5 ], [ undef, %bb ]
  %tmp9 = sub i32 %tmp8, undef
  %tmp10 = zext i32 %tmp9 to i64
  %tmp11 = add i32 %tmp7, 1
  %tmp12 = zext i32 %tmp11 to i64
  %tmp13 = add i64 %tmp1, %tmp12
  %tmp14 = add i64 %tmp6, %tmp10
  %tmp15 = sub i64 %tmp14, %tmp13
  %tmp16 = trunc i64 %tmp15 to i32
  %tmp17 = add i32 undef, %tmp16
  %tmp18 = add i64 %tmp6, 2
  %tmp19 = add i32 %tmp7, 2
  %tmp20 = xor i64 %tmp6, -1
  %tmp21 = add i64 %tmp1, %tmp20
  %tmp22 = trunc i64 %tmp21 to i32
  %tmp23 = add i32 %tmp19, %tmp22
  %tmp24 = add i32 %tmp17, %tmp23
  br label %bb5
}
