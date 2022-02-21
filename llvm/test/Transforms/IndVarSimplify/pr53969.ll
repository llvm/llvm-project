; RUN: opt -passes="loop(indvars,loop-deletion)" -S  < %s | FileCheck %s
; XFAIL: *
; REQUIRES: asserts

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

; Make sure we don't crash.
define void @test() {
; CHECK-LABEL: test
bb:
  br label %bb1

bb1:                                              ; preds = %bb31, %bb
  %tmp = phi i32 [ %tmp29, %bb31 ], [ undef, %bb ]
  %tmp2 = phi i32 [ %tmp4, %bb31 ], [ 11, %bb ]
  %tmp3 = add nsw i32 112, -1
  %tmp4 = add nuw nsw i32 %tmp2, 1
  %tmp5 = mul i32 %tmp3, %tmp3
  %tmp6 = mul nsw i32 %tmp2, -6
  %tmp7 = mul i32 %tmp6, %tmp5
  %tmp8 = add i32 %tmp7, %tmp2
  %tmp9 = and i32 undef, 1
  %tmp10 = icmp eq i32 %tmp9, 0
  br i1 %tmp10, label %bb33, label %bb34

bb11:                                             ; preds = %bb34
  br i1 undef, label %bb33, label %bb34

bb12:                                             ; preds = %bb34
  %tmp13 = icmp eq i8 addrspace(1)* undef, null
  br label %bb14

bb14:                                             ; preds = %bb25, %bb12
  %tmp15 = phi i32 [ %tmp29, %bb25 ], [ %tmp37, %bb12 ]
  %tmp16 = phi i64 [ undef, %bb25 ], [ %tmp41, %bb12 ]
  %tmp17 = phi i32 [ %tmp26, %bb25 ], [ 4, %bb12 ]
  %tmp18 = add i64 %tmp16, undef
  %tmp19 = add i32 %tmp15, 1
  %tmp20 = and i32 %tmp19, 1
  %tmp21 = icmp eq i32 %tmp20, 0
  br i1 %tmp21, label %bb32, label %bb22

bb22:                                             ; preds = %bb14
  %tmp23 = or i32 %tmp17, undef
  %tmp24 = add i32 %tmp23, undef
  br i1 %tmp13, label %bb42, label %bb25

bb25:                                             ; preds = %bb22
  %tmp26 = add nuw nsw i32 %tmp17, 1
  %tmp27 = zext i32 %tmp26 to i64
  %tmp28 = getelementptr inbounds i32, i32 addrspace(1)* undef, i64 %tmp27
  %tmp29 = add i32 %tmp15, 3
  %tmp30 = icmp ugt i32 %tmp17, 110
  br i1 %tmp30, label %bb31, label %bb14

bb31:                                             ; preds = %bb25
  br label %bb1

bb32:                                             ; preds = %bb14
  ret void

bb33:                                             ; preds = %bb11, %bb1
  call void @use(i32 %tmp2)
  ret void

bb34:                                             ; preds = %bb11, %bb1
  %tmp35 = phi i32 [ %tmp37, %bb11 ], [ %tmp, %bb1 ]
  %tmp36 = xor i32 0, %tmp8
  %tmp37 = add i32 %tmp35, 2
  %tmp38 = add i32 %tmp36, undef
  %tmp39 = add i32 %tmp38, undef
  %tmp40 = sext i32 %tmp39 to i64
  %tmp41 = add i64 undef, %tmp40
  br i1 undef, label %bb11, label %bb12

bb42:                                             ; preds = %bb22
  store atomic i64 %tmp18, i64 addrspace(1)* undef unordered, align 8
  call void @use(i32 %tmp24)
  ret void
}

declare void @use(i32)
