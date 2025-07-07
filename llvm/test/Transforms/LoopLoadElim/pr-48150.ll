; RUN: opt -passes=loop-load-elim -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

define void @test(i1 %arg) {
; CHECK-LABEL: test
bb:
  br i1 %arg, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  ret void

bb2:                                              ; preds = %bb29, %bb
  %tmp = phi i64 [ %tmp30, %bb29 ], [ 2, %bb ]
  %tmp3 = add nuw nsw i64 %tmp, 99
  br label %bb4

bb4:                                              ; preds = %bb4, %bb2
  %tmp5 = phi i64 [ %tmp9, %bb4 ], [ 1, %bb2 ]
  %tmp6 = phi i32 [ %tmp10, %bb4 ], [ undef, %bb2 ]
  %tmp7 = trunc i64 %tmp5 to i32
  %tmp8 = add i32 %tmp6, %tmp7
  %tmp9 = add nuw nsw i64 %tmp5, 1
  %tmp10 = add i32 %tmp8, -1
  %tmp11 = icmp ugt i64 %tmp3, %tmp5
  br i1 %tmp11, label %bb4, label %bb12

bb12:                                             ; preds = %bb4
  %tmp13 = sext i32 %tmp8 to i64
  %tmp14 = add i64 0, %tmp13
  %tmp15 = mul i64 %tmp14, undef
  %tmp16 = add i64 %tmp15, 83
  %tmp17 = and i64 %tmp16, -2
  br label %bb18

bb18:                                             ; preds = %bb18, %bb12
  %tmp19 = phi i64 [ 0, %bb12 ], [ %tmp27, %bb18 ]
  %tmp20 = add i64 %tmp19, 3
  %tmp21 = add i64 %tmp19, 5
  %tmp22 = getelementptr inbounds i32, ptr addrspace(1) undef, i64 %tmp20
  %tmp24 = load <2 x i32>, ptr addrspace(1) %tmp22, align 4
  %tmp25 = getelementptr inbounds i32, ptr addrspace(1) undef, i64 %tmp21
  store <2 x i32> undef, ptr addrspace(1) %tmp25, align 4
  %tmp27 = add i64 %tmp19, 2
  %tmp28 = icmp eq i64 %tmp27, %tmp17
  br i1 %tmp28, label %bb29, label %bb18

bb29:                                             ; preds = %bb18
  %tmp30 = add nuw nsw i64 %tmp, 1
  br label %bb2
}
