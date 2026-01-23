; RUN: opt -passes=loop-load-elim -S %s | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

; Make sure it does not crash with assert.
define void @test(i1 %arg) {
; CHECK-LABEL: test

bb:
  br label %bb1

bb1:                                              ; preds = %bb6, %bb1, %bb
  %tmp = phi i32 [ undef, %bb ], [ 0, %bb1 ], [ %tmp3, %bb6 ]
  br i1 %arg, label %bb1, label %bb2

bb2:                                              ; preds = %bb1
  %tmp3 = add i32 %tmp, 1
  %tmp4 = icmp ult i32 %tmp, undef
  br i1 %tmp4, label %bb6, label %bb5

bb5:                                              ; preds = %bb2
  ret void

bb6:                                              ; preds = %bb2
  br i1 %arg, label %bb7, label %bb1

bb7:                                              ; preds = %bb7, %bb6
  %tmp8 = phi i32 [ %tmp15, %bb7 ], [ %tmp3, %bb6 ]
  %tmp9 = phi i32 [ %tmp8, %bb7 ], [ %tmp, %bb6 ]
  %tmp10 = zext i32 %tmp9 to i64
  %tmp11 = getelementptr inbounds float, ptr addrspace(1) null, i64 %tmp10
  %tmp12 = load float, ptr addrspace(1) %tmp11, align 4
  %tmp13 = zext i32 %tmp8 to i64
  %tmp14 = getelementptr inbounds float, ptr addrspace(1) null, i64 %tmp13
  store float 1.000000e+00, ptr addrspace(1) %tmp14, align 4
  %tmp15 = add nuw nsw i32 %tmp8, 1
  %tmp16 = icmp sgt i32 %tmp8, 78
  br i1 %tmp16, label %bb17, label %bb7

bb17:                                             ; preds = %bb7
  unreachable
}
