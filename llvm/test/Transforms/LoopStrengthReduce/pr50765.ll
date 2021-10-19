; RUN: opt -S -loop-reduce < %s | FileCheck %s
;
;This test produces zero factor that becomes a denumerator and fails an assetion.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

define void @test() {
; CHECK-LABEL: test
bb:
  %tmp = load i32, i32 addrspace(3)* undef, align 4
  br label %bb1

bb1:                                              ; preds = %bb38, %bb
  %tmp2 = phi i64 [ undef, %bb ], [ %tmp6, %bb38 ]
  %tmp3 = phi i32 [ %tmp, %bb ], [ 1, %bb38 ]
  %tmp4 = add i32 %tmp3, 1
  %tmp5 = call i32 @llvm.smax.i32(i32 %tmp4, i32 74)
  %tmp6 = add nuw nsw i64 %tmp2, 1
  br i1 undef, label %bb7, label %bb38

bb7:                                              ; preds = %bb1
  %tmp8 = trunc i64 %tmp6 to i32
  %tmp9 = sub nsw i32 3, %tmp5
  %tmp10 = mul i32 %tmp9, %tmp8
  br label %bb11

bb11:                                             ; preds = %bb11, %bb7
  %tmp12 = phi i32 [ undef, %bb7 ], [ %tmp17, %bb11 ]
  %tmp13 = phi i64 [ 3, %bb7 ], [ %tmp22, %bb11 ]
  %tmp14 = phi i64 [ undef, %bb7 ], [ %tmp23, %bb11 ]
  %tmp15 = add i32 %tmp12, %tmp10
  %tmp16 = add nuw nsw i64 %tmp13, 1
  %tmp17 = add i32 %tmp15, %tmp10
  %tmp18 = add i32 %tmp17, undef
  %tmp19 = sub i32 %tmp18, undef
  %tmp20 = sext i32 %tmp19 to i64
  %tmp21 = add nsw i64 undef, %tmp20
  %tmp22 = add nuw nsw i64 %tmp13, 2
  %tmp23 = add i64 %tmp14, -2
  %tmp24 = icmp eq i64 %tmp23, 0
  br i1 %tmp24, label %bb25, label %bb11

bb25:                                             ; preds = %bb11
  %tmp26 = trunc i64 %tmp16 to i32
  %tmp27 = icmp ult i32 %tmp26, 52
  %tmp28 = trunc i64 %tmp22 to i32
  %tmp29 = mul i32 %tmp9, %tmp28
  %tmp30 = add i32 undef, %tmp29
  %tmp31 = mul i32 %tmp30, %tmp8
  %tmp32 = add i32 undef, %tmp31
  %tmp33 = add i32 %tmp32, 34
  %tmp34 = trunc i64 %tmp21 to i32
  %tmp35 = add i32 %tmp33, undef
  %tmp36 = sub i32 %tmp35, %tmp34
  %tmp37 = sext i32 %tmp36 to i64
  unreachable

bb38:                                             ; preds = %bb1
  br label %bb1
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.smax.i32(i32, i32) #0

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }
