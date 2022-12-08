; RUN: opt -S -passes=indvars -indvars-predicate-loops=0  < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

@a = common global i32 0, align 4

; Function Attrs: nounwind ssp uwtable
define void @fn1() {
; CHECK-LABEL: @fn1(
bb:
  br label %bb4.preheader

bb4.preheader:                                    ; preds = %bb, %bb16
; CHECK-LABEL:  bb4.preheader:
  %b.03 = phi i8 [ 0, %bb ], [ %tmp17, %bb16 ]
; CHECK: %tmp9 = icmp ugt i8 %b.03, 1
; CHECK-NOT: %tmp9 = icmp ugt i8 0, 1

  %tmp9 = icmp ugt i8 %b.03, 1
  br i1 %tmp9, label %bb4.preheader.bb18.loopexit.split_crit_edge, label %bb4.preheader.bb4.preheader.split_crit_edge

bb4.preheader.bb4.preheader.split_crit_edge:      ; preds = %bb4.preheader
  br label %bb4.preheader.split

bb4.preheader.bb18.loopexit.split_crit_edge:      ; preds = %bb4.preheader
  store i32 0, i32* @a, align 4
  br label %bb18.loopexit.split

bb4.preheader.split:                              ; preds = %bb4.preheader.bb4.preheader.split_crit_edge
  br label %bb7

bb4:                                              ; preds = %bb7
  %tmp6 = icmp slt i32 %storemerge2, 0
  br i1 %tmp6, label %bb7, label %bb16

bb7:                                              ; preds = %bb4.preheader.split, %bb4
  %storemerge2 = phi i32 [ 0, %bb4.preheader.split ], [ %tmp14, %bb4 ]
  %tmp14 = add nsw i32 %storemerge2, 1
  br i1 false, label %bb18.loopexit, label %bb4

bb16:                                             ; preds = %bb4
  %tmp14.lcssa5 = phi i32 [ %tmp14, %bb4 ]
  %tmp17 = add i8 %b.03, -1
  %tmp2 = icmp eq i8 %tmp17, -2
  br i1 %tmp2, label %bb18.loopexit1, label %bb4.preheader

bb18.loopexit:                                    ; preds = %bb7
  br label %bb18.loopexit.split

bb18.loopexit.split:                              ; preds = %bb4.preheader.bb18.loopexit.split_crit_edge, %bb18.loopexit
  br label %bb18

bb18.loopexit1:                                   ; preds = %bb16
  %tmp14.lcssa5.lcssa = phi i32 [ %tmp14.lcssa5, %bb16 ]
  store i32 %tmp14.lcssa5.lcssa, i32* @a, align 4
  br label %bb18

bb18:                                             ; preds = %bb18.loopexit1, %bb18.loopexit.split
  ret void
}

declare void @abort()
