; RUN: opt < %s -loop-reduce -S | FileCheck %s
; rdar://7382068

define void @t(i32 %c) nounwind optsize {
entry:
  br label %bb6

bb1:                                              ; preds = %bb6
  %tmp = icmp eq i32 %c_addr.1, 20                ; <i1> [#uses=1]
  br i1 %tmp, label %bb2, label %bb3

bb2:                                              ; preds = %bb1
  %tmp1 = tail call i32 @f20(i32 %c_addr.1) nounwind ; <i32> [#uses=1]
  br label %bb7

bb3:                                              ; preds = %bb1
  %tmp2 = icmp slt i32 %c_addr.1, 10              ; <i1> [#uses=1]
  %tmp3 = add nsw i32 %c_addr.1, 1                ; <i32> [#uses=1]
  %tmp4 = add i32 %c_addr.1, -1                   ; <i32> [#uses=1]
  %c_addr.1.be = select i1 %tmp2, i32 %tmp3, i32 %tmp4 ; <i32> [#uses=1]
  %indvar.next = add i32 %indvar, 1               ; <i32> [#uses=1]
; CHECK: add i32 %lsr.iv, -1
  br label %bb6

bb6:                                              ; preds = %bb3, %entry
  %indvar = phi i32 [ %indvar.next, %bb3 ], [ 0, %entry ] ; <i32> [#uses=2]
  %c_addr.1 = phi i32 [ %c_addr.1.be, %bb3 ], [ %c, %entry ] ; <i32> [#uses=7]
  %tmp5 = icmp eq i32 %indvar, 9999               ; <i1> [#uses=1]
; CHECK: icmp eq i32 %lsr.iv, 0
  %tmp6 = icmp eq i32 %c_addr.1, 100              ; <i1> [#uses=1]
  %or.cond = or i1 %tmp5, %tmp6                   ; <i1> [#uses=1]
  br i1 %or.cond, label %bb7, label %bb1

bb7:                                              ; preds = %bb6, %bb2
  %c_addr.0 = phi i32 [ %tmp1, %bb2 ], [ %c_addr.1, %bb6 ] ; <i32> [#uses=1]
  tail call void @bar(i32 %c_addr.0) nounwind
  ret void
}

declare i32 @f20(i32)

declare void @bar(i32)
