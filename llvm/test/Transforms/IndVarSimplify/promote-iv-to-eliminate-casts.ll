; RUN: opt < %s -indvars -S | FileCheck %s

; Provide legal integer types.
target datalayout = "n8:16:32:64"

; CHECK-NOT: sext

define i64 @test(i64* nocapture %first, i32 %count) nounwind readonly {
entry:
	%t0 = icmp sgt i32 %count, 0		; <i1> [#uses=1]
	br i1 %t0, label %bb.nph, label %bb2

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%result.02 = phi i64 [ %t5, %bb1 ], [ 0, %bb.nph ]		; <i64> [#uses=1]
	%n.01 = phi i32 [ %t6, %bb1 ], [ 0, %bb.nph ]		; <i32> [#uses=2]
	%t1 = sext i32 %n.01 to i64		; <i64> [#uses=1]
	%t2 = getelementptr i64, i64* %first, i64 %t1		; <i64*> [#uses=1]
	%t3 = load i64, i64* %t2, align 8		; <i64> [#uses=1]
	%t4 = lshr i64 %t3, 4		; <i64> [#uses=1]
	%t5 = add i64 %t4, %result.02		; <i64> [#uses=2]
	%t6 = add i32 %n.01, 1		; <i32> [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	%t7 = icmp slt i32 %t6, %count		; <i1> [#uses=1]
	br i1 %t7, label %bb, label %bb1.bb2_crit_edge

bb1.bb2_crit_edge:		; preds = %bb1
	%.lcssa = phi i64 [ %t5, %bb1 ]		; <i64> [#uses=1]
	br label %bb2

bb2:		; preds = %bb1.bb2_crit_edge, %entry
	%result.0.lcssa = phi i64 [ %.lcssa, %bb1.bb2_crit_edge ], [ 0, %entry ]		; <i64> [#uses=1]
	ret i64 %result.0.lcssa
}

define void @foo(i16 signext %N, i32* nocapture %P) nounwind {
entry:
	%t0 = icmp sgt i16 %N, 0		; <i1> [#uses=1]
	br i1 %t0, label %bb.nph, label %return

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%i.01 = phi i16 [ %t3, %bb1 ], [ 0, %bb.nph ]		; <i16> [#uses=2]
	%t1 = sext i16 %i.01 to i64		; <i64> [#uses=1]
	%t2 = getelementptr i32, i32* %P, i64 %t1		; <i32*> [#uses=1]
	store i32 123, i32* %t2, align 4
	%t3 = add i16 %i.01, 1		; <i16> [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	%t4 = icmp slt i16 %t3, %N		; <i1> [#uses=1]
	br i1 %t4, label %bb, label %bb1.return_crit_edge

bb1.return_crit_edge:		; preds = %bb1
	br label %return

return:		; preds = %bb1.return_crit_edge, %entry
	ret void
}

; Test cases from PR1301:

define void @kinds__srangezero([21 x i32]* nocapture %a) nounwind {
bb.thread:
  br label %bb

bb:             ; preds = %bb, %bb.thread
  %i.0.reg2mem.0 = phi i8 [ -10, %bb.thread ], [ %tmp7, %bb ]           ; <i8> [#uses=2]
  %tmp12 = sext i8 %i.0.reg2mem.0 to i32                ; <i32> [#uses=1]
  %tmp4 = add i32 %tmp12, 10            ; <i32> [#uses=1]
  %tmp5 = getelementptr [21 x i32], [21 x i32]* %a, i32 0, i32 %tmp4                ; <i32*> [#uses=1]
  store i32 0, i32* %tmp5
  %tmp7 = add i8 %i.0.reg2mem.0, 1              ; <i8> [#uses=2]
  %0 = icmp sgt i8 %tmp7, 10            ; <i1> [#uses=1]
  br i1 %0, label %return, label %bb

return:         ; preds = %bb
  ret void
}

define void @kinds__urangezero([21 x i32]* nocapture %a) nounwind {
bb.thread:
  br label %bb

bb:             ; preds = %bb, %bb.thread
  %i.0.reg2mem.0 = phi i8 [ 10, %bb.thread ], [ %tmp7, %bb ]            ; <i8> [#uses=2]
  %tmp12 = sext i8 %i.0.reg2mem.0 to i32                ; <i32> [#uses=1]
  %tmp4 = add i32 %tmp12, -10           ; <i32> [#uses=1]
  %tmp5 = getelementptr [21 x i32], [21 x i32]* %a, i32 0, i32 %tmp4                ; <i32*> [#uses=1]
  store i32 0, i32* %tmp5
  %tmp7 = add i8 %i.0.reg2mem.0, 1              ; <i8> [#uses=2]
  %0 = icmp sgt i8 %tmp7, 30            ; <i1> [#uses=1]
  br i1 %0, label %return, label %bb

return:         ; preds = %bb
  ret void
}

define void @promote_latch_condition_decrementing_loop_01(i32* %p, i32* %a) {

; CHECK-LABEL: @promote_latch_condition_decrementing_loop_01(
; CHECK-NOT:     trunc

entry:
  %len = load i32, i32* %p, align 4, !range !0
  %len.minus.1 = add nsw i32 %len, -1
  %zero_check = icmp eq i32 %len, 0
  br i1 %zero_check, label %loopexit, label %preheader

preheader:
  br label %loop

loopexit:
  ret void

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ %len.minus.1, %preheader ]
  ; CHECK: %indvars.iv = phi i64
  %iv.wide = zext i32 %iv to i64
  %el = getelementptr inbounds i32, i32* %a, i64 %iv.wide
  store atomic i32 0, i32* %el unordered, align 4
  %iv.next = add nsw i32 %iv, -1
  ; CHECK: %loopcond = icmp slt i64 %indvars.iv, 1
  %loopcond = icmp slt i32 %iv, 1
  br i1 %loopcond, label %loopexit, label %loop
}

define void @promote_latch_condition_decrementing_loop_02(i32* %p, i32* %a) {

; CHECK-LABEL: @promote_latch_condition_decrementing_loop_02(
; CHECK-NOT:     trunc

entry:
  %len = load i32, i32* %p, align 4, !range !0
  %zero_check = icmp eq i32 %len, 0
  br i1 %zero_check, label %loopexit, label %preheader

preheader:
  br label %loop

loopexit:
  ret void

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ %len, %preheader ]
  ; CHECK: %indvars.iv = phi i64
  %iv.wide = zext i32 %iv to i64
  %el = getelementptr inbounds i32, i32* %a, i64 %iv.wide
  store atomic i32 0, i32* %el unordered, align 4
  %iv.next = add nsw i32 %iv, -1
  ; CHECK: %loopcond = icmp slt i64 %indvars.iv, 1
  %loopcond = icmp slt i32 %iv, 1
  br i1 %loopcond, label %loopexit, label %loop
}

define void @promote_latch_condition_decrementing_loop_03(i32* %p, i32* %a) {

; CHECK-LABEL: @promote_latch_condition_decrementing_loop_03(
; CHECK-NOT:     trunc

entry:
  %len = load i32, i32* %p, align 4, !range !0
  %len.plus.1 = add i32 %len, 1
  %zero_check = icmp eq i32 %len, 0
  br i1 %zero_check, label %loopexit, label %preheader

preheader:
  br label %loop

loopexit:
  ret void

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ %len.plus.1, %preheader ]
  ; CHECK: %indvars.iv = phi i64
  %iv.wide = zext i32 %iv to i64
  %el = getelementptr inbounds i32, i32* %a, i64 %iv.wide
  store atomic i32 0, i32* %el unordered, align 4
  %iv.next = add nsw i32 %iv, -1
  ; CHECK: %loopcond = icmp slt i64 %indvars.iv, 1
  %loopcond = icmp slt i32 %iv, 1
  br i1 %loopcond, label %loopexit, label %loop
}


!0 = !{i32 0, i32 2147483647}
