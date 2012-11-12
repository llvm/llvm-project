; RUN: llc < %s -mtriple=i386-apple-darwin10.0 -relocation-model=pic | FileCheck %s

define void @dot(i16* nocapture %A, i32 %As, i16* nocapture %B, i32 %Bs, i16* nocapture %C, i32 %N) nounwind ssp {
; CHECK: dot:
; CHECK: decl %
; CHECK-NEXT: jne
entry:
	%0 = icmp sgt i32 %N, 0		; <i1> [#uses=1]
	br i1 %0, label %bb, label %bb2

bb:		; preds = %bb, %entry
	%i.03 = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=3]
	%sum.04 = phi i32 [ 0, %entry ], [ %10, %bb ]		; <i32> [#uses=1]
	%1 = mul i32 %i.03, %As		; <i32> [#uses=1]
	%2 = getelementptr i16* %A, i32 %1		; <i16*> [#uses=1]
	%3 = load i16* %2, align 2		; <i16> [#uses=1]
	%4 = sext i16 %3 to i32		; <i32> [#uses=1]
	%5 = mul i32 %i.03, %Bs		; <i32> [#uses=1]
	%6 = getelementptr i16* %B, i32 %5		; <i16*> [#uses=1]
	%7 = load i16* %6, align 2		; <i16> [#uses=1]
	%8 = sext i16 %7 to i32		; <i32> [#uses=1]
	%9 = mul i32 %8, %4		; <i32> [#uses=1]
	%10 = add i32 %9, %sum.04		; <i32> [#uses=2]
	%indvar.next = add i32 %i.03, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %N		; <i1> [#uses=1]
	br i1 %exitcond, label %bb1.bb2_crit_edge, label %bb

bb1.bb2_crit_edge:		; preds = %bb
	%phitmp = trunc i32 %10 to i16		; <i16> [#uses=1]
	br label %bb2

bb2:		; preds = %entry, %bb1.bb2_crit_edge
	%sum.0.lcssa = phi i16 [ %phitmp, %bb1.bb2_crit_edge ], [ 0, %entry ]		; <i16> [#uses=1]
	store i16 %sum.0.lcssa, i16* %C, align 2
	ret void
}
