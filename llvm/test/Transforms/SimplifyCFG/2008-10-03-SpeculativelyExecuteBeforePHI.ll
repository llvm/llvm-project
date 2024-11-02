; RUN: opt < %s -passes=simplifycfg
; PR2855

define i32 @_Z1fPii(i32* %b, i32 %f) nounwind {
entry:
	br label %bb

bb:		; preds = %bb9, %bb7, %bb, %entry
	%__c2.2 = phi i32 [ undef, %entry ], [ %__c2.1, %bb7 ], [ %__c2.1, %bb9 ]		; <i32> [#uses=2]
	%s.0 = phi i32 [ 0, %entry ], [ 0, %bb7 ], [ %2, %bb9 ]		; <i32> [#uses=1]
	br label %bb1

bb1:		; preds = %bb
	%0 = icmp slt i32 0, %f		; <i1> [#uses=1]
	br i1 %0, label %bb3, label %bb6

bb3:		; preds = %bb1
	%1 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb6, label %bb5

bb5:		; preds = %bb3
	br label %bb7

bb6:		; preds = %bb3, %bb1
	%__c2.0 = phi i32 [ 0, %bb3 ], [ %__c2.2, %bb1 ]		; <i32> [#uses=1]
	br label %bb7

bb7:		; preds = %bb6, %bb5
	%__c2.1 = phi i32 [ 0, %bb5 ], [ %__c2.0, %bb6 ]		; <i32> [#uses=2]
	%iftmp.1.0 = phi i1 [ false, %bb5 ], [ true, %bb6 ]		; <i1> [#uses=1]
	br i1 %iftmp.1.0, label %bb, label %bb9

bb9:		; preds = %bb7
	%2 = add i32 %s.0, 2		; <i32> [#uses=1]
	br label %bb
}
