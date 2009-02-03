; RUN: llvm-as < %s | llc | not grep 1_0
; This used to create an extra branch to 'entry', LBB1_0.

; ModuleID = 'bug.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64"
target triple = "arm-apple-darwin8"
	%struct.HexxagonMove = type { i8, i8, i32 }
	%struct.HexxagonMoveList = type { i32, %struct.HexxagonMove* }

define void @_ZN16HexxagonMoveList8sortListEv(%struct.HexxagonMoveList* %this) {
entry:
	%tmp51 = getelementptr %struct.HexxagonMoveList* %this, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp2 = getelementptr %struct.HexxagonMoveList* %this, i32 0, i32 1		; <%struct.HexxagonMove**> [#uses=2]
	br label %bb49

bb1:		; preds = %bb49
	%tmp3 = load %struct.HexxagonMove** %tmp2		; <%struct.HexxagonMove*> [#uses=5]
	%tmp6 = getelementptr %struct.HexxagonMove* %tmp3, i32 %i.1, i32 2		; <i32*> [#uses=1]
	%tmp7 = load i32* %tmp6		; <i32> [#uses=2]
	%tmp12 = add i32 %i.1, 1		; <i32> [#uses=7]
	%tmp14 = getelementptr %struct.HexxagonMove* %tmp3, i32 %tmp12, i32 2		; <i32*> [#uses=1]
	%tmp15 = load i32* %tmp14		; <i32> [#uses=1]
	%tmp16 = icmp slt i32 %tmp7, %tmp15		; <i1> [#uses=1]
	br i1 %tmp16, label %cond_true, label %bb49

cond_true:		; preds = %bb1
	%tmp23.0 = getelementptr %struct.HexxagonMove* %tmp3, i32 %i.1, i32 0		; <i8*> [#uses=2]
	%tmp67 = load i8* %tmp23.0		; <i8> [#uses=1]
	%tmp23.1 = getelementptr %struct.HexxagonMove* %tmp3, i32 %i.1, i32 1		; <i8*> [#uses=1]
	%tmp68 = load i8* %tmp23.1		; <i8> [#uses=1]
	%tmp3638 = getelementptr %struct.HexxagonMove* %tmp3, i32 %tmp12, i32 0		; <i8*> [#uses=1]
	tail call void @llvm.memcpy.i32( i8* %tmp23.0, i8* %tmp3638, i32 8, i32 4 )
	%tmp41 = load %struct.HexxagonMove** %tmp2		; <%struct.HexxagonMove*> [#uses=3]
	%tmp44.0 = getelementptr %struct.HexxagonMove* %tmp41, i32 %tmp12, i32 0		; <i8*> [#uses=1]
	store i8 %tmp67, i8* %tmp44.0
	%tmp44.1 = getelementptr %struct.HexxagonMove* %tmp41, i32 %tmp12, i32 1		; <i8*> [#uses=1]
	store i8 %tmp68, i8* %tmp44.1
	%tmp44.2 = getelementptr %struct.HexxagonMove* %tmp41, i32 %tmp12, i32 2		; <i32*> [#uses=1]
	store i32 %tmp7, i32* %tmp44.2
	br label %bb49

bb49:		; preds = %bb59, %cond_true, %bb1, %entry
	%i.1 = phi i32 [ 0, %entry ], [ %tmp12, %bb1 ], [ %tmp12, %cond_true ], [ 0, %bb59 ]		; <i32> [#uses=5]
	%move.2 = phi i32 [ 0, %entry ], [ 1, %cond_true ], [ %move.2, %bb1 ], [ 0, %bb59 ]		; <i32> [#uses=2]
	%tmp52 = load i32* %tmp51		; <i32> [#uses=1]
	%tmp53 = add i32 %tmp52, -1		; <i32> [#uses=1]
	%tmp55 = icmp sgt i32 %tmp53, %i.1		; <i1> [#uses=1]
	br i1 %tmp55, label %bb1, label %bb59

bb59:		; preds = %bb49
	%tmp61 = icmp eq i32 %move.2, 0		; <i1> [#uses=1]
	br i1 %tmp61, label %return, label %bb49

return:		; preds = %bb59
	ret void
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)
