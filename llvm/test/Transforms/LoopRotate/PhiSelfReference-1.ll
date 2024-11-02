; RUN: opt < %s -passes=loop-rotate -verify-dom-info -verify-loop-info -verify-memoryssa -disable-output
; ModuleID = 'PhiSelfReference-1.bc'

define void @snrm2(i32 %incx) {
entry:
	br i1 false, label %START, label %return

START:		; preds = %entry
	br i1 false, label %bb85, label %cond_false93

bb52:		; preds = %bb85
	br i1 false, label %bb307, label %cond_next71

cond_next71:		; preds = %bb52
	ret void

bb85:		; preds = %START
	br i1 false, label %bb52, label %bb88

bb88:		; preds = %bb85
	ret void

cond_false93:		; preds = %START
	ret void

bb243:		; preds = %bb307
	br label %bb307

bb307:		; preds = %bb243, %bb52
	%sx_addr.2.pn = phi float* [ %sx_addr.5, %bb243 ], [ null, %bb52 ]		; <float*> [#uses=1]
	%sx_addr.5 = getelementptr float, float* %sx_addr.2.pn, i32 %incx		; <float*> [#uses=1]
	br i1 false, label %bb243, label %bb310

bb310:		; preds = %bb307
	ret void

return:		; preds = %entry
	ret void
}
