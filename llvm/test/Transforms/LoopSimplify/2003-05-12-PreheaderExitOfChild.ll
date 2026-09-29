; This (complex) testcase causes an assertion failure because a preheader is
; inserted for the "fail" loop, but the exit block of a loop is not updated
; to be the preheader instead of the exit loop itself.

; RUN: opt < %s -passes=loop-simplify
define i32 @re_match_2() {
	br label %loopentry.1
loopentry.1:		; preds = %endif.82, %0
	br label %shortcirc_done.36
shortcirc_done.36:		; preds = %loopentry.1
	br i1 false, label %fail, label %endif.40
endif.40:		; preds = %shortcirc_done.36
	br label %loopexit.20
loopentry.20:		; preds = %endif.46
	br label %loopexit.20
loopexit.20:		; preds = %loopentry.20, %endif.40
	br label %loopentry.21
loopentry.21:		; preds = %no_exit.19, %loopexit.20
	br i1 false, label %no_exit.19, label %loopexit.21
no_exit.19:		; preds = %loopentry.21
	br i1 false, label %fail, label %loopentry.21
loopexit.21:		; preds = %loopentry.21
	br label %endif.45
endif.45:		; preds = %loopexit.21
	br label %cond_true.15
cond_true.15:		; preds = %endif.45
	br i1 false, label %fail, label %endif.46
endif.46:		; preds = %cond_true.15
	br label %loopentry.20
fail:		; preds = %loopexit.37, %cond_true.15, %no_exit.19, %shortcirc_done.36
	br label %then.80
then.80:		; preds = %fail
	br label %endif.81
endif.81:		; preds = %then.80
	br label %loopexit.37
loopexit.37:		; preds = %endif.81
	br i1 false, label %fail, label %endif.82
endif.82:		; preds = %loopexit.37
	br label %loopentry.1
}


