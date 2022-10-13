; RUN: opt < %s -passes=indvars -disable-output

declare void @q_atomic_increment()

declare void @_Z9qt_assertPKcS0_i()

define void @_ZN13QMetaResourceC1EPKh() personality i32 (...)* @__gxx_personality_v0 {
entry:
	invoke void @_Z9qt_assertPKcS0_i( )
			to label %endif.1 unwind label %then.i.i551

then.i.i551:		; preds = %entry
        %exn551 = landingpad {i8*, i32}
                 cleanup
	ret void

endif.1:		; preds = %entry
	br i1 false, label %then.2, label %then.i.i

then.2:		; preds = %endif.1
	invoke void @q_atomic_increment( )
			to label %loopentry.0 unwind label %invoke_catch.6

invoke_catch.6:		; preds = %then.2
        %exn6 = landingpad {i8*, i32}
                 cleanup
	ret void

loopentry.0:		; preds = %then.2
	br i1 false, label %shortcirc_next.i, label %endif.3

endif.3:		; preds = %loopentry.0
	ret void

shortcirc_next.i:		; preds = %loopentry.0
	br i1 false, label %_ZNK7QString2atEi.exit, label %then.i

then.i:		; preds = %shortcirc_next.i
	ret void

_ZNK7QString2atEi.exit:		; preds = %shortcirc_next.i
	br i1 false, label %endif.4, label %then.4

then.4:		; preds = %_ZNK7QString2atEi.exit
	ret void

endif.4:		; preds = %_ZNK7QString2atEi.exit
	%tmp.115 = load i8, i8* null		; <i8> [#uses=1]
	br i1 false, label %loopexit.1, label %no_exit.0

no_exit.0:		; preds = %no_exit.0, %endif.4
	%bytes_in_len.4.5 = phi i8 [ %dec, %no_exit.0 ], [ %tmp.115, %endif.4 ]		; <i8> [#uses=1]
	%off.5.5.in = phi i32 [ %off.5.5, %no_exit.0 ], [ 0, %endif.4 ]		; <i32> [#uses=1]
	%off.5.5 = add i32 %off.5.5.in, 1		; <i32> [#uses=2]
	%dec = add i8 %bytes_in_len.4.5, -1		; <i8> [#uses=2]
	%tmp.123631 = icmp eq i8 %dec, 0		; <i1> [#uses=1]
	br i1 %tmp.123631, label %loopexit.1, label %no_exit.0

loopexit.1:		; preds = %no_exit.0, %endif.4
	%off.5.in.6 = phi i32 [ 0, %endif.4 ], [ %off.5.5, %no_exit.0 ]		; <i32> [#uses=0]
	ret void

then.i.i:		; preds = %endif.1
	ret void
}

declare i32 @__gxx_personality_v0(...)
