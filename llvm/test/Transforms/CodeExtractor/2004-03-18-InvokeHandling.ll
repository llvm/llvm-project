; RUN: opt < %s -passes='function(loop-simplify),loop-extract'  -disable-output

declare i32 @_IO_getc()

declare void @__errno_location()

define void @yylex() personality i32 (...)* @__gcc_personality_v0 {
entry:
	switch i32 0, label %label.126 [
		 i32 0, label %return
		 i32 61, label %combine
		 i32 33, label %combine
		 i32 94, label %combine
		 i32 37, label %combine
		 i32 47, label %combine
		 i32 42, label %combine
		 i32 62, label %combine
		 i32 60, label %combine
		 i32 58, label %combine
		 i32 124, label %combine
		 i32 38, label %combine
		 i32 45, label %combine
		 i32 43, label %combine
		 i32 34, label %string_constant
		 i32 39, label %char_constant
		 i32 46, label %loopexit.2
		 i32 57, label %loopexit.2
		 i32 56, label %loopexit.2
		 i32 55, label %loopexit.2
		 i32 54, label %loopexit.2
		 i32 53, label %loopexit.2
		 i32 52, label %loopexit.2
		 i32 51, label %loopexit.2
		 i32 50, label %loopexit.2
		 i32 49, label %loopexit.2
		 i32 48, label %loopexit.2
		 i32 95, label %letter
		 i32 122, label %letter
		 i32 121, label %letter
		 i32 120, label %letter
		 i32 119, label %letter
		 i32 118, label %letter
		 i32 117, label %letter
		 i32 116, label %letter
		 i32 115, label %letter
		 i32 114, label %letter
		 i32 113, label %letter
		 i32 112, label %letter
		 i32 111, label %letter
		 i32 110, label %letter
		 i32 109, label %letter
		 i32 108, label %letter
		 i32 107, label %letter
		 i32 106, label %letter
		 i32 105, label %letter
		 i32 104, label %letter
		 i32 103, label %letter
		 i32 102, label %letter
		 i32 101, label %letter
		 i32 100, label %letter
		 i32 99, label %letter
		 i32 98, label %letter
		 i32 97, label %letter
		 i32 90, label %letter
		 i32 89, label %letter
		 i32 88, label %letter
		 i32 87, label %letter
		 i32 86, label %letter
		 i32 85, label %letter
		 i32 84, label %letter
		 i32 83, label %letter
		 i32 82, label %letter
		 i32 81, label %letter
		 i32 80, label %letter
		 i32 79, label %letter
		 i32 78, label %letter
		 i32 77, label %letter
		 i32 75, label %letter
		 i32 74, label %letter
		 i32 73, label %letter
		 i32 72, label %letter
		 i32 71, label %letter
		 i32 70, label %letter
		 i32 69, label %letter
		 i32 68, label %letter
		 i32 67, label %letter
		 i32 66, label %letter
		 i32 65, label %letter
		 i32 64, label %label.13
		 i32 76, label %label.12
		 i32 36, label %label.11
		 i32 -1, label %label.10
	]

label.10:		; preds = %entry
	ret void

label.11:		; preds = %entry
	ret void

label.12:		; preds = %entry
	ret void

label.13:		; preds = %entry
	ret void

letter:		; preds = %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry
	ret void

loopexit.2:		; preds = %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry
	switch i32 0, label %shortcirc_next.14 [
		 i32 48, label %then.20
		 i32 46, label %endif.38
	]

then.20:		; preds = %loopexit.2
	switch i32 0, label %else.4 [
		 i32 120, label %then.21
		 i32 88, label %then.21
	]

then.21:		; preds = %then.20, %then.20
	ret void

else.4:		; preds = %then.20
	ret void

shortcirc_next.14:		; preds = %loopexit.2
	ret void

endif.38:		; preds = %loopexit.2
	br i1 false, label %then.40, label %then.39

then.39:		; preds = %endif.38
	ret void

then.40:		; preds = %endif.38
	invoke void @__errno_location( )
			to label %switchexit.2 unwind label %LongJmpBlkPre

loopentry.6:		; preds = %endif.52
	switch i32 0, label %switchexit.2 [
		 i32 73, label %label.82
		 i32 105, label %label.82
		 i32 76, label %label.80
		 i32 108, label %label.80
		 i32 70, label %label.78
		 i32 102, label %label.78
	]

label.78:		; preds = %loopentry.6, %loopentry.6
	ret void

label.80:		; preds = %loopentry.6, %loopentry.6
	ret void

label.82:		; preds = %loopentry.6, %loopentry.6
	%c.0.15.5 = phi i32 [ %tmp.79417, %loopentry.6 ], [ %tmp.79417, %loopentry.6 ]		; <i32> [#uses=0]
	ret void

switchexit.2:		; preds = %loopentry.6, %then.40
	br i1 false, label %endif.51, label %loopexit.6

endif.51:		; preds = %switchexit.2
	br i1 false, label %endif.52, label %then.52

then.52:		; preds = %endif.51
	ret void

endif.52:		; preds = %endif.51
	%tmp.79417 = invoke i32 @_IO_getc( )
			to label %loopentry.6 unwind label %LongJmpBlkPre		; <i32> [#uses=2]

loopexit.6:		; preds = %switchexit.2
	ret void

char_constant:		; preds = %entry
	ret void

string_constant:		; preds = %entry
	ret void

combine:		; preds = %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry
	ret void

label.126:		; preds = %entry
	ret void

return:		; preds = %entry
	ret void

LongJmpBlkPre:		; preds = %endif.52, %then.40
        %exn = landingpad { i8*, i32 }
                 catch i8* null
	ret void
}

declare i32 @__gcc_personality_v0(...)
