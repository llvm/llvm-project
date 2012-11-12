; Though this case seems to be fairly unlikely to occur in the wild, someone
; plunked it into the demo script, so maybe they care about it.
;
; RUN: opt < %s -tailcallelim -S | not grep call

define i32 @aaa(i32 %c) {
entry:
	%tmp.1 = icmp eq i32 %c, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %return, label %else
else:		; preds = %entry
	%tmp.5 = add i32 %c, -1		; <i32> [#uses=1]
	%tmp.3 = call i32 @aaa( i32 %tmp.5 )		; <i32> [#uses=0]
	ret i32 0
return:		; preds = %entry
	ret i32 0
}

