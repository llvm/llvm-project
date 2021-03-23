; RUN: %lli %s > /dev/null

; We were accidentally inverting the signedness of right shifts.  Whoops.

define i32 @main() {
	%X = ashr i32 -1, 16		; <i32> [#uses=1]
	%Y = ashr i32 %X, 16		; <i32> [#uses=1]
	%Z = add i32 %Y, 1		; <i32> [#uses=1]
	ret i32 %Z
}

