; RUN: llc < %s  -tailcallopt -mtriple=powerpc-apple-darwin -relocation-model=pic | grep TC_RETURN



define protected fastcc i32 @tailcallee(i32 %a1, i32 %a2, i32 %a3, i32 %a4) {
entry:
	ret i32 %a3
}

define fastcc i32 @tailcaller(i32 %in1, i32 %in2) {
entry:
	%tmp11 = tail call fastcc i32 @tailcallee( i32 %in1, i32 %in2, i32 %in1, i32 %in2 )		; <i32> [#uses=1]
	ret i32 %tmp11
}
