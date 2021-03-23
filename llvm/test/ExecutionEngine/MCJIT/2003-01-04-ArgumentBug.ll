; RUN: %lli %s > /dev/null

define i32 @foo(i32 %X, i32 %Y, double %A) {
	%cond212 = fcmp une double %A, 1.000000e+00		; <i1> [#uses=1]
	%cast110 = zext i1 %cond212 to i32		; <i32> [#uses=1]
	ret i32 %cast110
}

define i32 @main() {
	%reg212 = call i32 @foo( i32 0, i32 1, double 1.000000e+00 )		; <i32> [#uses=1]
	ret i32 %reg212
}

