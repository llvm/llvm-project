; RUN: %lli -mtriple=%mcjit_triple -use-mcjit %s > /dev/null

define i32 @bar() {
	ret i32 0
}

define i32 @main() {
	%r = call i32 @bar( )		; <i32> [#uses=1]
	ret i32 %r
}

