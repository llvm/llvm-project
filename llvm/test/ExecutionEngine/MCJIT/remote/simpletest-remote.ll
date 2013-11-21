; RUN: %lli_mcjit -remote-mcjit -mcjit-remote-process=lli-child-target %s > /dev/null

define i32 @bar() {
	ret i32 0
}

define i32 @main() {
	%r = call i32 @bar( )		; <i32> [#uses=1]
	ret i32 %r
}
